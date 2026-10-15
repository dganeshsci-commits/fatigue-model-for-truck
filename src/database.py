"""
Local SQLite Time-Series and Event Database for Raspberry Pi 5.
Stores high-frequency telemetry and critical safety events for historical reports.
Implements asynchronous batch writing to protect SD card / NVMe write endurance.
"""

import sqlite3
import os
import time
import threading
import queue


class TelemetryDatabase:
    """
    Thread-safe SQLite storage for real-time and historical truck telematics.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(base_dir, "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "truck_telemetry.db")

        self.db_path = db_path
        self.write_queue = queue.Queue(maxsize=5000)
        self.running = True

        self._init_tables()
        self._writer_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
        self._writer_thread.start()

    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                iso_time TEXT,
                device_id TEXT,
                accel_x REAL,
                accel_y REAL,
                accel_z REAL,
                gyro_x REAL,
                gyro_y REAL,
                gyro_z REAL,
                fuel_level REAL,
                fuel_litres REAL,
                driver_card TEXT,
                driver_name TEXT,
                fatigue_score INTEGER,
                fatigue_state TEXT,
                speed_kmh REAL,
                latitude REAL,
                longitude REAL,
                driving_state TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                iso_time TEXT,
                event_type TEXT,
                severity TEXT,
                fatigue_score INTEGER,
                driving_state TEXT,
                latitude REAL,
                longitude REAL,
                driver_name TEXT
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry_logs (timestamp);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON safety_events (timestamp);")
        conn.commit()
        conn.close()

    def log_payload(self, payload: dict):
        """Enqueue payload for batched asynchronous write."""
        try:
            self.write_queue.put_nowait(("telemetry", payload))
        except queue.Full:
            pass

    def log_safety_event(self, event_type: str, severity: str, payload: dict):
        """Log significant safety alert (Harsh Braking, Drowsiness, etc.)."""
        try:
            self.write_queue.put_nowait(("event", (event_type, severity, payload)))
        except queue.Full:
            pass

    def _batch_writer_loop(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        batch = []
        last_commit = time.time()

        while self.running:
            try:
                item = self.write_queue.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                pass

            now = time.time()
            if len(batch) >= 50 or (batch and now - last_commit >= 1.0):
                for item_type, data in batch:
                    if item_type == "telemetry":
                        iso_t = data.get("timestamp_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                        imu = data.get("imu", {})
                        fuel = data.get("fuel", {})
                        rfid = data.get("rfid", {})
                        fatigue = data.get("fatigue", {})
                        gps = data.get("gps", {})
                        events = data.get("events", {})

                        cur.execute("""
                            INSERT INTO telemetry_logs (
                                timestamp, iso_time, device_id, accel_x, accel_y, accel_z,
                                gyro_x, gyro_y, gyro_z, fuel_level, fuel_litres,
                                driver_card, driver_name, fatigue_score, fatigue_state,
                                speed_kmh, latitude, longitude, driving_state
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            data.get("timestamp_epoch", now), iso_t, data.get("device_id", "truck_01"),
                            imu.get("accel_x", 0), imu.get("accel_y", 0), imu.get("accel_z", 0),
                            imu.get("gyro_x", 0), imu.get("gyro_y", 0), imu.get("gyro_z", 0),
                            fuel.get("level_percent", 0), fuel.get("volume_litres", 0),
                            rfid.get("card_id", ""), rfid.get("driver_name", ""),
                            fatigue.get("score", 0), fatigue.get("state", "NORMAL"),
                            gps.get("speed_kmh", 0), gps.get("latitude", 0), gps.get("longitude", 0),
                            events.get("driving_state", "normal")
                        ))

                    elif item_type == "event":
                        evt_name, severity, p = data
                        iso_t = p.get("timestamp_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ"))
                        gps = p.get("gps", {})
                        fatigue = p.get("fatigue", {})
                        events = p.get("events", {})
                        rfid = p.get("rfid", {})

                        cur.execute("""
                            INSERT INTO safety_events (
                                timestamp, iso_time, event_type, severity, fatigue_score,
                                driving_state, latitude, longitude, driver_name
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            now, iso_t, evt_name, severity, fatigue.get("score", 0),
                            events.get("driving_state", "normal"),
                            gps.get("latitude", 0), gps.get("longitude", 0),
                            rfid.get("driver_name", "")
                        ))

                conn.commit()
                batch.clear()
                last_commit = now

        conn.close()

    def get_recent_history(self, limit: int = 100) -> list:
        """Fetch latest records for dashboard graphs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows[::-1]

    def close(self):
        self.running = False
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)

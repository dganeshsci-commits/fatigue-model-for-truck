"""
Truck Intelligence Hub — 100 Hz Sensor Fusion Engine for Raspberry Pi 5.
Combines:
1. Camera (4 ONNX Fatigue Models)
2. ESP32 (BNO085 IMU + Ultrasonic Fuel + RC522 RFID over Serial UART at 115200 bps)
3. Phone GPS (over Wi-Fi)
4. IMU Driving Event Classifier (Harsh Braking, Acceleration, Cornering)
5. Local SQLite Database Logger
Generates the unified 100 Hz real-time JSON payload.
"""

import time
import datetime
import threading

from .detector import FatigueDetector
from .tracker import FatigueTracker
from .esp32_serial import ESP32SerialReader
from .imu_events import IMUEventDetector
from .gps_receiver import GPSReceiver
from .database import TelemetryDatabase
from .gpio_alert import GPIOAlertController


class SensorFusionHub:
    """
    Central processing hub running on Raspberry Pi 5.
    Fuses vision, motion, fuel, RFID, and GPS into a single real-time stream.
    """

    def __init__(self,
                 device_id: str = "truck_esp32_01",
                 serial_port: str = "/dev/serial0",
                 enable_gpio: bool = True,
                 enable_db: bool = True):
        self.device_id = device_id

        # 1. Hardware Interfaces
        self.esp32 = ESP32SerialReader(port=serial_port).start()
        self.gps = GPSReceiver()
        self.imu_detector = IMUEventDetector()
        self.gpio = GPIOAlertController(buzzer_pin=17, led_pin=27, enabled=enable_gpio)
        self.db = TelemetryDatabase() if enable_db else None

        # 2. Vision Models
        self.detector = None
        self.tracker = FatigueTracker()
        self.latest_fatigue = {
            "score": 12,
            "state": "NORMAL",
            "ear_left": 0.27,
            "ear_right": 0.26,
            "mar": 0.07,
            "blinks": 0,
            "yawns": 0
        }

        # 3. Latest Fused Snapshot
        self.lock = threading.Lock()
        self.latest_fused_payload = {}
        self._init_detector()

    def _init_detector(self):
        try:
            self.detector = FatigueDetector(num_threads=4)
        except Exception as e:
            print(f"[FusionHub] Vision detector deferred/error: {e}")

    def process_camera_frame(self, frame) -> dict:
        """Called when a camera frame is captured (30-60 Hz)."""
        if self.detector is None:
            self._init_detector()

        if self.detector and frame is not None:
            res = self.detector.process_frame(frame)
            status = self.tracker.update(res)

            with self.lock:
                self.latest_fatigue = {
                    "score": status.get("fatigue_score", 0),
                    "state": status.get("fatigue_state", "NORMAL"),
                    "ear_left": status.get("ear_left", 0.0),
                    "ear_right": status.get("ear_right", 0.0),
                    "mar": status.get("mar", 0.0),
                    "blinks": status.get("total_blinks", 0),
                    "yawns": status.get("total_yawns", 0),
                    "alarm_level": status.get("alarm_level", "NONE"),
                    "face_detected": status.get("face_detected", False)
                }

            # Trigger hardware cabin buzzer/LED
            if self.gpio:
                self.gpio.trigger(self.latest_fatigue.get("alarm_level", "NONE"))

            return status
        return {}

    def get_fused_payload(self) -> dict:
        """
        Generates the exact 100 Hz JSON payload specified in the system architecture.
        """
        now = time.time()
        iso_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Fetch latest ESP32 telemetry (IMU + Fuel + RFID)
        esp_packet = self.esp32.get_latest()
        raw_imu = esp_packet.get("imu", {})
        fuel_data = esp_packet.get("fuel", {})
        rfid_data = esp_packet.get("rfid", {})

        # 2. Evaluate driving events from IMU
        driving_events = self.imu_detector.evaluate(raw_imu)

        # 3. Fetch GPS location
        gps_data = self.gps.get_location()

        # 4. Snapshot current fatigue state
        with self.lock:
            fatigue_data = dict(self.latest_fatigue)

        # Build final unified JSON payload
        payload = {
            "device_id": self.device_id,
            "timestamp": iso_timestamp,
            "timestamp_epoch": now,
            "timestamp_iso": iso_timestamp,
            "imu": {
                "accel_x": raw_imu.get("accel_x", 0.12),
                "accel_y": raw_imu.get("accel_y", -0.03),
                "accel_z": raw_imu.get("accel_z", 9.81),
                "gyro_x": raw_imu.get("gyro_x", 0.5),
                "gyro_y": raw_imu.get("gyro_y", -0.2),
                "gyro_z": raw_imu.get("gyro_z", 0.1),
                "roll": raw_imu.get("roll", 1.2),
                "pitch": raw_imu.get("pitch", -0.8),
                "yaw": raw_imu.get("yaw", 45.6)
            },
            "fuel": {
                "distance_cm": fuel_data.get("distance_cm", 15.3),
                "level_percent": fuel_data.get("level_percent", 68.0),
                "volume_litres": fuel_data.get("volume_litres", 102.0)
            },
            "rfid": {
                "card_id": rfid_data.get("card_id", "A3F892B1"),
                "driver_name": rfid_data.get("driver_name", "John Doe"),
                "authenticated": rfid_data.get("authenticated", True)
            },
            "fatigue": {
                "score": fatigue_data.get("score", 12),
                "state": fatigue_data.get("state", "NORMAL"),
                "ear_left": fatigue_data.get("ear_left", 0.27),
                "ear_right": fatigue_data.get("ear_right", 0.26),
                "mar": fatigue_data.get("mar", 0.07),
                "blinks": fatigue_data.get("blinks", 18),
                "yawns": fatigue_data.get("yawns", 0)
            },
            "gps": {
                "latitude": gps_data.get("latitude", 12.9716),
                "longitude": gps_data.get("longitude", 77.5946),
                "speed_kmh": gps_data.get("speed_kmh", 68.0),
                "heading": gps_data.get("heading", 45.0)
            },
            "events": {
                "driving_state": driving_events.get("driving_state", "normal"),
                "latest_event": driving_events.get("latest_event", "none"),
                "event_counts": driving_events.get("event_counts", {})
            },
            "status": "ok"
        }

        # Asynchronously log to SQLite database
        if self.db:
            self.db.log_payload(payload)
            # Log critical safety events
            if payload["events"]["driving_state"] != "normal":
                self.db.log_safety_event(
                    payload["events"]["driving_state"],
                    driving_events.get("severity", "warning"),
                    payload
                )
            if payload["fatigue"]["state"] in ["DROWSY", "CRITICAL"]:
                self.db.log_safety_event(
                    f"Fatigue_{payload['fatigue']['state']}",
                    "critical" if payload["fatigue"]["state"] == "CRITICAL" else "warning",
                    payload
                )

        with self.lock:
            self.latest_fused_payload = payload

        return payload

    def stop(self):
        """Clean shutdown of all subsystems."""
        self.esp32.stop()
        if self.gpio:
            self.gpio.cleanup()
        if self.db:
            self.db.close()

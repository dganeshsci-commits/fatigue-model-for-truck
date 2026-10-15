"""
ESP32 High-Speed UART Serial Interface for Raspberry Pi 5 (115200 bps).
Reads real-time 100 Hz sensor packets from ESP32:
- BNO085 9-DOF IMU (accel_x/y/z, gyro_x/y/z, roll/pitch/yaw)
- Ultrasonic Fuel Level (distance_cm, level_percent, volume_litres)
- RC522 RFID Driver Authentication (card_id, driver_name, authenticated)
Includes hardware auto-detection with realistic physics fallback if unplugged.
"""

import threading
import time
import json
import math
import random

try:
    import serial
except ImportError:
    serial = None


class ESP32SerialReader:
    """
    Asynchronous threaded reader for ESP32 UART communication at 115200 baud.
    """

    def __init__(self, port="/dev/serial0", baudrate=115200, simulated_fallback=True):
        self.port = port
        self.baudrate = baudrate
        self.simulated_fallback = simulated_fallback

        self.ser = None
        self.is_connected = False
        self.running = False
        self._thread = None
        self.lock = threading.Lock()

        # Latest decoded sensor packet
        self.latest_data = {
            "device_id": "truck_esp32_01",
            "timestamp": time.time(),
            "imu": {
                "accel_x": 0.12,
                "accel_y": -0.03,
                "accel_z": 9.81,
                "gyro_x": 0.5,
                "gyro_y": -0.2,
                "gyro_z": 0.1,
                "roll": 1.2,
                "pitch": -0.8,
                "yaw": 45.6
            },
            "fuel": {
                "distance_cm": 15.3,
                "level_percent": 68.0,
                "volume_litres": 102.0
            },
            "rfid": {
                "card_id": "A3F892B1",
                "driver_name": "John Doe",
                "authenticated": True
            },
            "source": "SIMULATED"
        }

        self._try_connect()

    def _try_connect(self):
        if serial is None:
            self.is_connected = False
            return

        candidates = [self.port, "/dev/ttyAMA0", "/dev/ttyUSB0", "/dev/ttyACM0", "COM3", "COM4"]
        for p in candidates:
            try:
                self.ser = serial.Serial(p, self.baudrate, timeout=0.1)
                self.port = p
                self.is_connected = True
                print(f"[ESP32 Serial] Connected to hardware port: {self.port} at {self.baudrate} baud")
                return
            except Exception:
                continue

        self.is_connected = False
        if self.simulated_fallback:
            print("[ESP32 Serial] No physical ESP32 UART port found. Running with high-precision physics simulation.")

    def start(self):
        """Starts background ingestion thread."""
        if self.running:
            return self
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self):
        sim_step = 0
        while self.running:
            if self.is_connected and self.ser:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{') and line.endswith('}'):
                        parsed = json.loads(line)
                        with self.lock:
                            self.latest_data = {
                                **parsed,
                                "source": "ESP32_HARDWARE_UART"
                            }
                        continue
                except Exception:
                    # Serial read error or disconnection
                    self.is_connected = False
                    try:
                        self.ser.close()
                    except Exception:
                        pass

            # Simulation fallback: generate realistic 100 Hz truck physics
            sim_step += 1
            t = time.time()
            vibe_x = math.sin(t * 12.0) * 0.15 + (random.random() - 0.5) * 0.05
            vibe_y = math.cos(t * 10.0) * 0.12 + (random.random() - 0.5) * 0.05
            vibe_z = 9.81 + math.sin(t * 8.0) * 0.18

            gyro_z = math.sin(t * 0.5) * 0.8
            roll = math.sin(t * 0.8) * 1.5
            pitch = math.cos(t * 0.6) * 1.2
            yaw = (t * 2.0) % 360.0

            fuel_dist = 15.3 + math.sin(t * 0.1) * 0.4
            fuel_pct = max(0.0, min(100.0, ((40.0 - fuel_dist) / 35.0) * 100.0))
            fuel_litres = fuel_pct * 1.5

            with self.lock:
                self.latest_data = {
                    "device_id": "truck_esp32_01",
                    "timestamp": t,
                    "imu": {
                        "accel_x": round(vibe_x, 3),
                        "accel_y": round(vibe_y, 3),
                        "accel_z": round(vibe_z, 3),
                        "gyro_x": round(math.sin(t) * 0.5, 3),
                        "gyro_y": round(math.cos(t) * 0.3, 3),
                        "gyro_z": round(gyro_z, 3),
                        "roll": round(roll, 1),
                        "pitch": round(pitch, 1),
                        "yaw": round(yaw, 1)
                    },
                    "fuel": {
                        "distance_cm": round(fuel_dist, 1),
                        "level_percent": round(fuel_pct, 1),
                        "volume_litres": round(fuel_litres, 1)
                    },
                    "rfid": {
                        "card_id": "A3F892B1",
                        "driver_name": "John Doe",
                        "authenticated": True
                    },
                    "source": "SIMULATED"
                }

            # 10ms delay = 100 Hz simulation rate
            time.sleep(0.01)

    def get_latest(self) -> dict:
        """Returns the latest sensor packet."""
        with self.lock:
            return dict(self.latest_data)

    def stop(self):
        """Stops background thread."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.ser and self.ser.is_open:
            self.ser.close()

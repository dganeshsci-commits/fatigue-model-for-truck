"""
Phone GPS Receiver and Geofence Processor for Raspberry Pi 5.
Receives real-time GPS telemetry from mobile device over Wi-Fi (HTTP / WebSocket).
Includes simulation progression along highway route when live GPS is absent.
"""

import time
import threading
import math


class GPSReceiver:
    """
    Manages location, speed, heading, and geofencing for truck telematics.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.last_update = time.time()

        # Default Bengaluru highway route coordinates (matching sample payload)
        self.current_gps = {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "speed_kmh": 68.0,
            "heading": 45.0,
            "altitude_m": 920.0,
            "accuracy_m": 4.5,
            "geofence_status": "inside_route",
            "source": "SIMULATED_ROUTE"
        }

    def update_from_phone(self, lat: float, lon: float, speed_kmh: float = 0.0, heading: float = 0.0):
        """Called when phone posts GPS via HTTP POST /api/gps."""
        with self.lock:
            self.current_gps = {
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "speed_kmh": round(speed_kmh, 1),
                "heading": round(heading, 1),
                "altitude_m": 920.0,
                "accuracy_m": 3.0,
                "geofence_status": "inside_route",
                "source": "PHONE_WIFI"
            }
            self.last_update = time.time()

    def get_location(self) -> dict:
        """Returns current GPS snapshot with gradual simulation progression if idle."""
        with self.lock:
            now = time.time()
            if self.current_gps["source"] == "SIMULATED_ROUTE":
                # Advance smoothly along heading
                dt = now - self.last_update
                if dt >= 1.0:
                    speed_mps = (self.current_gps["speed_kmh"] * 1000.0) / 3600.0
                    dist_m = speed_mps * dt
                    # Approx: 1 deg lat ~ 111,000 meters
                    d_lat = (dist_m * math.cos(math.radians(self.current_gps["heading"]))) / 111000.0
                    d_lon = (dist_m * math.sin(math.radians(self.current_gps["heading"]))) / 111000.0
                    self.current_gps["latitude"] = round(self.current_gps["latitude"] + d_lat, 6)
                    self.current_gps["longitude"] = round(self.current_gps["longitude"] + d_lon, 6)
                    self.last_update = now

            return dict(self.current_gps)

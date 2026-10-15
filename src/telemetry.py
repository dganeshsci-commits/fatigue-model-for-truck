"""
Raspberry Pi 5 System Telemetry and Performance Monitor.
Gathers CPU temperature, system load, memory usage, and inference FPS.
"""

import os
import time
import psutil


class SystemTelemetry:
    """Monitors Raspberry Pi 5 board metrics."""

    def __init__(self):
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

    def update_fps(self) -> float:
        """Call once per processed frame to calculate rolling FPS."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = round(self.fps_frame_count / elapsed, 1)
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        return self.current_fps

    @staticmethod
    def get_cpu_temp() -> float:
        """Reads Raspberry Pi 5 Broadcom BCM2712 SoC temperature."""
        thermal_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal_path):
            try:
                with open(thermal_path, "r") as f:
                    temp_raw = f.read().strip()
                return round(float(temp_raw) / 1000.0, 1)
            except Exception:
                pass
        return 0.0

    @staticmethod
    def get_system_stats() -> dict:
        """Returns overall CPU and RAM load."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_temp": SystemTelemetry.get_cpu_temp()
            }
        except Exception:
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "cpu_temp": 0.0
            }

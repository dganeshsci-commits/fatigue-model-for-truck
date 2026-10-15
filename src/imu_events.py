"""
Harsh Driving Event Detection Module using BNO085 IMU.
Classifies:
- Harsh Braking
- Harsh Acceleration
- Harsh Cornering
- Excessive Vibration
- Normal Driving
"""

import time
import math


class IMUEventDetector:
    """
    Detects driving behavior anomalies from BNO085 accelerometer and gyroscope data.
    Uses calibrated thresholds and temporal persistence to eliminate road noise.
    """

    def __init__(self,
                 brake_threshold: float = -3.5,     # m/s^2 (longitudinal deceleration)
                 accel_threshold: float = 3.2,      # m/s^2 (longitudinal acceleration)
                 corner_threshold: float = 3.0,     # m/s^2 (lateral acceleration)
                 yaw_rate_threshold: float = 24.0,  # deg/s
                 vibration_threshold: float = 4.5): # m/s^2 RMS variance
        self.brake_threshold = brake_threshold
        self.accel_threshold = accel_threshold
        self.corner_threshold = corner_threshold
        self.yaw_rate_threshold = yaw_rate_threshold
        self.vibration_threshold = vibration_threshold

        # Event statistics
        self.current_state = "normal"
        self.latest_event = "none"
        self.event_severity = "low"
        self.last_event_time = 0.0

        self.event_counts = {
            "harsh_braking": 0,
            "harsh_acceleration": 0,
            "harsh_cornering": 0,
            "excessive_vibration": 0
        }

        # Rolling window for persistence
        self._consecutive_brake = 0
        self._consecutive_accel = 0
        self._consecutive_corner = 0

    def evaluate(self, imu_data: dict) -> dict:
        """
        Evaluates a single IMU frame and returns classified driving state and event summary.
        """
        ax = float(imu_data.get("accel_x", 0.0))
        ay = float(imu_data.get("accel_y", 0.0))
        az = float(imu_data.get("accel_z", 9.81))
        gz = abs(float(imu_data.get("gyro_z", 0.0)))

        now = time.time()
        detected_event = "none"
        severity = "none"

        # 1. Harsh Braking (Assumes X or Y is longitudinal based on truck mounting)
        # Using negative longitudinal acceleration
        if ax <= self.brake_threshold:
            self._consecutive_brake += 1
            if self._consecutive_brake >= 3:
                detected_event = "harsh_braking"
                severity = "critical" if ax <= -5.0 else "warning"
        else:
            self._consecutive_brake = max(0, self._consecutive_brake - 1)

        # 2. Harsh Acceleration
        if ax >= self.accel_threshold:
            self._consecutive_accel += 1
            if self._consecutive_accel >= 3:
                detected_event = "harsh_acceleration"
                severity = "warning"
        else:
            self._consecutive_accel = max(0, self._consecutive_accel - 1)

        # 3. Harsh Cornering (Lateral acceleration or high yaw rate)
        if abs(ay) >= self.corner_threshold or gz >= self.yaw_rate_threshold:
            self._consecutive_corner += 1
            if self._consecutive_corner >= 3:
                detected_event = "harsh_cornering"
                severity = "critical" if abs(ay) >= 4.5 or gz >= 35.0 else "warning"
        else:
            self._consecutive_corner = max(0, self._consecutive_corner - 1)

        # 4. Excessive Vibration (Z-axis variance from 1G)
        z_dev = abs(az - 9.81)
        if z_dev >= self.vibration_threshold and detected_event == "none":
            detected_event = "excessive_vibration"
            severity = "warning"

        # Update event counts on rising edge
        if detected_event != "none":
            self.current_state = detected_event
            if self.latest_event != detected_event or (now - self.last_event_time > 2.0):
                self.event_counts[detected_event] = self.event_counts.get(detected_event, 0) + 1
                self.latest_event = detected_event
                self.event_severity = severity
                self.last_event_time = now
        else:
            self.current_state = "normal"
            if now - self.last_event_time > 3.0:
                self.latest_event = "none"
                self.event_severity = "none"

        return {
            "driving_state": self.current_state,
            "latest_event": self.latest_event,
            "severity": self.event_severity,
            "event_counts": dict(self.event_counts)
        }

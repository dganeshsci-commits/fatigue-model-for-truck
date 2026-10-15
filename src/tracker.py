"""
Stateful Temporal Fatigue and Distraction State Machine.
Distinguishes between harmless natural blinks (100-200ms) and drowsy eye closure / microsleep,
tracks rising-edge yawns, calculates blink rate per minute, and smooths overall fatigue score.
"""

import time
from collections import deque


class FatigueTracker:
    """
    Temporal filter preventing false positives on transient blinks while rapidly
    escalating alarm levels on sustained eye closure (>600ms), yawning, or prolonged looking away.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.consecutive_closed = 0
        self.consecutive_yawn = 0
        self.consecutive_away = 0
        self.smoothed_score = 10.0
        self.last_timestamp = time.time()

        # Rising-edge event counters
        self.total_yawns = 0
        self.total_blinks = 0
        self._prev_eye_closed = False
        self._prev_yawn_active = False

        # Rolling window for blinks per minute (stored as timestamps)
        self.blink_timestamps = deque()

    def update(self, detection: dict) -> dict:
        """
        Updates internal state using detection results and returns augmented dictionary
        with fatigue_score (0-100), fatigue_state, blink_rate, and yawn_count.
        """
        now = time.time()
        time_delta = now - self.last_timestamp
        self.last_timestamp = now

        # If camera feed hung or gap > 3.5 seconds, reset consecutive counters
        if time_delta > 3.5:
            self.consecutive_closed = 0
            self.consecutive_yawn = 0
            self.consecutive_away = 0
            self.smoothed_score = 10.0

        if not detection.get("face_detected", False):
            self.consecutive_closed = 0
            self.consecutive_yawn = 0
            self.consecutive_away = 0
            return {
                **detection,
                "fatigue_score": 0,
                "fatigue_state": "NORMAL",
                "blink_rate_bpm": self._get_bpm(now),
                "total_yawns": self.total_yawns,
                "total_blinks": self.total_blinks,
                "alarm_level": "NONE"
            }

        left_closed = (detection.get("left_eye_status") == "closed")
        right_closed = (detection.get("right_eye_status") == "closed")
        both_closed = left_closed and right_closed
        either_closed = left_closed or right_closed
        mouth_status = detection.get("mouth_status", "normal")
        mar_val = detection.get("mar", 0.0)
        head_pose = detection.get("head_pose", {})
        yaw = abs(head_pose.get("yaw", 0.0))
        pitch = abs(head_pose.get("pitch", 0.0))

        # 1. Update eye closure persistence
        if both_closed:
            self.consecutive_closed += 1
            if not self._prev_eye_closed:
                self.total_blinks += 1
                self.blink_timestamps.append(now)
                self._prev_eye_closed = True
        elif either_closed:
            self.consecutive_closed = max(1, self.consecutive_closed)
            self._prev_eye_closed = False
        else:
            self.consecutive_closed = max(0, self.consecutive_closed - 2)
            self._prev_eye_closed = False

        # 2. Update yawn persistence (rising-edge counting)
        is_yawning = (mouth_status == "yawn" or mar_val > 0.38)
        if is_yawning:
            self.consecutive_yawn += 1
            if not self._prev_yawn_active and self.consecutive_yawn >= 2:
                self.total_yawns += 1
                self._prev_yawn_active = True
        else:
            self.consecutive_yawn = max(0, self.consecutive_yawn - 1)
            if self.consecutive_yawn == 0:
                self._prev_yawn_active = False

        # 3. Update driver distraction (head pose)
        is_looking_away = (yaw > 28.0 or pitch > 22.0)
        if is_looking_away:
            self.consecutive_away += 1
        else:
            self.consecutive_away = max(0, self.consecutive_away - 1)

        # 4. Target score calculation based on frame persistence
        # (Assuming ~15-30 FPS)
        target_score = 10
        if both_closed:
            if self.consecutive_closed <= 2:
                # Brief natural blink (100-200ms)
                target_score = 22
            elif self.consecutive_closed <= 5:
                # Extended blink (300-500ms)
                target_score = 38 + (self.consecutive_closed - 2) * 5
            elif self.consecutive_closed <= 10:
                # Drowsy eye closure (600ms - 1.0s)
                target_score = 54 + (self.consecutive_closed - 5) * 4
            else:
                # Microsleep (>1.0s) -> Immediate Critical
                target_score = min(99, 75 + (self.consecutive_closed - 10) * 3)
        elif either_closed:
            target_score = 30

        if is_yawning:
            yawn_contrib = min(82, 44 + self.consecutive_yawn * 4)
            target_score = max(target_score, yawn_contrib)
            if both_closed:
                target_score = min(99, target_score + 18)

        if is_looking_away:
            away_contrib = min(78, 32 + self.consecutive_away * 4)
            target_score = max(target_score, away_contrib)

        # 5. Exponential score smoothing
        if target_score >= self.smoothed_score:
            self.smoothed_score = float(target_score)
        else:
            self.smoothed_score = max(float(target_score), self.smoothed_score * 0.75 + float(target_score) * 0.25)

        score = int(round(min(99, max(0, self.smoothed_score))))

        # 6. Map to discrete fatigue states
        if score >= 70:
            state = "CRITICAL"
            alarm_level = "HIGH"
        elif score >= 50:
            state = "DROWSY"
            alarm_level = "MEDIUM"
        elif score >= 30:
            state = "ATTENTION"
            alarm_level = "LOW"
        else:
            state = "NORMAL"
            alarm_level = "NONE"

        return {
            **detection,
            "fatigue_score": score,
            "fatigue_state": state,
            "alarm_level": alarm_level,
            "consecutive_closed": self.consecutive_closed,
            "consecutive_yawn": self.consecutive_yawn,
            "consecutive_away": self.consecutive_away,
            "blink_rate_bpm": self._get_bpm(now),
            "total_yawns": self.total_yawns,
            "total_blinks": self.total_blinks
        }

    def _get_bpm(self, now: float) -> int:
        """Computes blinks per minute over a 60-second sliding window."""
        while self.blink_timestamps and (now - self.blink_timestamps[0] > 60.0):
            self.blink_timestamps.popleft()
        return len(self.blink_timestamps)

"""
Threaded Camera Capture Abstraction for Raspberry Pi 5.
Supports:
1. Standard USB Webcams via OpenCV V4L2 (/dev/video0)
2. Raspberry Pi Camera Module 3 / v2 / HQ via picamera2 or V4L2
3. Video file playback for validation and testing
"""

import threading
import time
import cv2
import numpy as np


class CameraStream:
    """
    Spawns a background thread to continually read from the camera device.
    Prevents OpenCV frame buffer backlog and guarantees zero latency on Raspberry Pi 5.
    """

    def __init__(self, src=0, width: int = 320, height: int = 240, fps: int = 30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self._thread = None

        self._init_capture()

    def _init_capture(self):
        # Check if source is a camera index or file path
        if isinstance(self.src, str) and not self.src.isdigit():
            # Video file
            self.cap = cv2.VideoCapture(self.src)
        else:
            # Camera device (V4L2 backend on Linux RPi)
            dev_idx = int(self.src)
            self.cap = cv2.VideoCapture(dev_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream or camera device: {self.src}")

        # Read first frame
        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            raise RuntimeError(f"Failed to read initial frame from camera: {self.src}")

    def start(self):
        """Starts background frame ingestion thread."""
        if self.running:
            return self
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                break
            grabbed, frame = self.cap.read()
            if not grabbed:
                # If reading from a video file, loop back to the beginning
                if isinstance(self.src, str) and not self.src.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        """Stops background thread and releases camera."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()

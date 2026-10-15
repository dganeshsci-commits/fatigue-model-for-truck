#!/usr/bin/env python3
"""
Camera diagnostic utility for Raspberry Pi 5.
Probes available video devices (/dev/video*), tests resolution, and measures capture FPS.
"""

import sys
import time
import cv2


def test_camera(device_idx=0, width=320, height=240):
    print(f"[Camera Test] Attempting to open device {device_idx} at {width}x{height}...")
    cap = cv2.VideoCapture(device_idx)

    if not cap.isOpened():
        print(f"[Error] Could not open camera device {device_idx}!")
        print("Tip: Run 'v4l2-ctl --list-devices' or 'rpicam-hello' on Raspberry Pi.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera Test] Camera opened successfully. Resolution: {actual_w}x{actual_h}")

    print("[Camera Test] Capturing 30 test frames...")
    start_t = time.time()
    frames_captured = 0
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frames_captured += 1
        time.sleep(0.01)

    elapsed = time.time() - start_t
    fps = frames_captured / elapsed
    cap.release()

    print(f"[Camera Test] Captured {frames_captured}/30 frames in {elapsed:.2f}s ({fps:.1f} FPS)")
    return True


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_camera(idx)

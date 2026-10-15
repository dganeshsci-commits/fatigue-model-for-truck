#!/usr/bin/env python3
"""
Standalone In-Cabin Edge Fatigue & Distraction Monitor for Raspberry Pi 5.

Usage:
    python edge_runner.py                      # Default camera (0) with GUI HUD
    python edge_runner.py --headless           # Headless truck deployment (Buzzer + Telemetry only)
    python edge_runner.py --camera /dev/video0 --buzzer-pin 17 --led-pin 27
    python edge_runner.py --video sample.mp4   # Offline video validation
"""

import argparse
import sys
import time
import json
import cv2
import requests

from src.detector import FatigueDetector
from src.tracker import FatigueTracker
from src.gpio_alert import GPIOAlertController
from src.camera import CameraStream
from src.telemetry import SystemTelemetry
from src.utils import draw_hud_overlay


def parse_arguments():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Edge Fatigue Detector")
    parser.add_argument("--camera", type=str, default="0",
                        help="Camera device index (e.g. 0) or device path (e.g. /dev/video0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to an optional test video file instead of live camera")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless background mode without displaying OpenCV window")
    parser.add_argument("--buzzer-pin", type=int, default=17,
                        help="BCM GPIO pin for active warning buzzer (default: 17)")
    parser.add_argument("--led-pin", type=int, default=27,
                        help="BCM GPIO pin for warning LED (default: 27)")
    parser.add_argument("--no-gpio", action="store_true",
                        help="Disable physical GPIO hardware alerts")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of intra-op threads for ONNX Runtime (default: 4 for Pi 5)")
    parser.add_argument("--dashboard-url", type=str, default=None,
                        help="Optional URL to post telemetry events to Truck Intelligence Hub")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print(" FleetLogix — Raspberry Pi 5 Edge Fatigue Detection System")
    print("=" * 60)

    # 1. Initialize ONNX detector
    print(f"[Init] Loading 4 ONNX models on {args.threads} CPU execution threads...")
    try:
        detector = FatigueDetector(num_threads=args.threads)
        print("[Init] ONNX models loaded successfully.")
    except Exception as e:
        print(f"[Fatal] Failed to load ONNX models: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Initialize temporal state machine
    tracker = FatigueTracker()

    # 3. Initialize GPIO hardware alerts
    gpio_controller = GPIOAlertController(
        buzzer_pin=args.buzzer_pin,
        led_pin=args.led_pin,
        enabled=not args.no_gpio
    )

    # 4. Initialize Telemetry
    telemetry = SystemTelemetry()

    # 5. Initialize Camera
    source = args.video if args.video else args.camera
    print(f"[Init] Connecting to video input: {source} ...")
    try:
        camera = CameraStream(src=source, width=320, height=240, fps=30).start()
        print("[Init] Camera stream active.")
    except Exception as e:
        print(f"[Fatal] Failed to initialize camera: {e}", file=sys.stderr)
        gpio_controller.cleanup()
        sys.exit(1)

    last_dashboard_sync = time.time()
    last_reported_state = "NORMAL"

    print("\n[Running] Edge driver monitoring started. Press Ctrl+C or 'q' to stop.\n")

    try:
        while True:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Run inference pipeline
            detection = detector.process_frame(frame)

            # Update temporal state machine
            status = tracker.update(detection)

            # Trigger hardware alert
            alarm_level = status.get("alarm_level", "NONE")
            gpio_controller.trigger(alarm_level)

            # Update telemetry
            fps = telemetry.update_fps()
            cpu_temp = telemetry.get_cpu_temp()

            # Optional telemetry sync to Truck Intelligence Hub dashboard
            now = time.time()
            current_state = status.get("fatigue_state", "NORMAL")
            state_changed = (current_state != last_reported_state)

            if args.dashboard_url and (state_changed or now - last_dashboard_sync >= 2.0):
                last_dashboard_sync = now
                last_reported_state = current_state
                try:
                    payload = {
                        "device_id": "rpi5-truck-unit-01",
                        "timestamp": now,
                        "fatigueScore": status.get("fatigue_score", 0),
                        "fatigueState": current_state,
                        "earLeft": status.get("ear_left", 0.0),
                        "earRight": status.get("ear_right", 0.0),
                        "mar": status.get("mar", 0.0),
                        "headPose": status.get("head_pose", {}),
                        "blinkCount": status.get("total_blinks", 0),
                        "yawnCount": status.get("total_yawns", 0),
                        "fps": fps,
                        "cpuTemp": cpu_temp
                    }
                    requests.post(f"{args.dashboard_url}/api/inference/telemetry", json=payload, timeout=0.15)
                except Exception:
                    pass

            # Display HUD if not headless
            if not args.headless:
                annotated = draw_hud_overlay(frame, status, fps=fps, cpu_temp=cpu_temp)
                cv2.imshow("FleetLogix — Pi 5 Driver Safety", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Log critical events to terminal in headless mode
                if alarm_level in ["MEDIUM", "HIGH"]:
                    print(f"[{time.strftime('%H:%M:%S')}] ALERT: {current_state} "
                          f"| Score: {status.get('fatigue_score')}% "
                          f"| EAR: {(status.get('ear_left',0)+status.get('ear_right',0))/2:.2f} "
                          f"| MAR: {status.get('mar',0):.2f} "
                          f"| FPS: {fps}")

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping edge monitor...")
    finally:
        camera.stop()
        gpio_controller.cleanup()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[Shutdown] Completed successfully.")


if __name__ == "__main__":
    main()

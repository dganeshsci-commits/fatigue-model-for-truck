#!/usr/bin/env python3
"""
Standalone In-Cabin Edge Fatigue & Multi-Sensor Monitor for Raspberry Pi 5.
Runs complete 100 Hz fusion pipeline:
- Camera 4 ONNX Fatigue Models
- ESP32 UART (BNO085 IMU + Ultrasonic Fuel + RC522 RFID)
- Phone GPS (Wi-Fi)
- IMU Harsh Driving Event Detection
- Physical Cabin GPIO Alerts (Buzzer + LED)
- Local SQLite Database Logging

Usage:
    python edge_runner.py                      # Default camera (0) with GUI HUD & 100 Hz fusion
    python edge_runner.py --headless           # Headless truck deployment
    python edge_runner.py --camera /dev/video0 --esp32-port /dev/serial0
"""

import argparse
import sys
import time
import json
import cv2
import requests

from src.sensor_fusion import SensorFusionHub
from src.camera import CameraStream
from src.telemetry import SystemTelemetry
from src.utils import draw_hud_overlay


def parse_arguments():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Edge Fatigue & Telemetry Monitor")
    parser.add_argument("--camera", type=str, default="0",
                        help="Camera device index (e.g. 0) or path (e.g. /dev/video0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to an optional test video file")
    parser.add_argument("--esp32-port", type=str, default="/dev/serial0",
                        help="Serial UART port for ESP32 (default: /dev/serial0 at 115200 baud)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless background mode without OpenCV window")
    parser.add_argument("--buzzer-pin", type=int, default=17,
                        help="BCM GPIO pin for active warning buzzer (default: 17)")
    parser.add_argument("--led-pin", type=int, default=27,
                        help="BCM GPIO pin for warning LED (default: 27)")
    parser.add_argument("--no-gpio", action="store_true",
                        help="Disable physical GPIO hardware alerts")
    parser.add_argument("--no-db", action="store_true",
                        help="Disable local SQLite database logging")
    parser.add_argument("--dashboard-url", type=str, default=None,
                        help="Optional URL to stream 100 Hz telemetry events to central dashboard")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 65)
    print(" Truck Intelligence Hub — Raspberry Pi 5 Edge Monitor (100 Hz)")
    print("=" * 65)

    # 1. Initialize complete Sensor Fusion Hub
    print(f"[Init] Initializing Sensor Fusion Hub (ESP32 on {args.esp32_port})...")
    hub = SensorFusionHub(
        device_id="truck_esp32_01",
        serial_port=args.esp32_port,
        enable_gpio=not args.no_gpio,
        enable_db=not args.no_db
    )

    # 2. Initialize Telemetry & Camera
    telemetry = SystemTelemetry()
    source = args.video if args.video else args.camera
    print(f"[Init] Connecting to video input: {source} ...")
    try:
        camera = CameraStream(src=source, width=320, height=240, fps=30).start()
        print("[Init] Camera stream active.")
    except Exception as e:
        print(f"[Warning] Camera not available: {e}. Running fusion hub without live video.")
        camera = None

    last_dashboard_sync = time.time()
    last_reported_state = "NORMAL"

    print("\n[Running] 100 Hz Real-time telematics running. Press Ctrl+C or 'q' to stop.\n")

    try:
        while True:
            # 1. Process camera frame if available
            detection_status = {}
            frame = None
            if camera:
                grabbed, frame = camera.read()
                if grabbed and frame is not None:
                    detection_status = hub.process_camera_frame(frame)

            # 2. Get instantaneous fused 100 Hz payload
            payload = hub.get_fused_payload()

            fps = telemetry.update_fps()
            cpu_temp = telemetry.get_cpu_temp()

            # 3. Optional sync to remote Truck Intelligence Hub dashboard
            now = time.time()
            current_state = payload["fatigue"]["state"]
            state_changed = (current_state != last_reported_state)

            if args.dashboard_url and (state_changed or now - last_dashboard_sync >= 0.1): # 10-100 Hz
                last_dashboard_sync = now
                last_reported_state = current_state
                try:
                    requests.post(f"{args.dashboard_url}/api/inference/telemetry", json=payload, timeout=0.08)
                except Exception:
                    pass

            # 4. Display or log
            if not args.headless and frame is not None:
                annotated = draw_hud_overlay(frame, detection_status, fps=fps, cpu_temp=cpu_temp)

                # Add bottom banner with ESP32 IMU, Fuel, RFID, and GPS
                h_img, w_img = annotated.shape[:2]
                cv2.rectangle(annotated, (10, h_img - 50), (w_img - 10, h_img - 10), (20, 24, 30), -1)
                cv2.rectangle(annotated, (10, h_img - 50), (w_img - 10, h_img - 10), (60, 70, 80), 1)

                info_txt1 = f"Driver: {payload['rfid']['driver_name']} | Fuel: {payload['fuel']['level_percent']}% ({payload['fuel']['volume_litres']}L)"
                info_txt2 = f"IMU: Accel({payload['imu']['accel_x']:.2f}, {payload['imu']['accel_y']:.2f}) | Speed: {payload['gps']['speed_kmh']} km/h | Event: {payload['events']['driving_state']}"
                cv2.putText(annotated, info_txt1, (18, h_img - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(annotated, info_txt2, (18, h_img - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

                cv2.imshow("Truck Intelligence Hub — Pi 5 HUD", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Headless logging on state changes or alerts
                if payload["fatigue"]["state"] in ["DROWSY", "CRITICAL"] or payload["events"]["driving_state"] != "normal":
                    print(f"[{payload['timestamp']}] ALERT: Fatigue={payload['fatigue']['state']} ({payload['fatigue']['score']}%) "
                          f"| Driving={payload['events']['driving_state']} "
                          f"| Fuel={payload['fuel']['level_percent']}% "
                          f"| Driver={payload['rfid']['driver_name']}")
                time.sleep(0.01) # 100 Hz loop

    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping edge monitor...")
    finally:
        if camera:
            camera.stop()
        hub.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[Shutdown] Completed successfully.")


if __name__ == "__main__":
    main()

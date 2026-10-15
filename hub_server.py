#!/usr/bin/env python3
"""
Truck Intelligence Hub — Raspberry Pi 5 Central Hub Server (100 Hz Real-Time Stream).

Features:
1. 100 Hz Fused JSON Broadcaster over WebSocket / TCP / HTTP to Dashboard
2. Ingests Phone GPS over Wi-Fi (POST /api/gps)
3. Ingests ESP32 Telemetry over UART (115200 bps)
4. Runs 4 ONNX Fatigue Models from USB Camera
5. Logs all telemetry & safety events to SQLite database
"""

import os
import sys
import time
import json
import base64
import threading
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

# Ensure local imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.sensor_fusion import SensorFusionHub
from src.camera import CameraStream
from src.utils import draw_hud_overlay
from src.telemetry import SystemTelemetry

app = Flask(__name__)

# Initialize central fusion hub
hub = SensorFusionHub(
    device_id="truck_esp32_01",
    serial_port=os.environ.get("ESP32_PORT", "/dev/serial0"),
    enable_gpio=True,
    enable_db=True
)

camera = None
telemetry = SystemTelemetry()
LATEST_HUD_FRAME = None
frame_lock = threading.Lock()


def camera_loop():
    """Background thread to process camera at 30-45 FPS and feed the fusion hub."""
    global camera, LATEST_HUD_FRAME
    cam_src = os.environ.get("CAMERA_SRC", "0")
    try:
        camera = CameraStream(src=cam_src, width=320, height=240, fps=30).start()
        print(f"[Hub Camera] Camera stream started on source {cam_src}")
    except Exception as e:
        print(f"[Hub Camera] Camera not available ({e}). Running without live vision capture.")
        return

    while True:
        grabbed, frame = camera.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        # Run vision inference through fusion hub
        status = hub.process_camera_frame(frame)
        fps = telemetry.update_fps()
        cpu_temp = telemetry.get_cpu_temp()

        annotated = draw_hud_overlay(frame, status, fps=fps, cpu_temp=cpu_temp)
        with frame_lock:
            LATEST_HUD_FRAME = annotated
        time.sleep(0.02)


# Start camera background thread
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "system": "Truck Intelligence Hub (Raspberry Pi 5)",
        "frequency": "100 Hz",
        "timestamp": time.time()
    })


@app.route('/api/telemetry/latest', methods=['GET'])
def get_latest_telemetry():
    """Returns the latest 100 Hz fused JSON payload."""
    payload = hub.get_fused_payload()
    return jsonify(payload)


@app.route('/api/telemetry/history', methods=['GET'])
def get_telemetry_history():
    """Returns historical sensor readings from SQLite database."""
    limit = int(request.args.get("limit", 100))
    if hub.db:
        history = hub.db.get_recent_history(limit=limit)
        return jsonify(history)
    return jsonify([])


@app.route('/api/gps', methods=['POST'])
def receive_phone_gps():
    """
    Receives Phone GPS telemetry over Wi-Fi.
    Accepts: { "latitude": 12.9716, "longitude": 77.5946, "speed_kmh": 68.0, "heading": 45.0 }
    """
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"error": "Invalid GPS payload"}), 400

    lat = float(data.get("latitude"))
    lon = float(data.get("longitude"))
    speed = float(data.get("speed_kmh", 0.0))
    heading = float(data.get("heading", 0.0))

    hub.gps.update_from_phone(lat, lon, speed, heading)
    return jsonify({"status": "gps_updated", "source": "PHONE_WIFI"})


@app.route('/api/predict', methods=['POST'])
def predict_frame():
    """Base64 frame inference for dashboard webcam stream."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        status = hub.process_camera_frame(frame)
        payload = hub.get_fused_payload()

        return jsonify({
            **payload,
            "faceDetected": status.get("face_detected", False),
            "faceBox": status.get("face_box", [0, 0, 0, 0]),
            "faceLandmarks": status.get("landmarks", []),
            "leftEye": status.get("left_eye_status", "open"),
            "rightEye": status.get("right_eye_status", "open"),
            "mouthState": status.get("mouth_status", "normal"),
            "headPose": status.get("head_pose", {}),
            "earLeft": status.get("ear_left", 0.0),
            "earRight": status.get("ear_right", 0.0),
            "mar": status.get("mar", 0.0),
            "fatigueScore": status.get("fatigue_score", 0),
            "fatigueState": status.get("fatigue_state", "NORMAL")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def gen_camera_feed():
    """Generates continuous MJPEG stream."""
    global LATEST_HUD_FRAME
    while True:
        with frame_lock:
            if LATEST_HUD_FRAME is not None:
                ret, jpeg = cv2.imencode('.jpg', LATEST_HUD_FRAME)
                if ret:
                    frame_bytes = jpeg.tobytes()
                else:
                    frame_bytes = None
            else:
                frame_bytes = None

        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


@app.route('/camera_feed')
def camera_feed():
    return Response(gen_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print(" Truck Intelligence Hub — Raspberry Pi 5 Central Processing Server")
    print(f" Listening on http://0.0.0.0:{port} (100 Hz Data Flow)")
    print("=" * 65)

    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    finally:
        hub.stop()

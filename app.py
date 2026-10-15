#!/usr/bin/env python3
"""
Raspberry Pi 5 Edge Fatigue REST API & MJPEG Streaming Server.

Endpoints:
- POST /api/predict       : Base64 frame inference (100% compatible with Truck Intelligence Hub)
- GET  /camera_feed       : Live MJPEG stream with ADAS HUD overlays
- GET  /api/status        : Real-time driver status & Pi 5 hardware telemetry
- GET  /api/health        : Health check
"""

import os
import sys
import time
import base64
import json
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

from src.detector import FatigueDetector
from src.tracker import FatigueTracker
from src.gpio_alert import GPIOAlertController
from src.camera import CameraStream
from src.telemetry import SystemTelemetry
from src.utils import draw_hud_overlay

app = Flask(__name__)

# Global instances
detector = None
tracker = FatigueTracker()
gpio_controller = None
camera = None
telemetry = SystemTelemetry()

LATEST_STATUS = {
    "faceDetected": False,
    "fatigueScore": 0,
    "fatigueState": "NORMAL",
    "leftEye": "open",
    "rightEye": "open",
    "mouthState": "normal",
    "headPose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "earLeft": 0.0,
    "earRight": 0.0,
    "mar": 0.0,
    "fps": 0.0,
    "cpuTemp": 0.0
}


def get_detector():
    global detector
    if detector is None:
        detector = FatigueDetector(num_threads=4)
    return detector


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": "Raspberry Pi 5",
        "engine": "ONNX_RUNTIME_ARM64_NEON",
        "timestamp": time.time()
    })


@app.route('/api/status', methods=['GET'])
def status():
    sys_stats = telemetry.get_system_stats()
    return jsonify({
        **LATEST_STATUS,
        "system": sys_stats
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Frame inference endpoint compatible with FleetLogix Truck Intelligence Hub.
    Accepts: { "image": "data:image/jpeg;base64,..." }
    """
    global LATEST_STATUS
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Run ONNX inference
        det = get_detector()
        detection = det.process_frame(frame, target_w=320, target_h=240)
        res = tracker.update(detection)

        # Trigger hardware alert
        if gpio_controller:
            gpio_controller.trigger(res.get("alarm_level", "NONE"))

        fps = telemetry.update_fps()
        cpu_temp = telemetry.get_cpu_temp()

        # Update cache
        LATEST_STATUS = {
            "faceDetected": res.get("face_detected", False),
            "faceBox": res.get("face_box", [0, 0, 0, 0]),
            "faceLandmarks": res.get("landmarks", []),
            "leftEye": res.get("left_eye_status", "open"),
            "rightEye": res.get("right_eye_status", "open"),
            "mouthState": res.get("mouth_status", "normal"),
            "headPose": res.get("head_pose", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}),
            "earLeft": res.get("ear_left", 0.0),
            "earRight": res.get("ear_right", 0.0),
            "mar": res.get("mar", 0.0),
            "fatigueScore": res.get("fatigue_score", 0),
            "fatigueState": res.get("fatigue_state", "NORMAL"),
            "totalBlinks": res.get("total_blinks", 0),
            "totalYawns": res.get("total_yawns", 0),
            "blinkRateBpm": res.get("blink_rate_bpm", 0),
            "fps": fps,
            "cpuTemp": cpu_temp,
            "latencyMs": res.get("latency_ms", 0.0)
        }

        return jsonify(LATEST_STATUS)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def gen_mjpeg_stream():
    """Generates continuous MJPEG stream with HUD overlays."""
    global camera, LATEST_STATUS
    if camera is None:
        try:
            camera = CameraStream(src=0, width=320, height=240, fps=30).start()
        except Exception as e:
            print(f"[Camera Error] {e}")
            return

    det = get_detector()

    while True:
        grabbed, frame = camera.read()
        if not grabbed or frame is None:
            time.sleep(0.01)
            continue

        detection = det.process_frame(frame)
        res = tracker.update(detection)

        if gpio_controller:
            gpio_controller.trigger(res.get("alarm_level", "NONE"))

        fps = telemetry.update_fps()
        cpu_temp = telemetry.get_cpu_temp()

        annotated = draw_hud_overlay(frame, res, fps=fps, cpu_temp=cpu_temp)
        ret, jpeg = cv2.imencode('.jpg', annotated)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/camera_feed')
def camera_feed():
    return Response(gen_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Raspberry Pi 5 Fatigue API Service on port {port}...")

    # Initialize GPIO controller
    gpio_controller = GPIOAlertController(buzzer_pin=17, led_pin=27, enabled=True)

    # Preload detector
    try:
        get_detector()
        print("ONNX models initialized successfully.")
    except Exception as err:
        print(f"ONNX preload warning: {err}")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

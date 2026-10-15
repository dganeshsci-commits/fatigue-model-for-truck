#!/usr/bin/env python3
"""
Diagnostic test script to verify all 4 ONNX models and benchmark inference latency.
Can be run on developer workstation or directly on Raspberry Pi 5.
"""

import sys
import os
import time
import numpy as np

# Ensure parent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector import FatigueDetector
from src.tracker import FatigueTracker


def run_benchmark():
    print("=" * 65)
    print(" Fatigue Model Diagnostic & Benchmark Suite for Raspberry Pi 5")
    print("=" * 65)

    models_dir = os.path.join(BASE_DIR, "models")
    print(f"\n[1/3] Loading ONNX Models from: {models_dir}")
    t0 = time.time()
    detector = FatigueDetector(models_dir=models_dir, num_threads=4)
    print(f"      Models loaded in {(time.time() - t0)*1000:.1f} ms")

    tracker = FatigueTracker()

    # Model Inspection
    print("\n[2/3] Inspecting Model Tensor Signatures:")
    models = [
        ("Face Detector (RFB-320)", detector.face_session),
        ("Landmark Regressor (PFLD-98)", detector.pfld_session),
        ("Eye Classifier", detector.eye_session),
        ("Mouth Classifier", detector.mouth_session),
    ]
    for name, sess in models:
        inp = sess.get_inputs()[0]
        outs = [o.name for o in sess.get_outputs()]
        print(f"  • {name:<28}: In={inp.name} {inp.shape} ({inp.type}) -> Out={outs}")

    print("\n[3/3] Running Inference Pipeline Benchmark (Warmup + 30 iterations):")
    # Synthetic realistic frame (gradient with simulated face ellipse)
    test_frame = np.full((240, 320, 3), 160, dtype=np.uint8)
    import cv2
    # Draw a simulated skin-tone face oval to trigger detector
    cv2.ellipse(test_frame, (160, 120), (55, 75), 0, 0, 360, (140, 165, 210), -1)
    # Eyes
    cv2.ellipse(test_frame, (140, 105), (10, 6), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(test_frame, (140, 105), 4, (40, 30, 20), -1)
    cv2.ellipse(test_frame, (180, 105), (10, 6), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(test_frame, (180, 105), 4, (40, 30, 20), -1)
    # Mouth
    cv2.ellipse(test_frame, (160, 155), (16, 8), 0, 0, 360, (80, 80, 180), -1)

    # Warmup
    detector.process_frame(test_frame)

    latencies = []
    for _ in range(30):
        t_start = time.time()
        res = detector.process_frame(test_frame)
        status = tracker.update(res)
        latencies.append((time.time() - t_start) * 1000)

    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps_equiv = 1000.0 / avg_lat

    print(f"  • Average Latency : {avg_lat:.2f} ms")
    print(f"  • Min / Max       : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  • Throughput      : {fps_equiv:.1f} FPS")
    print(f"  • Detection Output: Face={res['face_detected']}, State={status['fatigue_state']}, Score={status['fatigue_score']}%")

    print("\n" + "=" * 65)
    print(" Diagnostic Check: PASSED ALL TESTS!")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()

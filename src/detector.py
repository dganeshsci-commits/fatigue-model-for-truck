"""
Multi-stage ONNX Fatigue Detection Engine for Raspberry Pi 5.
Orchestrates:
1. version-slim-320.onnx         -> Ultra-light face detection
2. PFLD_GhostOne_112_1_opt_sim   -> 98-point facial landmark regression
3. eye_model.onnx                -> Eye state classification
4. mouth_model.onnx              -> Mouth / yawn classification
"""

import os
import time
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

from .utils import (
    cut_resize_letterbox,
    predict_boxes,
    compute_ear_from_landmarks,
    compute_mar_from_landmarks,
    compute_head_pose_from_landmarks
)


class FatigueDetector:
    """
    High-performance ONNX inference engine tuned for Raspberry Pi 5's Cortex-A76 cores.
    """

    def __init__(self, models_dir: str = None, num_threads: int = 4):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")

        self.models_dir = models_dir
        self.num_threads = num_threads

        # ONNX Runtime Session Options optimized for ARM64 NEON on RPi 5
        self.sess_opts = ort.SessionOptions()
        self.sess_opts.intra_op_num_threads = self.num_threads
        self.sess_opts.inter_op_num_threads = 1
        self.sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Providers
        providers = ['CPUExecutionProvider']

        # Paths
        self.face_model_path = os.path.join(models_dir, "version-slim-320.onnx")
        self.pfld_model_path = os.path.join(models_dir, "PFLD_GhostOne_112_1_opt_sim.onnx")
        self.eye_model_path = os.path.join(models_dir, "eye_model.onnx")
        self.mouth_model_path = os.path.join(models_dir, "mouth_model.onnx")

        # Verify models exist
        for p in [self.face_model_path, self.pfld_model_path, self.eye_model_path, self.mouth_model_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"ONNX Model not found at: {p}")

        # Load sessions
        self.face_session = ort.InferenceSession(self.face_model_path, self.sess_opts, providers=providers)
        self.pfld_session = ort.InferenceSession(self.pfld_model_path, self.sess_opts, providers=providers)
        self.eye_session = ort.InferenceSession(self.eye_model_path, self.sess_opts, providers=providers)
        self.mouth_session = ort.InferenceSession(self.mouth_model_path, self.sess_opts, providers=providers)

        # Landmark smoothing buffer
        self.prev_pts = []

    def process_frame(self, frame: np.ndarray, target_w: int = 320, target_h: int = 240) -> dict:
        """
        Runs complete inference pipeline on a BGR video frame.
        Returns a dictionary with face detection status, landmark coordinates,
        EAR, MAR, head pose, eye states, and inference latency benchmarks.
        """
        t0 = time.time()
        result = {
            "face_detected": False,
            "face_box": None,
            "landmarks": None,
            "ear_left": 0.0,
            "ear_right": 0.0,
            "mar": 0.0,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "left_eye_status": "open",
            "right_eye_status": "open",
            "mouth_status": "normal",
            "eye_left_box": None,
            "eye_right_box": None,
            "mouth_box": None,
            "latency_ms": 0.0
        }

        # 1. Resize frame to model's expected 320x240
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame_resized = cv2.resize(frame, (target_w, target_h))
        else:
            frame_resized = frame

        # 2. Stage 1: Face Detection (version-slim-320.onnx)
        # Preprocessing: BGR -> RGB, normalize to [-1.0, 1.0], transpose to NCHW
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        face_input = (rgb_frame.astype(np.float32) - 127.0) / 128.0
        face_input = np.transpose(face_input, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)

        confidences, boxes = self.face_session.run(
            None,
            {self.face_session.get_inputs()[0].name: face_input}
        )

        boxes_pixel, labels, probs = predict_boxes(
            confidences, boxes,
            prob_threshold=0.45,
            image_size=(target_w, target_h),
            iou_threshold=0.3
        )

        if len(probs) == 0:
            self.prev_pts = []
            result["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return result

        result["face_detected"] = True

        # Pick candidate with highest confidence score
        best_idx = int(np.argmax(probs))
        face_x1, face_y1, face_x2, face_y2 = boxes_pixel[best_idx]
        face_box = [int(face_x1), int(face_y1), int(face_x2 - face_x1), int(face_y2 - face_y1)]
        result["face_box"] = face_box

        # 3. Stage 2: Facial Landmark Regression (PFLD 98 points)
        pil_frame = Image.fromarray(rgb_frame)
        face_crop, scale_factor, x_offset, y_offset = cut_resize_letterbox(
            pil_frame, face_box, (112, 112)
        )

        # Preprocessing: normalize to [0.0, 1.0], transpose to [1, 3, 112, 112]
        pfld_input = np.array(face_crop).astype(np.float32) / 255.0
        pfld_input = np.transpose(pfld_input, (2, 0, 1))
        pfld_input = np.expand_dims(pfld_input, axis=0)

        pfld_output = self.pfld_session.run(
            None,
            {self.pfld_session.get_inputs()[0].name: pfld_input}
        )
        preds = np.squeeze(pfld_output)

        # Check for sudden face jumps across frame to reset smoothing buffer
        if len(self.prev_pts) == 196:
            old_nose_x = self.prev_pts[54 * 2]
            old_nose_y = self.prev_pts[54 * 2 + 1]
            new_nose_x = preds[54 * 2] * 112 * scale_factor + x_offset
            new_nose_y = preds[54 * 2 + 1] * 112 * scale_factor + y_offset
            if abs(new_nose_x - old_nose_x) > 45 or abs(new_nose_y - old_nose_y) > 45:
                self.prev_pts = []

        # Landmark exponential smoothing (alpha=0.78 keeps fast eye blinks sharp)
        if len(self.prev_pts) == 0:
            for i in range(98):
                self.prev_pts.append(preds[i * 2] * 112 * scale_factor + x_offset)
                self.prev_pts.append(preds[i * 2 + 1] * 112 * scale_factor + y_offset)
        else:
            alpha = 0.78
            for i in range(98):
                cx = preds[i * 2] * 112 * scale_factor + x_offset
                cy = preds[i * 2 + 1] * 112 * scale_factor + y_offset
                self.prev_pts[i * 2] = cx * alpha + self.prev_pts[i * 2] * (1.0 - alpha)
                self.prev_pts[i * 2 + 1] = cy * alpha + self.prev_pts[i * 2 + 1] * (1.0 - alpha)

        result["landmarks"] = list(self.prev_pts)

        # 4. Geometric EAR, MAR, and Head Pose estimation
        # Also compute on raw points so quick blinks aren't masked by smoothing
        raw_pts = []
        for i in range(98):
            raw_pts.append(preds[i * 2] * 112 * scale_factor + x_offset)
            raw_pts.append(preds[i * 2 + 1] * 112 * scale_factor + y_offset)

        r_eye_indices = [60, 61, 62, 63, 64, 65, 66, 67]
        l_eye_indices = [68, 69, 70, 71, 72, 73, 74, 75]

        ear_r = min(compute_ear_from_landmarks(self.prev_pts, r_eye_indices),
                    compute_ear_from_landmarks(raw_pts, r_eye_indices))
        ear_l = min(compute_ear_from_landmarks(self.prev_pts, l_eye_indices),
                    compute_ear_from_landmarks(raw_pts, l_eye_indices))
        mar = max(compute_mar_from_landmarks(self.prev_pts),
                  compute_mar_from_landmarks(raw_pts))
        head_pose = compute_head_pose_from_landmarks(self.prev_pts)

        result["ear_right"] = round(ear_r, 4)
        result["ear_left"] = round(ear_l, 4)
        result["mar"] = round(mar, 4)
        result["head_pose"] = head_pose

        # 5. Crop Eye and Mouth regions for secondary CNN models
        cut_bias = 15
        left_eye_box = [
            self.prev_pts[68 * 2] - cut_bias,
            self.prev_pts[70 * 2 + 1] - cut_bias,
            self.prev_pts[72 * 2] - self.prev_pts[68 * 2] + cut_bias * 2,
            self.prev_pts[74 * 2 + 1] - self.prev_pts[70 * 2 + 1] + cut_bias * 1.5
        ]
        right_eye_box = [
            self.prev_pts[60 * 2] - cut_bias,
            self.prev_pts[62 * 2 + 1] - cut_bias,
            self.prev_pts[64 * 2] - self.prev_pts[60 * 2] + cut_bias * 2,
            self.prev_pts[66 * 2 + 1] - self.prev_pts[62 * 2 + 1] + cut_bias * 1.5
        ]
        mouth_box = [
            self.prev_pts[76 * 2] - cut_bias,
            self.prev_pts[79 * 2 + 1] - cut_bias,
            self.prev_pts[82 * 2] - self.prev_pts[76 * 2] + cut_bias * 2,
            self.prev_pts[85 * 2 + 1] - self.prev_pts[79 * 2 + 1] + cut_bias * 1.5
        ]

        result["eye_left_box"] = left_eye_box
        result["eye_right_box"] = right_eye_box
        result["mouth_box"] = mouth_box

        mouth_crop, _, _, _ = cut_resize_letterbox(pil_frame, mouth_box, (64, 64))
        mouth_input = np.array(mouth_crop).astype(np.float32) / 255.0
        mouth_input = np.transpose(mouth_input, (2, 0, 1))
        mouth_input = np.expand_dims(mouth_input, axis=0)

        mouth_out = self.mouth_session.run(
            None,
            {self.mouth_session.get_inputs()[0].name: mouth_input}
        )
        mouth_cnn_open = (np.argmax(np.squeeze(mouth_out)) == 0)

        # 6. Resolve states
        # Geometric EAR threshold: < 0.21 -> Closed
        result["right_eye_status"] = "closed" if ear_r < 0.21 else "open"
        result["left_eye_status"] = "closed" if ear_l < 0.21 else "open"
        if (ear_r + ear_l) / 2.0 < 0.215:
            result["right_eye_status"] = "closed"
            result["left_eye_status"] = "closed"

        # MAR threshold: > 0.36 -> Yawn, < 0.28 -> Normal, between -> use CNN
        if mar > 0.36:
            result["mouth_status"] = "yawn"
        elif mar < 0.28:
            result["mouth_status"] = "normal"
        else:
            result["mouth_status"] = "yawn" if mouth_cnn_open else "normal"

        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return result

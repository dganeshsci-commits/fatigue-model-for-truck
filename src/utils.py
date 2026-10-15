"""
Utility functions for face detection, landmark transformations,
geometric aspect ratio metrics, and visual HUD rendering on Raspberry Pi 5.
"""

import numpy as np
from PIL import Image
import cv2


def cut_resize_letterbox(image: Image.Image, det: list, target_size: tuple):
    """
    Crops the face bounding box with uniform square padding to preserve aspect ratio,
    then resizes to target_size (e.g. 112x112 for PFLD, 64x64 for eye/mouth).
    """
    iw, ih = image.size

    facebox_x = int(det[0])
    facebox_y = int(det[1])
    facebox_w = int(det[2])
    facebox_h = int(det[3])

    facebox_max_length = max(facebox_w, facebox_h)
    width_margin_length = int((facebox_max_length - facebox_w) / 2)
    height_margin_length = int((facebox_max_length - facebox_h) / 2)

    face_letterbox_x = facebox_x - width_margin_length
    face_letterbox_y = facebox_y - height_margin_length
    face_letterbox_w = facebox_max_length
    face_letterbox_h = facebox_max_length

    top = -face_letterbox_y if face_letterbox_y < 0 else 0
    left = -face_letterbox_x if face_letterbox_x < 0 else 0
    bottom = face_letterbox_y + face_letterbox_h - ih if face_letterbox_y + face_letterbox_h - ih > 0 else 0
    right = face_letterbox_x + face_letterbox_w - iw if face_letterbox_x + face_letterbox_w - iw > 0 else 0

    margin_image = Image.new('RGB', (iw + right - left, ih + bottom - top), (0, 0, 0))
    margin_image.paste(image, (left, top))

    face_letterbox = margin_image.crop((
        face_letterbox_x,
        face_letterbox_y,
        face_letterbox_x + face_letterbox_w,
        face_letterbox_y + face_letterbox_h
    ))
    face_letterbox = face_letterbox.resize(target_size, Image.Resampling.BICUBIC)

    scale_factor = facebox_max_length / target_size[0]
    return face_letterbox, scale_factor, face_letterbox_x, face_letterbox_y


def area_of(left_top, right_bottom):
    """Compute rectangular area given top-left and bottom-right corners."""
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Intersection-over-Union (IoU) calculation."""
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold=0.3, top_k=1, candidate_size=200):
    """Non-maximum suppression to filter overlapping face bounding boxes."""
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict_boxes(confidences, boxes, prob_threshold=0.45, image_size=(320, 240), iou_threshold=0.3, top_k=1):
    """Parses output tensors from version-slim-320 into pixel coordinate boxes."""
    boxes = np.squeeze(boxes)
    confidences = np.squeeze(confidences)
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= image_size[0]
    picked_box_probs[:, 1] *= image_size[1]
    picked_box_probs[:, 2] *= image_size[0]
    picked_box_probs[:, 3] *= image_size[1]
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def compute_ear_from_landmarks(pts: list, eye_indices: list) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from WFLW 98 facial landmark points.
    Typical values: Open eye ~0.24-0.34, Closed eye <0.21.
    """
    try:
        p = [np.array([pts[i * 2], pts[i * 2 + 1]], dtype=np.float32) for i in eye_indices]
        v1 = np.linalg.norm(p[1] - p[7])
        v2 = np.linalg.norm(p[2] - p[6])
        v3 = np.linalg.norm(p[3] - p[5])
        h = np.linalg.norm(p[0] - p[4])
        if h < 1e-4:
            return 0.0
        return float((v1 + v2 + v3) / (3.0 * h))
    except Exception:
        return 0.0


def compute_mar_from_landmarks(pts: list) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR) using WFLW 98 inner-lip (89-95) and outer-lip (76-87) landmarks.
    Compensates for lip thickness so closed mouth is ~0.06-0.08, yawn >0.36.
    """
    try:
        p_left = np.array([pts[76 * 2], pts[76 * 2 + 1]], dtype=np.float32)
        p_right = np.array([pts[82 * 2], pts[82 * 2 + 1]], dtype=np.float32)
        h = np.linalg.norm(p_left - p_right)
        if h < 1e-4:
            return 0.0

        p89 = np.array([pts[89 * 2], pts[89 * 2 + 1]], dtype=np.float32)
        p95 = np.array([pts[95 * 2], pts[95 * 2 + 1]], dtype=np.float32)
        p90 = np.array([pts[90 * 2], pts[90 * 2 + 1]], dtype=np.float32)
        p94 = np.array([pts[94 * 2], pts[94 * 2 + 1]], dtype=np.float32)
        p91 = np.array([pts[91 * 2], pts[91 * 2 + 1]], dtype=np.float32)
        p93 = np.array([pts[93 * 2], pts[93 * 2 + 1]], dtype=np.float32)
        inner_v = (np.linalg.norm(p89 - p95) + np.linalg.norm(p90 - p94) + np.linalg.norm(p91 - p93)) / 3.0
        inner_mar = float(inner_v / h)

        p_top = np.array([pts[79 * 2], pts[79 * 2 + 1]], dtype=np.float32)
        p_bot = np.array([pts[85 * 2], pts[85 * 2 + 1]], dtype=np.float32)
        outer_mar = float(np.linalg.norm(p_top - p_bot) / h)

        effective_mar = max(inner_mar * 1.45, max(0.0, outer_mar - 0.24) * 1.15)
        return float(effective_mar)
    except Exception:
        return 0.0


def compute_head_pose_from_landmarks(pts: list) -> dict:
    """
    Estimate head orientation (yaw, pitch, roll in degrees) from WFLW 98 landmarks.
    Looking away flagged if |yaw| > 28° or |pitch| > 22°.
    """
    try:
        x_eye_r, y_eye_r = pts[60 * 2], pts[60 * 2 + 1]
        x_eye_l, y_eye_l = pts[72 * 2], pts[72 * 2 + 1]
        x_nose, y_nose = pts[54 * 2], pts[54 * 2 + 1]
        x_chin, y_chin = pts[16 * 2], pts[16 * 2 + 1]
        x_cheek_r = pts[0 * 2]
        x_cheek_l = pts[32 * 2]

        dx = x_eye_l - x_eye_r
        dy = y_eye_l - y_eye_r
        roll = np.degrees(np.arctan2(dy, dx))

        span_x = max(abs(x_cheek_l - x_cheek_r), 1.0)
        nose_ratio = (x_nose - x_cheek_r) / span_x
        yaw = (nose_ratio - 0.5) * 90.0

        eye_y_mid = (y_eye_r + y_eye_l) / 2.0
        face_h = max(abs(y_chin - eye_y_mid), 1.0)
        nose_v = (y_nose - eye_y_mid) / face_h
        pitch = (nose_v - 0.38) * 90.0

        return {
            "yaw": round(float(yaw), 1),
            "pitch": round(float(pitch), 1),
            "roll": round(float(roll), 1)
        }
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def draw_hud_overlay(frame: np.ndarray, result: dict, fps: float = 0.0, cpu_temp: float = 0.0) -> np.ndarray:
    """
    Renders high-contrast ADAS Heads-Up Display (HUD) overlay on the video frame
    suitable for Raspberry Pi 5 HDMI displays or MJPEG cabin monitoring streams.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Color scheme (BGR)
    COLOR_SAFE = (0, 220, 100)      # Emerald Green
    COLOR_WARN = (0, 190, 255)      # Amber / Orange
    COLOR_CRIT = (30, 30, 255)      # Bright Red
    COLOR_BG = (20, 24, 30)         # Dark slate
    COLOR_WHITE = (245, 245, 245)

    if not result.get("face_detected", False):
        # Draw "NO DRIVER DETECTED" banner
        cv2.putText(annotated, "NO DRIVER DETECTED", (int(w * 0.22), int(h * 0.5)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 120, 255), 2, cv2.LINE_AA)
    else:
        # 1. Bounding box & state color
        state = result.get("fatigue_state", "NORMAL")
        state_color = COLOR_SAFE if state == "NORMAL" else (COLOR_WARN if state in ["ATTENTION", "DROWSY"] else COLOR_CRIT)

        fb = result.get("face_box", [0, 0, 0, 0])
        fx, fy, fw, fh = int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3])
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), state_color, 2)

        # 2. Draw 98 Facial Landmarks
        landmarks = result.get("landmarks", [])
        if len(landmarks) == 196:
            for i in range(98):
                cx, cy = int(landmarks[i * 2]), int(landmarks[i * 2 + 1])
                cv2.circle(annotated, (cx, cy), 1, (0, 255, 200), -1)

        # 3. Eye bounding boxes & colors
        for eye_key, box_key, status_key in [("Left Eye", "eye_left_box", "left_eye_status"),
                                             ("Right Eye", "eye_right_box", "right_eye_status")]:
            eb = result.get(box_key)
            if eb:
                ex, ey, ew, eh = int(eb[0]), int(eb[1]), int(eb[2]), int(eb[3])
                c = COLOR_CRIT if result.get(status_key) == "closed" else COLOR_SAFE
                cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), c, 1)

        # 4. Status Pill in upper right
        score = result.get("fatigue_score", 0)
        cv2.rectangle(annotated, (w - 180, 10), (w - 10, 48), COLOR_BG, -1)
        cv2.rectangle(annotated, (w - 180, 10), (w - 10, 48), state_color, 2)
        cv2.putText(annotated, f"{state} ({score}%)", (w - 170, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, state_color, 1, cv2.LINE_AA)

    # Telemetry Bar at top-left
    ear_l = result.get("ear_left", 0.0)
    ear_r = result.get("ear_right", 0.0)
    mar_v = result.get("mar", 0.0)
    hp = result.get("head_pose", {})
    yaw = hp.get("yaw", 0.0)

    # Semi-transparent HUD header
    cv2.rectangle(annotated, (10, 10), (280, 85), COLOR_BG, -1)
    cv2.rectangle(annotated, (10, 10), (280, 85), (60, 70, 80), 1)

    fps_txt = f"RPi 5: {fps:.1f} FPS"
    if cpu_temp > 0:
        fps_txt += f" | {cpu_temp:.1f}C"
    cv2.putText(annotated, fps_txt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"EAR: {(ear_l + ear_r)/2.0:.2f} | MAR: {mar_v:.2f}", (18, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Yaw: {yaw:+.1f}deg | Blinks: {result.get('blink_count', 0)}", (18, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    return annotated

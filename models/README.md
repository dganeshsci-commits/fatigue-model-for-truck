# ONNX Fatigue Detection Models for Truck ADAS

This directory contains the 4 separate ONNX neural network models powering the real-time driver fatigue and distraction monitoring system on the **Raspberry Pi 5**.

---

## Model Inventory & Specifications

| Model File | Task | Input Shape | Input Range / Color | Output Shape / Description |
| :--- | :--- | :--- | :--- | :--- |
| **`version-slim-320.onnx`** | Face Detection (RFB-320) | `[1, 3, 240, 320]` (NCHW) | `[-1.0, 1.0]`, RGB | `confidences: [1, 4420, 2]`, `boxes: [1, 4420, 4]` |
| **`PFLD_GhostOne_112_1_opt_sim.onnx`** | 98 Facial Landmarks (PFLD) | `[1, 3, 112, 112]` (NCHW) | `[0.0, 1.0]`, RGB | `landmarks: [1, 196]` (Normalized x, y pairs) |
| **`eye_model.onnx`** | Eye State Classification | `[1, 3, 64, 64]` (NCHW) | `[0.0, 1.0]`, RGB | `logits: [1, 2]` (`0`: Open, `1`: Closed) |
| **`mouth_model.onnx`** | Mouth State Classification | `[1, 3, 64, 64]` (NCHW) | `[0.0, 1.0]`, RGB | `logits: [1, 2]` (`0`: Open / Yawn, `1`: Normal) |

---

## Detailed Model Descriptions

### 1. `version-slim-320.onnx` (~1.16 MB)
- **Architecture**: Ultra-Lightweight Fast Generic Face Detector (RFB - Receptive Field Block backbone).
- **Purpose**: Fast face bounding box localization under diverse cabin lighting (daylight, night IR illumination, sunglasses reflection).
- **Preprocessing**:
  1. Resize camera frame to `320 × 240` (Width × Height).
  2. Convert from BGR to RGB.
  3. Normalize: `(pixel - 127.0) / 128.0` (range `[-1.0, 1.0]`).
  4. Transpose from `HWC` to `CHW` and add batch dimension `[1, 3, 240, 320]`.
- **Post-processing**:
  - Hard Non-Maximum Suppression (Hard NMS) with IoU threshold `0.3` and probability threshold `0.45`.

---

### 2. `PFLD_GhostOne_112_1_opt_sim.onnx` (~2.84 MB)
- **Architecture**: Practical Facial Landmark Detector (PFLD) with GhostNet lightweight backbone optimized for edge ARM devices.
- **Purpose**: Regresses 98 facial keypoints (WFLW standard) spanning face contour, eyebrows, eyes, nose, and mouth.
- **Preprocessing**:
  1. Crop face bounding box using square letterbox padding (`cut_resize_letterbox`) to prevent aspect ratio distortion.
  2. Resize cropped face to `112 × 112`.
  3. Normalize: `pixel / 255.0` (range `[0.0, 1.0]`).
  4. Transpose to `[1, 3, 112, 112]`.
- **Derived Metrics**:
  - **Eye Aspect Ratio (EAR)**: Computed from 8 landmark pairs per eye:
    $$\text{EAR} = \frac{\|p_1 - p_7\| + \|p_2 - p_6\| + \|p_3 - p_5\|}{3 \cdot \|p_0 - p_4\|}$$
    Threshold: $\text{EAR} < 0.21 \implies \text{Closed Eye}$.
  - **Mouth Aspect Ratio (MAR)**: Computed from inner-lip landmarks (89–95) and lip-thickness compensated outer lips:
    $$\text{MAR} = \max\left(1.45 \cdot \text{MAR}_{\text{inner}}, 1.15 \cdot \max(0.0, \text{MAR}_{\text{outer}} - 0.24)\right)$$
    Threshold: $\text{MAR} > 0.36 \implies \text{Yawn}$.
  - **Head Pose Estimation**: Yaw, pitch, and roll estimated from 3D facial symmetry (nose-to-cheek and eye line angles).

---

### 3. `eye_model.onnx` (~735 KB)
- **Architecture**: MobileNetV3 / Custom lightweight CNN classifier.
- **Purpose**: Fine-grained secondary verification for eye closure.
- **Input**: `64 × 64 × 3` RGB eye crop normalized to `[0.0, 1.0]`.

---

### 4. `mouth_model.onnx` (~735 KB)
- **Architecture**: Lightweight CNN classifier for mouth state.
- **Purpose**: Detects yawning and mouth opening when driver is speaking or yawning.
- **Input**: `64 × 64 × 3` RGB mouth crop normalized to `[0.0, 1.0]`.

---

## Raspberry Pi 5 Optimization Notes
- On the Raspberry Pi 5 (Quad-core Broadcom BCM2712 Cortex-A76 @ 2.4GHz), ONNX Runtime uses ARM NEON SIMD optimizations natively.
- Inference time per pipeline pass is **~18–28 ms total**, achieving **35–50 FPS** real-time throughput.

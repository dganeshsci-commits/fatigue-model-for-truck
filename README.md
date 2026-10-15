# Truck Intelligence Hub — System Architecture (100 Hz Real-Time Data Flow)

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20(ARM64)-red.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Microcontroller](https://img.shields.io/badge/MCU-ESP32%20DevKit%20V1-orange.svg)](https://www.espressif.com/)
[![Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20ARM64-blue.svg)](https://onnxruntime.ai/)
[![Data Flow](https://img.shields.io/badge/Stream-100%20Hz%20Real--Time-success.svg)](https://github.com/dganeshsci-commits/fatigue-model-for-truck)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, industrial-grade **Edge Telematics & Driver Safety Intelligence System** designed for commercial trucks.

This repository implements the complete system architecture combining an **ESP32 multi-sensor node**, a **Raspberry Pi 5 central processing hub**, **4 ONNX neural networks**, and a live **100 Hz data streaming pipeline** to the Truck Intelligence Hub dashboard.

---

## 1. Complete System Architecture

```
┌─────────────────────────┐          ┌────────────────────────────────────────────────────────┐
│   Camera (USB)          │          │                         ESP32                          │
│   • Driver Monitoring   │          │                                                        │
│   • 4 ONNX Models       │          │   ┌─────────────┐   ┌───────────────┐  ┌───────────┐   │
│   • Drowsiness/Yawning  │          │   │ BNO085 IMU  │   │  Ultrasonic   │  │   RFID    │   │
└────────────┬────────────┘          │   │ (I2C 21/22) │   │ (GPIO 5 / 18) │  │  (RC522)  │   │
             │ USB                   │   └──────┬──────┘   └───────┬───────┘  └─────┬─────┘   │
             ▼                       └──────────┼──────────────────┼────────────────┼─────────┘
┌───────────────────────────────────────────────┴──────────────────┴────────────────┴─────────┐
│                                    Raspberry Pi 5 Hub                                       │
│                                                                                             │
│  • Reads camera over USB (30–60 FPS)                                                        │
│  • Reads ESP32 over High-Speed Serial / UART at 115200 bps (100 Hz)                         │
│  • Ingests Phone GPS over Wi-Fi (Latitude, Longitude, Speed, Heading)                       │
│  • Runs 4 ONNX Neural Models (Face Detection, 98 Landmarks, Eye State, Mouth/Yawn)          │
│  • Runs IMU Event Classifier (Harsh Braking, Harsh Acceleration, Sharp Cornering)           │
│  • Multi-Sensor Fusion Engine generates unified JSON Payload @ 100 Hz                       │
│  • Logs time-series data & safety events to SQLite Database                                 │
│  • Broadcasts live 100 Hz JSON Stream over Wi-Fi (TCP / WebSocket / HTTP)                   │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │ Wi-Fi (JSON @ 100 Hz)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          Truck Intelligence Hub — Dashboard / Display                       │
│                                                                                             │
│  • Driver Fatigue Status (Normal / Drowsy / Critical with EAR, MAR, Blink & Yawn counters)  │
│  • Real-time IMU Motion Gauges & Charts (Accel X/Y/Z, Gyro X/Y/Z, Roll, Pitch, Yaw)        │
│  • Harsh Driving Maneuver Alerts (Harsh Accel, Brake, Sharp Corner, Vibration)              │
│  • Ultrasonic Fuel Tank Level (% Level, Volume in Litres, Raw distance in cm)               │
│  • Live GPS Location, Speed Gauge, Route Tracking & Geofencing                              │
│  • Driver Identification & Authentication via RFID Card                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Wiring & Interconnects

### ESP32 Multi-Sensor Node
| Sensor | Interface | Sensor Pin | ESP32 GPIO | Description |
| :--- | :--- | :--- | :--- | :--- |
| **BNO085 9-DOF IMU** | I²C | `SDA` | **GPIO 21** | 100 Hz 3-Axis Accel, Gyro & Orientation |
| | I²C | `SCL` | **GPIO 22** | I2C Clock (400 kHz Fast Mode) |
| | Power | `VIN` / `GND` | `3.3V` / `GND` | Breakout Power Supply |
| **HC-SR04 Ultrasonic** | GPIO | `TRIG` | **GPIO 5** | Trigger Pulse (10 µs) |
| | GPIO | `ECHO` | **GPIO 18** | Echo Duration (Fuel level distance) |
| | Power | `VCC` / `GND` | `5V` / `GND` | Voltage supply (use divider on ECHO) |
| **RC522 RFID Reader** | SPI | `SCK` | **GPIO 14** | SPI Clock |
| | SPI | `MISO` | **GPIO 12** | Master In Slave Out |
| | SPI | `MOSI` | **GPIO 13** | Master Out Slave In |
| | SPI | `CS` / `SDA` | **GPIO 15** | Chip Select |
| | Reset | `RST` | **GPIO 4** | Reset Pin |

### ESP32 ➔ Raspberry Pi 5 Serial (UART)
Connect the ESP32 to the Raspberry Pi 5 UART header or USB serial port:
- **ESP32 TX (GPIO 1)** ➔ **Raspberry Pi 5 RX (GPIO 15 / `/dev/serial0`)**
- **ESP32 RX (GPIO 3)** ➔ **Raspberry Pi 5 TX (GPIO 14 / `/dev/serial0`)**
- **Common Ground**: Connect ESP32 `GND` to Raspberry Pi 5 `GND`.
- **Baud Rate**: **115200 bps**

---

## 3. 100 Hz Real-Time JSON Payload Specification

Every 10 milliseconds (100 Hz), the Raspberry Pi 5 Sensor Fusion Engine combines all inputs and broadcasts this standardized JSON payload:

```json
{
  "device_id": "truck_esp32_01",
  "timestamp": "2025-06-12T10:39:45Z",
  "imu": {
    "accel_x": 0.12,
    "accel_y": -0.03,
    "accel_z": 9.81,
    "gyro_x": 0.5,
    "gyro_y": -0.2,
    "gyro_z": 0.1,
    "roll": 1.2,
    "pitch": -0.8,
    "yaw": 45.6
  },
  "fuel": {
    "distance_cm": 15.3,
    "level_percent": 68.0,
    "volume_litres": 102.0
  },
  "rfid": {
    "card_id": "A3F892B1",
    "driver_name": "John Doe",
    "authenticated": true
  },
  "fatigue": {
    "score": 12,
    "state": "NORMAL",
    "ear_left": 0.27,
    "ear_right": 0.26,
    "mar": 0.07,
    "blinks": 18,
    "yawns": 0
  },
  "gps": {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "speed_kmh": 68.0,
    "heading": 45.0
  },
  "events": {
    "driving_state": "normal",
    "latest_event": "none"
  },
  "status": "ok"
}
```

---

## 4. The 4 ONNX Fatigue Neural Networks

Stored in [`models/`](models/):

| Model | Task | Input Shape | Tensor Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`version-slim-320.onnx`** | Face Detection | `[1, 3, 240, 320]` | `[-1.0, 1.0]` | Fast bounding box localization |
| **`PFLD_GhostOne_112_1_opt_sim.onnx`** | 98 Facial Landmarks | `[1, 3, 112, 112]` | `[0.0, 1.0]` | WFLW 98 keypoint regression |
| **`eye_model.onnx`** | Eye State Classifier | `[1, 3, 64, 64]` | `[0.0, 1.0]` | Classifies eye closure |
| **`mouth_model.onnx`** | Mouth / Yawn Classifier | `[1, 3, 64, 64]` | `[0.0, 1.0]` | Classifies yawning and open mouth |

Pipeline throughput on Raspberry Pi 5: **~10.4 ms total latency (~96 FPS)** using ARM NEON SIMD acceleration.

---

## 5. Quickstart & Deployment

### Step 1: Flash ESP32 Firmware
1. Open [`firmware/esp32_sensor_hub.ino`](firmware/esp32_sensor_hub.ino) in Arduino IDE.
2. Install libraries via Library Manager:
   - `Adafruit BNO08x`
   - `MFRC522`
   - `ArduinoJson` (v6 or v7)
3. Select board `DOIT ESP32 DEVKIT V1` and upload.

### Step 2: Set Up Raspberry Pi 5
```bash
git clone https://github.com/dganeshsci-commits/fatigue-model-for-truck.git
cd fatigue-model-for-truck
chmod +x scripts/install.sh
./scripts/install.sh
```

### Step 3: Run Central 100 Hz Processing Hub
```bash
source venv/bin/activate
python hub_server.py
```
- Server starts on `http://0.0.0.0:5000`
- 100 Hz Fused JSON Endpoint: `GET http://<pi_ip>:5000/api/telemetry/latest`
- Live MJPEG Camera Stream with HUD: `GET http://<pi_ip>:5000/camera_feed`
- Phone GPS Ingestion: `POST http://<pi_ip>:5000/api/gps`
- Historical Reports: `GET http://<pi_ip>:5000/api/telemetry/history`

### Step 4: Run Standalone In-Cabin ADAS Display
```bash
python edge_runner.py --camera 0 --esp32-port /dev/serial0
```

---

## 6. Systemd Auto-Start (Launch on Truck Boot)

```bash
sudo cp systemd/truck-fatigue.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now truck-fatigue.service
sudo systemctl status truck-fatigue.service
```

---

## 7. License

MIT License — see [LICENSE](LICENSE) for details.

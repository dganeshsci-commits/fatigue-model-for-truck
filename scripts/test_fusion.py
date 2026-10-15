#!/usr/bin/env python3
"""
Test script to verify 100 Hz Sensor Fusion Hub and JSON payload structure.
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.sensor_fusion import SensorFusionHub


def test_fusion():
    print("=" * 65)
    print(" Testing 100 Hz Sensor Fusion & JSON Payload Generation")
    print("=" * 65)

    hub = SensorFusionHub(device_id="truck_esp32_01", enable_gpio=False, enable_db=False)
    payload = hub.get_fused_payload()

    print("\nGenerated Unified 100 Hz Payload:")
    print(json.dumps(payload, indent=2))

    # Validate mandatory schema fields from architecture diagram
    required_keys = ["device_id", "timestamp", "imu", "fuel", "rfid", "fatigue", "gps", "events", "status"]
    for k in required_keys:
        assert k in payload, f"Missing key: {k}"

    imu_keys = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "roll", "pitch", "yaw"]
    for k in imu_keys:
        assert k in payload["imu"], f"Missing IMU key: {k}"

    fuel_keys = ["distance_cm", "level_percent", "volume_litres"]
    for k in fuel_keys:
        assert k in payload["fuel"], f"Missing fuel key: {k}"

    rfid_keys = ["card_id", "driver_name", "authenticated"]
    for k in rfid_keys:
        assert k in payload["rfid"], f"Missing RFID key: {k}"

    fatigue_keys = ["score", "state", "ear_left", "ear_right", "mar", "blinks", "yawns"]
    for k in fatigue_keys:
        assert k in payload["fatigue"], f"Missing fatigue key: {k}"

    gps_keys = ["latitude", "longitude", "speed_kmh", "heading"]
    for k in gps_keys:
        assert k in payload["gps"], f"Missing GPS key: {k}"

    events_keys = ["driving_state", "latest_event"]
    for k in events_keys:
        assert k in payload["events"], f"Missing events key: {k}"

    print("\n" + "=" * 65)
    print(" Payload Schema Validation: 100% PASSED!")
    print("=" * 65)

    hub.stop()


if __name__ == "__main__":
    test_fusion()

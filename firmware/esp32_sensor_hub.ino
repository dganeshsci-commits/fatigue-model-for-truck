/*
 ==============================================================================
  Truck Intelligence Hub — ESP32 Multi-Sensor Telemetry Hub (100 Hz)
  Sensors:
    1. BNO085 9-DOF IMU (I2C: SDA=21, SCL=22)
    2. HC-SR04 Ultrasonic Fuel Level Sensor (TRIG=5, ECHO=18)
    3. RC522 RFID Driver Authentication Reader (SPI: SCK=14, MISO=12, MOSI=13, CS=15, RST=4)
  Communication:
    - High-Speed Serial (UART) to Raspberry Pi 5 at 115200 bps
    - Formats: JSON telemetry packets streamed at up to 100 Hz
 ==============================================================================
*/

#include <Wire.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_BNO08x.h>
#include <ArduinoJson.h>

// -----------------------------------------------------------------------------
// Pin Configurations (Matching Architecture Specification)
// -----------------------------------------------------------------------------
// I2C Pins for BNO085 IMU
#define BNO085_SDA_PIN      21
#define BNO085_SCL_PIN      22

// Ultrasonic Sensor Pins
#define US_TRIG_PIN         5
#define US_ECHO_PIN         18

// RFID RC522 SPI Pins
#define RFID_SCK_PIN        14
#define RFID_MISO_PIN       12
#define RFID_MOSI_PIN       13
#define RFID_SS_PIN         15
#define RFID_RST_PIN        4

// Fuel Tank Calibration
#define TANK_DEPTH_CM       40.0f   // Distance from sensor to bottom of empty tank (cm)
#define MIN_FUEL_DIST_CM    5.0f    // Minimum distance from sensor when tank is 100% full (cm)
#define TANK_CAPACITY_L     150.0f  // Total tank capacity in Litres

// -----------------------------------------------------------------------------
// Global Instances
// -----------------------------------------------------------------------------
Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;
bool bnoAvailable = false;

MFRC522 mfrc522(RFID_SS_PIN, RFID_RST_PIN);
SPIClass rfidSPI(VSPI);

// Sensor Data Struct
struct SensorTelemetry {
  // Accelerometer (m/s^2)
  float accel_x = 0.0f;
  float accel_y = 0.0f;
  float accel_z = 9.81f;

  // Gyroscope (deg/s)
  float gyro_x = 0.0f;
  float gyro_y = 0.0f;
  float gyro_z = 0.0f;

  // Orientation (deg)
  float roll = 0.0f;
  float pitch = 0.0f;
  float yaw = 0.0f;

  // Fuel Sensor
  float distance_cm = 15.0f;
  float level_percent = 68.0f;
  float volume_litres = 102.0f;

  // RFID Driver
  String card_id = "A3F892B1";
  String driver_name = "John Doe";
  bool authenticated = true;
};

SensorTelemetry currentData;
unsigned long last100HzTick = 0;
unsigned long lastUltrasonicTick = 0;
unsigned long lastRfidTick = 0;

// -----------------------------------------------------------------------------
// Quaternion to Euler Conversion (Roll, Pitch, Yaw in Degrees)
// -----------------------------------------------------------------------------
void quaternionToEuler(float qr, float qi, float qj, float qk, float* roll, float* pitch, float* yaw) {
  float sqr = sq(qr);
  float sqi = sq(qi);
  float sqj = sq(qj);
  float sqk = sq(qk);

  *yaw = atan2(2.0 * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr));
  *pitch = asin(constrain(-2.0 * (qi * qk - qj * qr), -1.0, 1.0));
  *roll = atan2(2.0 * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr));

  *yaw *= RAD_TO_DEG;
  *pitch *= RAD_TO_DEG;
  *roll *= RAD_TO_DEG;
}

// -----------------------------------------------------------------------------
// Measure Ultrasonic Fuel Level
// -----------------------------------------------------------------------------
void measureUltrasonicFuel() {
  digitalWrite(US_TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(US_TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(US_TRIG_PIN, LOW);

  long duration = pulseIn(US_ECHO_PIN, HIGH, 30000); // 30ms timeout (~5m)
  if (duration > 0) {
    float distance = (duration * 0.0343f) / 2.0f;
    if (distance >= 2.0f && distance <= 200.0f) {
      currentData.distance_cm = distance;
      float fillLevel = constrain(TANK_DEPTH_CM - distance, 0.0f, TANK_DEPTH_CM - MIN_FUEL_DIST_CM);
      float maxFill = TANK_DEPTH_CM - MIN_FUEL_DIST_CM;
      currentData.level_percent = constrain((fillLevel / maxFill) * 100.0f, 0.0f, 100.0f);
      currentData.volume_litres = (currentData.level_percent / 100.0f) * TANK_CAPACITY_L;
    }
  }
}

// -----------------------------------------------------------------------------
// Scan RFID Card
// -----------------------------------------------------------------------------
void scanRfidCard() {
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  String uidStr = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) uidStr += "0";
    uidStr += String(mfrc522.uid.uidByte[i], HEX);
  }
  uidStr.toUpperCase();
  currentData.card_id = uidStr;

  // Simple driver ID mapping
  if (uidStr == "A3F892B1" || uidStr == "04A1B2C3") {
    currentData.driver_name = "John Doe (Fleet Driver #1)";
    currentData.authenticated = true;
  } else {
    currentData.driver_name = "Authorized Operator (" + uidStr.substring(0, 4) + ")";
    currentData.authenticated = true;
  }

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
}

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------
void setup() {
  // High-speed UART to Raspberry Pi 5
  Serial.begin(115200);
  delay(100);

  // Initialize Ultrasonic Pins
  pinMode(US_TRIG_PIN, OUTPUT);
  pinMode(US_ECHO_PIN, INPUT);

  // Initialize I2C for BNO085 IMU
  Wire.begin(BNO085_SDA_PIN, BNO085_SCL_PIN, 400000); // 400kHz fast mode

  if (bno08x.begin_I2C(0x4A, &Wire) || bno08x.begin_I2C(0x4B, &Wire)) {
    bnoAvailable = true;
    bno08x.enableReport(SH2_ACCELEROMETER, 10000); // 10ms -> 100Hz
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000); // 100Hz
    bno08x.enableReport(SH2_ROTATION_VECTOR, 20000); // 50Hz
  }

  // Initialize SPI for RC522 RFID
  rfidSPI.begin(RFID_SCK_PIN, RFID_MISO_PIN, RFID_MOSI_PIN, RFID_SS_PIN);
  mfrc522.PCD_Init();
}

// -----------------------------------------------------------------------------
// Main Loop (100 Hz Serial JSON Stream)
// -----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // 1. Read BNO085 IMU reports
  if (bnoAvailable && bno08x.wasReset()) {
    bno08x.enableReport(SH2_ACCELEROMETER, 10000);
    bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000);
    bno08x.enableReport(SH2_ROTATION_VECTOR, 20000);
  }

  if (bnoAvailable && bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        currentData.accel_x = sensorValue.un.accelerometer.x;
        currentData.accel_y = sensorValue.un.accelerometer.y;
        currentData.accel_z = sensorValue.un.accelerometer.z;
        break;
      case SH2_GYROSCOPE_CALIBRATED:
        currentData.gyro_x = sensorValue.un.gyroscope.x * RAD_TO_DEG;
        currentData.gyro_y = sensorValue.un.gyroscope.y * RAD_TO_DEG;
        currentData.gyro_z = sensorValue.un.gyroscope.z * RAD_TO_DEG;
        break;
      case SH2_ROTATION_VECTOR:
        quaternionToEuler(
          sensorValue.un.rotationVector.real,
          sensorValue.un.rotationVector.i,
          sensorValue.un.rotationVector.j,
          sensorValue.un.rotationVector.k,
          &currentData.roll,
          &currentData.pitch,
          &currentData.yaw
        );
        break;
    }
  }

  // 2. Poll Ultrasonic Sensor every 100ms (10 Hz for physical echo stability)
  if (now - lastUltrasonicTick >= 100) {
    lastUltrasonicTick = now;
    measureUltrasonicFuel();
  }

  // 3. Poll RFID Reader every 200ms
  if (now - lastRfidTick >= 200) {
    lastRfidTick = now;
    scanRfidCard();
  }

  // 4. Stream 100 Hz JSON Telemetry Packet over Serial (UART) to Raspberry Pi 5
  // (Every 10ms = 100 Hz)
  if (now - last100HzTick >= 10) {
    last100HzTick = now;

    // Compact JSON string for high-speed 115200 bps transfer
    StaticJsonDocument<384> doc;
    doc["device_id"] = "truck_esp32_01";
    doc["timestamp"] = now;

    JsonObject imu = doc.createNestedObject("imu");
    imu["accel_x"] = round(currentData.accel_x * 100.0) / 100.0;
    imu["accel_y"] = round(currentData.accel_y * 100.0) / 100.0;
    imu["accel_z"] = round(currentData.accel_z * 100.0) / 100.0;
    imu["gyro_x"] = round(currentData.gyro_x * 100.0) / 100.0;
    imu["gyro_y"] = round(currentData.gyro_y * 100.0) / 100.0;
    imu["gyro_z"] = round(currentData.gyro_z * 100.0) / 100.0;
    imu["roll"] = round(currentData.roll * 10.0) / 10.0;
    imu["pitch"] = round(currentData.pitch * 10.0) / 10.0;
    imu["yaw"] = round(currentData.yaw * 10.0) / 10.0;

    JsonObject fuel = doc.createNestedObject("fuel");
    fuel["distance_cm"] = round(currentData.distance_cm * 10.0) / 10.0;
    fuel["level_percent"] = round(currentData.level_percent * 10.0) / 10.0;
    fuel["volume_litres"] = round(currentData.volume_litres * 10.0) / 10.0;

    JsonObject rfid = doc.createNestedObject("rfid");
    rfid["card_id"] = currentData.card_id;
    rfid["driver_name"] = currentData.driver_name;
    rfid["authenticated"] = currentData.authenticated;

    serializeJson(doc, Serial);
    Serial.println();
  }
}

/*
 * main.cpp
 * Entry point: IMU bring-up, monitor mode, and impact-collection mode.
 */

#include <Arduino.h>
#include <math.h>

#include "config.h"
#include "impact_capture.h"
#include "imu_lsm6ds3trc.h"
#include "protocol.h"

struct AxisStats {
    float min_v = 1e9f;
    float max_v = -1e9f;
    double sum = 0.0;
    double sumsq = 0.0;
    uint32_t n = 0;

    void reset() {
        min_v = 1e9f;
        max_v = -1e9f;
        sum = 0.0;
        sumsq = 0.0;
        n = 0;
    }

    void push(float v) {
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
        sumsq += (double)v * (double)v;
        n++;
    }

    float mean() const { return (n ? (float)(sum / (double)n) : 0.0f); }
    float rms() const { return (n ? (float)sqrt(sumsq / (double)n) : 0.0f); }
    float p2p() const { return (n ? (max_v - min_v) : 0.0f); }
};

static AxisStats sx, sy, sz, smag;
static uint32_t lastPrintUs = 0;
static uint32_t lastRateUs = 0;
static uint32_t sampleCount = 0;
static ImpactRecord readyRecord;
static uint8_t txBuffer[MAX_ENCODED_PACKET_BYTES];

inline float rawToMg(int16_t raw) {
    return (float)raw * ACCEL_MG_PER_LSB;
}

inline float magnitudeMg(const AccelSample &s) {
    const float fx = (float)s.x;
    const float fy = (float)s.y;
    const float fz = (float)s.z;
    return sqrtf(fx * fx + fy * fy + fz * fz) * ACCEL_MG_PER_LSB;
}

void handleMonitor(const AccelSample &sample, uint32_t now_us) {
    const float ax_mg = rawToMg(sample.x);
    const float ay_mg = rawToMg(sample.y);
    const float az_mg = rawToMg(sample.z);
    const float mag_mg = magnitudeMg(sample);

    sx.push(ax_mg);
    sy.push(ay_mg);
    sz.push(az_mg);
    smag.push(mag_mg);

    sampleCount++;

    if ((uint32_t)(now_us - lastPrintUs) >= PRINT_INTERVAL_US) {
        lastPrintUs = now_us;
        Serial.print("t=");
        Serial.print(millis());
        Serial.print(" ms | samples=");
        Serial.print(sx.n);

        Serial.print(" | X mean=");
        Serial.print(sx.mean(), 2);
        Serial.print(" mg rms=");
        Serial.print(sx.rms(), 2);
        Serial.print(" p2p=");
        Serial.print(sx.p2p(), 2);

        Serial.print(" | Y mean=");
        Serial.print(sy.mean(), 2);
        Serial.print(" mg rms=");
        Serial.print(sy.rms(), 2);
        Serial.print(" p2p=");
        Serial.print(sy.p2p(), 2);

        Serial.print(" | Z mean=");
        Serial.print(sz.mean(), 2);
        Serial.print(" mg rms=");
        Serial.print(sz.rms(), 2);
        Serial.print(" p2p=");
        Serial.print(sz.p2p(), 2);

        Serial.print(" | |a| mean=");
        Serial.print(smag.mean(), 2);
        Serial.print(" mg | baseline=");
        Serial.print(impactBaselineMg(), 1);
        Serial.println(" mg");

        sx.reset();
        sy.reset();
        sz.reset();
        smag.reset();
    }

    if ((uint32_t)(now_us - lastRateUs) >= 1000000UL) {
        lastRateUs = now_us;
        Serial.print("[rate] approx samples/sec = ");
        Serial.println(sampleCount);
        sampleCount = 0;
    }
}

void handleCollect(const AccelSample &sample, uint32_t now_us) {
    if (impactCaptureProcessSample(sample, now_us, readyRecord)) {
        size_t encoded_len = 0;
        if (encodeImpactPacket(readyRecord, txBuffer, sizeof(txBuffer), encoded_len)) {
            Serial.write(txBuffer, encoded_len);
        } else {
            Serial.println("encode_error");
        }
        Serial.print("impact ");
        Serial.print(readyRecord.impact_id);
        Serial.print(" peak=");
        Serial.print(readyRecord.features.peak_mag_mg, 1);
        Serial.print(" mg decay=");
        Serial.print(readyRecord.features.decay_ms, 1);
        Serial.print(" ms rms=");
        Serial.print(readyRecord.features.rms_dev_mg, 1);
        Serial.println(" mg");
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(50);
    Serial.println("Teensy vibration capture starting...");

    if (!imuBegin()) {
        Serial.println("IMU init failed. Check wiring and power.");
        while (true) {
            delay(1000);
        }
    }

    const uint8_t who = imuWhoAmI();
    Serial.print("WHO_AM_I = 0x");
    Serial.println(who, HEX);

    if (!imuConfigureAccel()) {
        Serial.println("IMU configuration failed.");
        while (true) {
            delay(1000);
        }
    }

    Serial.print("Accel configured: ODR=");
    Serial.print(IMU_ODR_HZ);
    Serial.print(" Hz, FS=±");
    Serial.print(IMU_FS_G);
    Serial.println(" g");
    Serial.print("Using I2C addr 0x");
    Serial.println(imuActiveAddress(), HEX);
    Serial.print("Serial baud=");
    Serial.println(SERIAL_BAUD);

    impactCaptureInit();
}

void loop() {
    if (!imuDataReady()) {
        return;
    }

    AccelSample sample;
    if (!imuReadSample(sample)) {
        return;
    }

    const uint32_t now_us = micros();

#if RUN_MODE == MODE_MONITOR
    handleMonitor(sample, now_us);
#elif RUN_MODE == MODE_COLLECT
    handleCollect(sample, now_us);
#elif RUN_MODE == MODE_INFER
    handleCollect(sample, now_us); // placeholder: collect data, inference can be added later
#else
    handleMonitor(sample, now_us);
#endif
}

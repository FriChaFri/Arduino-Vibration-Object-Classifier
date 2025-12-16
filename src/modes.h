/*
 * modes.h
 * “Monitor mode” implementation matching your old main.cpp:
 *  - read accel as fast as possible
 *  - accumulate stats
 *  - print summary at PRINT_HZ
 *  - print reads/sec once per second
 */

#pragma once

#include <Arduino.h>
#include <math.h>
#include "config.h"
#include "imu_lsm6ds3.h"
#include "features.h"

// ---------------- Stats struct (same behavior as old file) ----------------
struct AxisStats {
    float min_mg =  1e9f;
    float max_mg = -1e9f;
    double sum_mg = 0.0;
    double sumsq_mg = 0.0;
    uint32_t n = 0;

    void reset() {
        min_mg =  1e9f;
        max_mg = -1e9f;
        sum_mg = 0.0;
        sumsq_mg = 0.0;
        n = 0;
    }

    void push(float v_mg) {
        if (v_mg < min_mg) min_mg = v_mg;
        if (v_mg > max_mg) max_mg = v_mg;
        sum_mg += v_mg;
        sumsq_mg += (double)v_mg * (double)v_mg;
        n++;
    }

    float mean() const { return (n ? (float)(sum_mg / (double)n) : 0.0f); }
    float rms()  const { return (n ? (float)sqrt(sumsq_mg / (double)n) : 0.0f); }
    float p2p()  const { return (n ? (max_mg - min_mg) : 0.0f); }
};

static inline float accelMagMg(float ax_mg, float ay_mg, float az_mg) {
    return sqrtf(ax_mg * ax_mg + ay_mg * ay_mg + az_mg * az_mg);
}

// ---------------- Monitor loop (call repeatedly from loop()) ----------------
static inline void monitorLoop() {
#if ENABLE_CONTINUOUS_STATS
    static AxisStats sx, sy, sz;
    static uint32_t lastPrintUs = 0;
    static uint32_t sampleCount = 0;
    static uint32_t lastRatePrintUs = 0;
#endif
    static float captureBuffer[WINDOW_SAMPLES];
    static size_t captureIndex = 0;
    static bool capturing = false;
    static bool triggerArmed = true;
    static uint32_t captureTimestampMs = 0;
    static uint32_t lastImpactMs = 0;
    static uint32_t impactIdCounter = 0;
    const float releaseThreshold = (TRIGGER_MG > HYST_MG) ? (TRIGGER_MG - HYST_MG) : 0.0f;

    // Read accel as fast as practical (I2C is limiting factor)
    float ax_mg, ay_mg, az_mg;
    imuReadAccelMg(ax_mg, ay_mg, az_mg);

    float mag_mg = accelMagMg(ax_mg, ay_mg, az_mg);
    float delta_mg = fabsf(mag_mg - IMPACT_BASELINE_MG);
    uint32_t nowMs = millis();
    bool inCooldown = (impactIdCounter > 0) && ((uint32_t)(nowMs - lastImpactMs) < COOLDOWN_MS);

    if (!capturing && !inCooldown && triggerArmed && delta_mg >= TRIGGER_MG) {
        capturing = true;
        captureIndex = 0;
        captureTimestampMs = nowMs;
        triggerArmed = false;
    } else if (!capturing && delta_mg <= releaseThreshold) {
        triggerArmed = true;
    }

    if (capturing) {
        if (captureIndex < WINDOW_SAMPLES) {
            captureBuffer[captureIndex++] = mag_mg;
        }
        if (captureIndex >= WINDOW_SAMPLES) {
            capturing = false;
            lastImpactMs = nowMs;
            FeatureVector fv = computeFeatures(captureBuffer, WINDOW_SAMPLES);
            uint32_t impactId = impactIdCounter++;
            Serial.print(impactId);
            Serial.print(',');
            Serial.print(captureTimestampMs);
            Serial.print(',');
            Serial.print((uint32_t)WINDOW_SAMPLES);
            Serial.print(',');
            Serial.print(fv.peak, 2);
            Serial.print(',');
            Serial.print(fv.rms, 2);
            Serial.print(',');
            Serial.print(fv.p2p, 2);
            Serial.print(',');
            Serial.println(fv.decay_ratio, 3);

#if ENABLE_WAVEFORM_OUTPUT
            Serial.print("waveform,");
            Serial.print(impactId);
            Serial.print(',');
            Serial.print(captureTimestampMs);
            for (size_t i = 0; i < WINDOW_SAMPLES; ++i) {
                if ((i % WAVEFORM_DECIMATE) != 0) {
                    continue;
                }
                Serial.print(',');
                Serial.print(captureBuffer[i], 2);
            }
            Serial.println();
#endif
        }
    }

#if ENABLE_CONTINUOUS_STATS
    sampleCount++;

    sx.push(ax_mg);
    sy.push(ay_mg);
    sz.push(az_mg);

    uint32_t nowUs = micros();

    // Summary print at controlled rate
    if ((uint32_t)(nowUs - lastPrintUs) >= PRINT_INTERVAL_US) {
        lastPrintUs = nowUs;

        float ax_g = sx.mean() / 1000.0f;
        float ay_g = sy.mean() / 1000.0f;
        float az_g = sz.mean() / 1000.0f;

        Serial.print("t=");
        Serial.print(millis());
        Serial.print(" ms | samples=");
        Serial.print(sx.n);

        Serial.print(" | X: mean=");
        Serial.print(sx.mean(), 2);
        Serial.print(" mg, rms=");
        Serial.print(sx.rms(), 2);
        Serial.print(" mg, p2p=");
        Serial.print(sx.p2p(), 2);
        Serial.print(" mg");

        Serial.print(" | Y: mean=");
        Serial.print(sy.mean(), 2);
        Serial.print(" mg, rms=");
        Serial.print(sy.rms(), 2);
        Serial.print(" mg, p2p=");
        Serial.print(sy.p2p(), 2);
        Serial.print(" mg");

        Serial.print(" | Z: mean=");
        Serial.print(sz.mean(), 2);
        Serial.print(" mg, rms=");
        Serial.print(sz.rms(), 2);
        Serial.print(" mg, p2p=");
        Serial.print(sz.p2p(), 2);
        Serial.print(" mg");

        Serial.print(" | mean_g=(");
        Serial.print(ax_g, 5); Serial.print(", ");
        Serial.print(ay_g, 5); Serial.print(", ");
        Serial.print(az_g, 5); Serial.print(")");

        Serial.print(" | mean_m/s^2=(");
        Serial.print(ax_g * G_TO_MPS2, 4); Serial.print(", ");
        Serial.print(ay_g * G_TO_MPS2, 4); Serial.print(", ");
        Serial.print(az_g * G_TO_MPS2, 4); Serial.print(")");

        Serial.println();

        sx.reset();
        sy.reset();
        sz.reset();
    }

    // Throughput line once per second
    if ((uint32_t)(nowUs - lastRatePrintUs) >= 1000000UL) {
        lastRatePrintUs = nowUs;
        Serial.print("[rate] approx reads/sec = ");
        Serial.println(sampleCount);
        sampleCount = 0;
    }
#endif
}

// Stubs for other modes (so main.cpp compiles if you switch RUN_MODE)
static inline void collectLoop() {
    // TODO: dataset capture mode (impact trigger + window + CSV)
    monitorLoop(); // placeholder
}

static inline void inferLoop() {
    // TODO: inference mode (impact trigger + window + features + classify)
    monitorLoop(); // placeholder
}

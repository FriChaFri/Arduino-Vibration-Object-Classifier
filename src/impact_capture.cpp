/*
 * impact_capture.cpp
 * Impact trigger, two-stage capture, and feature computation.
 */

#include "impact_capture.h"

#include <math.h>
#include <string.h>

#include "config.h"

namespace {
enum class CaptureState {
    Idle,
    Stage1,
    Stage2,
    Refractory
};

CaptureState state = CaptureState::Idle;
bool triggerArmed = true;

AccelSample preBuffer[PRETRIGGER_SAMPLES];
size_t preCount = 0;
size_t preHead = 0;

ImpactRecord currentRecord;
uint32_t impactCounter = 0;
uint32_t lockoutStartMs = 0;
float baselineMg = BASELINE_INIT_MG;
uint32_t stage2DecimCounter = 0;

inline float magnitudeMg(const AccelSample &s) {
    const float fx = (float)s.x;
    const float fy = (float)s.y;
    const float fz = (float)s.z;
    return sqrtf(fx * fx + fy * fy + fz * fz) * ACCEL_MG_PER_LSB;
}

void resetPretrigger() {
    preCount = 0;
    preHead = 0;
}

void pushPretrigger(const AccelSample &s) {
    if (PRETRIGGER_SAMPLES == 0) {
        return;
    }
    preBuffer[preHead] = s;
    preHead = (preHead + 1U) % PRETRIGGER_SAMPLES;
    if (preCount < PRETRIGGER_SAMPLES) {
        preCount++;
    }
}

void startCapture(const AccelSample &triggerSample, uint32_t timestamp_us) {
    currentRecord = ImpactRecord{};
    currentRecord.impact_id = impactCounter++;
    currentRecord.trigger_time_us = timestamp_us;
    currentRecord.stage2_decimation = STAGE2_DECIMATION;
    currentRecord.baseline_mg = baselineMg;

    // Copy pretrigger samples oldest -> newest into stage 1.
    const size_t toCopy = preCount;
    if (toCopy > 0) {
        const size_t start = (preHead + PRETRIGGER_SAMPLES - toCopy) % PRETRIGGER_SAMPLES;
        for (size_t i = 0; i < toCopy; ++i) {
            const size_t idx = (start + i) % PRETRIGGER_SAMPLES;
            currentRecord.stage1[i] = preBuffer[idx];
        }
    }
    currentRecord.pretrigger_recorded = (uint16_t)toCopy;
    currentRecord.stage1_count = (uint16_t)toCopy;

    // Store the trigger sample as the first post-trigger entry.
    if (currentRecord.stage1_count < STAGE1_SAMPLES) {
        currentRecord.stage1[currentRecord.stage1_count++] = triggerSample;
    }

    stage2DecimCounter = 0;
    state = CaptureState::Stage1;
}

bool finalizeCapture(ImpactRecord &out_record) {
    computeFeatures(currentRecord.stage1,
                    currentRecord.stage1_count,
                    currentRecord.stage2,
                    currentRecord.stage2_count,
                    currentRecord.stage2_decimation,
                    currentRecord.pretrigger_recorded,
                    currentRecord.baseline_mg,
                    currentRecord.features);
    out_record = currentRecord; // copy out (small, infrequent)
    lockoutStartMs = millis();
    state = CaptureState::Refractory;
    resetPretrigger();
    return true;
}
} // namespace

void impactCaptureInit() {
    baselineMg = BASELINE_INIT_MG;
    impactCounter = 0;
    lockoutStartMs = 0;
    stage2DecimCounter = 0;
    triggerArmed = true;
    resetPretrigger();
    state = CaptureState::Idle;
}

float impactBaselineMg() {
    return baselineMg;
}

bool impactCaptureProcessSample(const AccelSample &sample, uint32_t sample_time_us, ImpactRecord &out_record) {
    // Refractory handling
    if (state == CaptureState::Refractory) {
        const uint32_t elapsed = millis() - lockoutStartMs;
        if (elapsed >= REFRACTORY_MS) {
            state = CaptureState::Idle;
            triggerArmed = false; // wait for hysteresis release
        }
    }

    const float mag_mg = magnitudeMg(sample);

    // Baseline EMA when idle or refractory to track slow drift.
    if (state == CaptureState::Idle || state == CaptureState::Refractory) {
        baselineMg += (mag_mg - baselineMg) * BASELINE_ALPHA;
    }

    // Maintain pretrigger buffer whenever we are not mid-capture.
    if (state != CaptureState::Stage1 && state != CaptureState::Stage2) {
        pushPretrigger(sample);
    }

    // Trigger detection (idle only)
    if (state == CaptureState::Idle) {
        const float delta_mg = fabsf(mag_mg - baselineMg);
        const float release = (TRIGGER_MG > HYST_MG) ? (TRIGGER_MG - HYST_MG) : 0.0f;
        if (!triggerArmed && delta_mg <= release) {
            triggerArmed = true;
        }
        if (triggerArmed && delta_mg >= TRIGGER_MG) {
            triggerArmed = false;
            startCapture(sample, sample_time_us);
            return false;
        }
        return false;
    }

    // Stage 1: full-rate capture
    if (state == CaptureState::Stage1) {
        if (currentRecord.stage1_count < STAGE1_SAMPLES) {
            currentRecord.stage1[currentRecord.stage1_count++] = sample;
        }
        if (currentRecord.stage1_count >= STAGE1_SAMPLES) {
            state = CaptureState::Stage2;
        }
        return false;
    }

    // Stage 2: decimated tail
    if (state == CaptureState::Stage2) {
        stage2DecimCounter++;
        if (stage2DecimCounter >= currentRecord.stage2_decimation) {
            stage2DecimCounter = 0;
            if (currentRecord.stage2_count < STAGE2_SAMPLES) {
                currentRecord.stage2[currentRecord.stage2_count++] = sample;
            }
        }
        if (currentRecord.stage2_count >= STAGE2_SAMPLES) {
            return finalizeCapture(out_record);
        }
    }

    return false;
}

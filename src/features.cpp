/*
 * features.cpp
 * Deterministic feature extraction routines.
 */

#include "features.h"

#include <math.h>

#include "config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
inline float magMg(const AccelSample &s) {
    const float fx = (float)s.x;
    const float fy = (float)s.y;
    const float fz = (float)s.z;
    return sqrtf(fx * fx + fy * fy + fz * fz) * ACCEL_MG_PER_LSB;
}

float goertzelEnergy(const float *samples, size_t n, float target_hz, float fs_hz) {
    if (n == 0) {
        return 0.0f;
    }
    const float k = (float)n * target_hz / fs_hz;
    const float omega = 2.0f * (float)M_PI * k / (float)n;
    const float coeff = 2.0f * cosf(omega);
    float s_prev = 0.0f;
    float s_prev2 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float s = samples[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    return s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2;
}
} // namespace

void computeFeatures(const AccelSample *stage1,
                     size_t stage1_count,
                     const AccelSample *stage2,
                     size_t stage2_count,
                     uint8_t stage2_decimation,
                     uint16_t /*pretrigger_count*/,
                     float baseline_mg,
                     FeatureVector &out) {
    const float sample_dt_ms = 1000.0f / (float)IMU_ODR_HZ;
    const size_t n1 = stage1_count > STAGE1_SAMPLES ? STAGE1_SAMPLES : stage1_count;
    const size_t n2 = stage2_count > STAGE2_SAMPLES ? STAGE2_SAMPLES : stage2_count;

    static float mag1[STAGE1_SAMPLES];

    float peak_mag = 0.0f;
    uint32_t peak_index_samples = 0;

    double sumsq_dev = 0.0;
    for (size_t i = 0; i < n1; ++i) {
        const float mag = magMg(stage1[i]);
        mag1[i] = mag;
        const float dev = mag - baseline_mg;
        const float abs_mag = fabsf(mag);
        if (abs_mag > peak_mag) {
            peak_mag = abs_mag;
            peak_index_samples = (uint32_t)i;
        }
        sumsq_dev += (double)dev * (double)dev;
    }

    // Extend peak search into stage 2 (decimated tail).
    for (size_t i = 0; i < n2; ++i) {
        const float mag = magMg(stage2[i]);
        const float abs_mag = fabsf(mag);
        if (abs_mag > peak_mag) {
            peak_mag = abs_mag;
            peak_index_samples = (uint32_t)(n1 + i * stage2_decimation);
        }
    }

    out.peak_mag_mg = peak_mag;
    out.peak_dev_mg = peak_mag - baseline_mg;
    if (n1 > 0) {
        out.rms_dev_mg = (float)sqrt(sumsq_dev / (double)n1);
    }

    const float decay_target = peak_mag * DECAY_FRACTION;
    uint32_t decay_index_samples = n1 + (uint32_t)(n2 * stage2_decimation);
    // Scan stage 1 then stage 2 for first sample after peak below target.
    for (size_t i = peak_index_samples + 1; i < n1; ++i) {
        if (mag1[i] <= decay_target) {
            decay_index_samples = (uint32_t)i;
            break;
        }
    }
    if (decay_index_samples == n1 + (uint32_t)(n2 * stage2_decimation)) {
        for (size_t i = 0; i < n2; ++i) {
            const uint32_t global_idx = (uint32_t)(n1 + i * stage2_decimation);
            if (global_idx <= peak_index_samples) {
                continue;
            }
            const float mag = magMg(stage2[i]);
            if (mag <= decay_target) {
                decay_index_samples = global_idx;
                break;
            }
        }
    }
    const uint32_t decay_delta_samples = (decay_index_samples > peak_index_samples)
                                             ? (decay_index_samples - peak_index_samples)
                                             : 0;
    out.decay_ms = (float)decay_delta_samples * sample_dt_ms;

    // Band energies via Goertzel using stage 1 deviation series.
    for (uint8_t b = 0; b < NUM_BAND_FEATURES; ++b) {
        static float scratch[STAGE1_SAMPLES];
        for (size_t i = 0; i < n1; ++i) {
            scratch[i] = mag1[i] - baseline_mg;
        }
        out.band_energy[b] = goertzelEnergy(scratch, n1, BAND_FREQS_HZ[b], (float)IMU_ODR_HZ);
    }
}

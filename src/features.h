/*
 * features.h
 * Low-cost feature extraction for impact captures.
 *
 * Features:
 *  - peak magnitude (mg)
 *  - peak deviation from baseline (mg)
 *  - RMS deviation over stage 1 (mg)
 *  - decay time (ms) for envelope to fall below peak * DECAY_FRACTION
 *  - narrow-band Goertzel energy at BAND_FREQS_HZ (mg^2)
 */

#pragma once

#include <Arduino.h>
#include <stddef.h>

#include "config.h"
#include "imu_lsm6ds3trc.h"

struct FeatureVector {
    float peak_mag_mg = 0.0f;
    float peak_dev_mg = 0.0f;
    float rms_dev_mg  = 0.0f;
    float decay_ms    = 0.0f;
    float band_energy[NUM_BAND_FEATURES] = {0};
};

void computeFeatures(const AccelSample *stage1,
                     size_t stage1_count,
                     const AccelSample *stage2,
                     size_t stage2_count,
                     uint8_t stage2_decimation,
                     uint16_t pretrigger_count,
                     float baseline_mg,
                     FeatureVector &out);

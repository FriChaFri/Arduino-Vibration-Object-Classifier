/*
 * impact_capture.h
 * Trigger + staged capture with pretrigger buffer and decimated tail.
 */

#pragma once

#include <Arduino.h>

#include "config.h"
#include "features.h"
#include "imu_lsm6ds3trc.h"

struct ImpactRecord {
    uint32_t impact_id = 0;
    uint32_t trigger_time_us = 0;
    uint16_t pretrigger_recorded = 0;
    uint16_t stage1_count = 0;
    uint16_t stage2_count = 0;
    uint8_t  stage2_decimation = STAGE2_DECIMATION;
    float    baseline_mg = BASELINE_INIT_MG;
    FeatureVector features;
    AccelSample stage1[STAGE1_SAMPLES];
    AccelSample stage2[STAGE2_SAMPLES];
};

void impactCaptureInit();
bool impactCaptureProcessSample(const AccelSample &sample, uint32_t sample_time_us, ImpactRecord &out_record);
float impactBaselineMg();


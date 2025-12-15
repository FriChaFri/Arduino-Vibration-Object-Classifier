/*
 * features.h
 * Impact detection, capture window, and feature extraction.
 */

#pragma once
#include <Arduino.h>
#include <math.h>
#include "config.h"

struct FeatureVector {
    float rms;
    float peak;
    float p2p;
};

inline bool detectImpact(float ax, float ay, float az) {
    float mag = sqrtf(ax*ax + ay*ay + az*az);
    return mag > IMPACT_THRESHOLD_MG;
}

inline FeatureVector computeFeatures(const float *buffer, size_t n) {
    FeatureVector f = {};
    // TODO: compute RMS, peak, peak-to-peak
    return f;
}

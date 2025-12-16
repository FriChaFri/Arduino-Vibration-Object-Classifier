/*
 * features.h
 * Impact detection, capture window, and feature extraction.
 */

#pragma once
#include <Arduino.h>
#include <math.h>
#include "config.h"

struct FeatureVector {
    float rms = 0.0f;
    float peak = 0.0f;
    float p2p = 0.0f;
    float decay_ratio = 0.0f;
};

inline FeatureVector computeFeatures(const float *buffer, size_t n) {
    FeatureVector f;
    if (!buffer || n == 0) {
        return f;
    }

    double sumsq = 0.0;
    float min_v = buffer[0];
    float max_v = buffer[0];

    for (size_t i = 0; i < n; ++i) {
        float v = buffer[i];
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
        sumsq += (double)v * (double)v;
    }

    f.peak = max_v;
    f.p2p = max_v - min_v;
    f.rms = (float)sqrt(sumsq / (double)n);

    size_t quarter = n / 4;
    if (quarter == 0) {
        quarter = 1;
    }

    double sumsq_first = 0.0;
    double sumsq_last = 0.0;
    for (size_t i = 0; i < quarter; ++i) {
        float v = buffer[i];
        sumsq_first += (double)v * (double)v;
    }
    for (size_t i = n - quarter; i < n; ++i) {
        float v = buffer[i];
        sumsq_last += (double)v * (double)v;
    }

    float rms_first = (float)sqrt(sumsq_first / (double)quarter);
    float rms_last = (float)sqrt(sumsq_last / (double)quarter);
    f.decay_ratio = rms_first / (rms_last + DECAY_EPSILON);

    return f;
}

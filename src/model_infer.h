/*
 * model_infer.h
 * Lightweight inference helpers for on-device MLP execution.
 */

#pragma once

#include <stddef.h>

#include "features.h"
#include "model_weights.h"

namespace model_infer {

struct PredictResult {
    int class_index = -1;
    float probabilities[model_weights::kNumClasses] = {0};
};

const char *outputTypeString();

bool preprocess(const FeatureVector &features,
                float ordered_features_out[model_weights::kInputDim],
                float normalized_out[model_weights::kInputDim],
                bool &imputed);

bool forward(const float *input, float *output, size_t &output_dim);

bool postprocess(const float *logits, size_t logits_dim, PredictResult &result);

bool predict(const FeatureVector &features,
             PredictResult &result,
             float ordered_features_out[model_weights::kInputDim] = nullptr,
             bool *imputed = nullptr);

void printModelBanner();

} // namespace model_infer

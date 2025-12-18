// Placeholder header.
// This file is overwritten by scripts/export_model.py after training.
//
// Author:
//     ChatGPT5: EAGER project context
//     Reviewed and approved by Caleb Hottes.

#pragma once
#include <cstddef>

namespace model_weights {

constexpr bool kIsTrained = false;

constexpr std::size_t kInputDim = 1;
constexpr std::size_t kNumClasses = 1;
constexpr std::size_t kNumLayers = 1;

// Output interpretation:
// - kBinaryLogit: final neuron logistic -> P(class[kLogitPositiveClass]); other class = 1 - P.
// - kMultiClass: apply softmax to final layer logits; probabilities align with kClassNames order.
enum class OutputType { kBinaryLogit, kMultiClass };
constexpr OutputType kOutputType = OutputType::kMultiClass;
constexpr std::size_t kLogitPositiveClass = 0;

static const char* const kClassNames[kNumClasses] = {"untrained"};
static const char* const kFeatureNames[kInputDim] = {"placeholder"};

static const float kImputerMedian[kInputDim] = {0.0f};
static const float kScalerMean[kInputDim] = {0.0f};
static const float kScalerScale[kInputDim] = {1.0f};

struct Layer {
    std::size_t input_dim;
    std::size_t output_dim;
    const float* weights; // row-major [output][input]
    const float* biases;  // length == output_dim
};

static const float kLayer0Weights[] = {0.0f};
static const float kLayer0Bias[] = {0.0f};

static const Layer kLayers[kNumLayers] = {
    {1, 1, kLayer0Weights, kLayer0Bias},
};

} // namespace model_weights

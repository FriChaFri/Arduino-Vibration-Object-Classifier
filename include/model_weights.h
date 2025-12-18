// Auto-generated placeholder. Run scripts/export_model.py after training to regenerate with real parameters.
#pragma once
#include <cstddef>

namespace model_weights {

constexpr std::size_t kInputDim = 1;
constexpr std::size_t kNumClasses = 1;
constexpr std::size_t kNumLayers = 1;

// Output interpretation:
// - kBinaryLogit: final neuron logistic → P(class[kLogitPositiveClass]), other class = 1 - P.
// - kMultiClass: apply softmax to the final layer outputs to align with kClassNames order.
enum class OutputType { kBinaryLogit, kMultiClass };
constexpr OutputType kOutputType = OutputType::kBinaryLogit;
constexpr std::size_t kLogitPositiveClass = 0;  // valid only when kOutputType == OutputType::kBinaryLogit

static const char* const kClassNames[kNumClasses] = {"untrained"};
static const char* const kFeatureNames[kInputDim] = {"placeholder"};

static const float kImputerMedian[kInputDim] = {0.0f};
static const float kScalerMean[kInputDim] = {0.0f};
static const float kScalerScale[kInputDim] = {1.0f};

struct Layer {
    std::size_t input_dim;
    std::size_t output_dim;
    const float* weights;  // row-major [output][input]
    const float* biases;   // length == output_dim
};

static const float kLayer0Weights[] = {0.0f};
static const float kLayer0Bias[] = {0.0f};

static const Layer kLayers[kNumLayers] = {
    {1, 1, kLayer0Weights, kLayer0Bias},
};

}  // namespace model_weights

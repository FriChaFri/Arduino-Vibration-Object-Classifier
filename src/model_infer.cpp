/*
 * model_infer.cpp
 * Deterministic preprocessing, forward pass, and postprocessing for the
 * exported MLP weights in include/model_weights.h.
 */

#include "model_infer.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "config.h"

namespace model_infer {
namespace {
constexpr size_t kMaxLayerWidth = 64;
static_assert(model_weights::kInputDim <= kMaxLayerWidth,
              "Input dimension exceeds static buffer");

inline bool validNumber(float v) {
    return !isnan(v) && !isinf(v);
}

bool featureByName(const FeatureVector &f, const char *name, float &out_value) {
    if (!name) {
        return false;
    }
    if (strncmp(name, "band_", 5) == 0) {
        int idx = 0;
        const char *p = name + 5;
        if (*p == '\0') {
            return false;
        }
        while (*p) {
            if (*p < '0' || *p > '9') {
                return false;
            }
            idx = idx * 10 + (*p - '0');
            ++p;
        }
        if (idx < 0 || idx >= (int)NUM_BAND_FEATURES) {
            return false;
        }
        out_value = f.band_energy[idx];
        return true;
    }
    if (strcmp(name, "decay_ms") == 0) {
        out_value = f.decay_ms;
        return true;
    }
    if (strcmp(name, "peak_dev_mg") == 0) {
        out_value = f.peak_dev_mg;
        return true;
    }
    if (strcmp(name, "peak_mag_mg") == 0) {
        out_value = f.peak_mag_mg;
        return true;
    }
    if (strcmp(name, "rms_dev_mg") == 0) {
        out_value = f.rms_dev_mg;
        return true;
    }
    return false;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}
} // namespace

const char *outputTypeString() {
    switch (model_weights::kOutputType) {
    case model_weights::OutputType::kBinaryLogit:
        return "binary_logit";
    case model_weights::OutputType::kMultiClass:
        return "multi_class";
    default:
        return "unknown";
    }
}

bool preprocess(const FeatureVector &features,
                float ordered_features_out[model_weights::kInputDim],
                float normalized_out[model_weights::kInputDim],
                bool &imputed) {
    imputed = false;
    for (size_t i = 0; i < model_weights::kInputDim; ++i) {
        const char *fname = model_weights::kFeatureNames[i];
        float value = 0.0f;
        if (!featureByName(features, fname, value)) {
            Serial.print("infer_error unknown_feature ");
            Serial.println(fname ? fname : "(null)");
            return false;
        }
        if (!validNumber(value)) {
            value = model_weights::kImputerMedian[i];
            imputed = true;
        }
        ordered_features_out[i] = value;
        const float scale = model_weights::kScalerScale[i];
        if (scale == 0.0f) {
            Serial.print("infer_error bad_scale feature=");
            Serial.println(fname ? fname : "(null)");
            return false;
        }
        normalized_out[i] = (value - model_weights::kScalerMean[i]) / scale;
    }
    return true;
}

bool forward(const float *input, float *output, size_t &output_dim) {
    static float act0[kMaxLayerWidth];
    static float act1[kMaxLayerWidth];

    const float *cur = input;
    size_t cur_dim = model_weights::kInputDim;
    float *next = act0;

    for (size_t li = 0; li < model_weights::kNumLayers; ++li) {
        const model_weights::Layer &layer = model_weights::kLayers[li];
        if (!layer.weights || !layer.biases) {
            Serial.println("infer_error null_weights");
            return false;
        }
        if (layer.input_dim != cur_dim) {
            Serial.println("infer_error dim_mismatch");
            return false;
        }
        if (layer.output_dim > kMaxLayerWidth) {
            Serial.println("infer_error layer_too_wide");
            return false;
        }
        for (size_t o = 0; o < layer.output_dim; ++o) {
            float acc = layer.biases[o];
            const size_t base = o * layer.input_dim;
            for (size_t in = 0; in < layer.input_dim; ++in) {
                acc += layer.weights[base + in] * cur[in];
            }
            if (li + 1 < model_weights::kNumLayers && acc < 0.0f) {
                acc = 0.0f; // ReLU on hidden layers
            }
            next[o] = acc;
        }
        cur = next;
        cur_dim = layer.output_dim;
        next = (next == act0) ? act1 : act0;
    }

    for (size_t i = 0; i < cur_dim; ++i) {
        output[i] = cur[i];
    }
    output_dim = cur_dim;
    return true;
}

bool postprocess(const float *logits, size_t logits_dim, PredictResult &result) {
    for (size_t i = 0; i < model_weights::kNumClasses; ++i) {
        result.probabilities[i] = 0.0f;
    }

    if (model_weights::kOutputType == model_weights::OutputType::kBinaryLogit) {
        if (model_weights::kNumClasses != 2 || logits_dim != 1) {
            Serial.println("infer_error bad_binary_shape");
            return false;
        }
        if (model_weights::kLogitPositiveClass >= model_weights::kNumClasses) {
            Serial.println("infer_error bad_positive_class");
            return false;
        }
        const float p_pos = sigmoidf(logits[0]);
        const size_t pos_idx = model_weights::kLogitPositiveClass;
        const size_t neg_idx = pos_idx == 0 ? 1 : 0;
        result.probabilities[pos_idx] = p_pos;
        result.probabilities[neg_idx] = 1.0f - p_pos;
        result.class_index = (result.probabilities[0] >= result.probabilities[1]) ? 0 : 1;
        return true;
    }

    if (logits_dim != model_weights::kNumClasses) {
        Serial.println("infer_error bad_multiclass_shape");
        return false;
    }

    float max_logit = logits[0];
    for (size_t i = 1; i < logits_dim; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    float sum = 0.0f;
    for (size_t i = 0; i < logits_dim; ++i) {
        const float e = expf(logits[i] - max_logit);
        result.probabilities[i] = e;
        sum += e;
    }
    if (sum <= 0.0f) {
        Serial.println("infer_error softmax_sum");
        return false;
    }
    float best = -1.0f;
    int best_idx = -1;
    for (size_t i = 0; i < logits_dim; ++i) {
        result.probabilities[i] /= sum;
        if (result.probabilities[i] > best) {
            best = result.probabilities[i];
            best_idx = (int)i;
        }
    }
    result.class_index = best_idx;
    return true;
}

bool predict(const FeatureVector &features,
             PredictResult &result,
             float ordered_features_out[model_weights::kInputDim],
             bool *imputed) {
    if (!model_weights::kIsTrained) {
        Serial.println("infer_error untrained_model");
        return false;
    }

    float local_ordered[model_weights::kInputDim];
    float *ordered_ptr = ordered_features_out ? ordered_features_out : local_ordered;
    float normalized[model_weights::kInputDim];
    bool imputed_local = false;

    if (!preprocess(features, ordered_ptr, normalized, imputed_local)) {
        return false;
    }

    float logits[kMaxLayerWidth];
    size_t logits_dim = 0;
    if (!forward(normalized, logits, logits_dim)) {
        return false;
    }
    if (!postprocess(logits, logits_dim, result)) {
        return false;
    }

    if (imputed) {
        *imputed = imputed_local;
    }
    return true;
}

void printModelBanner() {
    Serial.print("[model] trained=");
    Serial.print(model_weights::kIsTrained ? "true" : "false");
    Serial.print(" input_dim=");
    Serial.print(model_weights::kInputDim);
    Serial.print(" classes=");
    Serial.print(model_weights::kNumClasses);
    Serial.print(" type=");
    Serial.println(outputTypeString());

    Serial.print("[model] class_names:");
    for (size_t i = 0; i < model_weights::kNumClasses; ++i) {
        Serial.print(' ');
        Serial.print(model_weights::kClassNames[i]);
    }
    Serial.println();

    Serial.print("[model] feature_names:");
    for (size_t i = 0; i < model_weights::kInputDim; ++i) {
        Serial.print(' ');
        Serial.print(model_weights::kFeatureNames[i]);
    }
    Serial.println();
}

} // namespace model_infer

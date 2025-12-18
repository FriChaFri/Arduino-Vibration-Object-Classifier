#!/usr/bin/env python3
"""
Export a trained MLP (produced by train_model.py) to a Teensy-friendly C header.

Typical use:
python3 scripts/export_model.py --modeldir models/latest --out include/model_weights.h
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np
try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback for minimal environments
    import pickle

    class _JoblibCompat:
        @staticmethod
        def dump(obj, path):
            with Path(path).open("wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path):
            with Path(path).open("rb") as f:
                return pickle.load(f)

    joblib = _JoblibCompat()  # type: ignore

SANITY_SAMPLE_LIMIT = 10
SANITY_ABS_TOL = 5e-5


def format_float(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    return f"{val:.8f}f"


def format_array(
    values: Iterable,
    indent: str = "    ",
    per_line: int = 8,
    formatter: Callable[[object], str] = format_float,
) -> str:
    vals = list(values)
    if not vals:
        return ""
    lines: List[str] = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append(indent + ", ".join(formatter(v) for v in chunk))
    return ",\n".join(lines)


def resolve_feature_paths(entries: Sequence[str], model_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.exists():
            resolved.append(path)
            continue
        candidate = model_dir / entry
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def load_sample_matrix(csv_paths: Sequence[Path], feature_names: Sequence[str], limit: int) -> np.ndarray:
    rows: List[List[float]] = []
    for csv_path in csv_paths:
        if len(rows) >= limit:
            break
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample: List[float] = []
                for feat in feature_names:
                    val = row.get(feat)
                    if val is None or val == "":
                        sample.append(np.nan)
                    else:
                        try:
                            sample.append(float(val))
                        except ValueError:
                            sample.append(np.nan)
                rows.append(sample)
                if len(rows) >= limit:
                    break
    return np.array(rows, dtype=np.float32)


def reshape_layers_for_eval(layers: List[dict]) -> List[tuple[np.ndarray, np.ndarray]]:
    prepared: List[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        weights = np.asarray(layer["weights"], dtype=np.float32).reshape(layer["output_dim"], layer["input_dim"])
        bias = np.asarray(layer["bias"], dtype=np.float32)
        prepared.append((weights, bias))
    return prepared


def manual_predict_proba(
    raw_samples: np.ndarray,
    imputer_stats: Sequence[float],
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
    prepared_layers: List[tuple[np.ndarray, np.ndarray]],
    output_type: str,
    num_classes: int,
    positive_index: int,
) -> np.ndarray:
    stats = np.asarray(imputer_stats, dtype=np.float64)
    mean = np.asarray(scaler_mean, dtype=np.float64)
    scale = np.asarray(scaler_scale, dtype=np.float64)
    safe_scale = np.where(scale == 0, 1.0, scale)

    samples = np.asarray(raw_samples, dtype=np.float64)
    if samples.size == 0:
        return np.empty((0, num_classes), dtype=np.float64)
    samples = np.where(np.isnan(samples), stats, samples)
    samples = (samples - mean) / safe_scale

    outputs: List[np.ndarray] = []
    for vec in samples:
        activation = vec
        for idx, (weights, bias) in enumerate(prepared_layers):
            activation = weights @ activation + bias
            if idx < len(prepared_layers) - 1:
                activation = np.maximum(activation, 0.0)
        if output_type == "kBinaryLogit":
            if num_classes != 2:
                raise ValueError("Binary logit output requires exactly two classes.")
            logit = float(activation[0])
            prob_pos = 1.0 / (1.0 + np.exp(-logit))
            probs = np.zeros(num_classes, dtype=np.float64)
            probs[positive_index] = prob_pos
            other_index = 1 - positive_index
            probs[other_index] = 1.0 - prob_pos
        else:
            logits = activation.astype(np.float64)
            logits -= np.max(logits)
            exp_vals = np.exp(logits)
            probs = exp_vals / np.sum(exp_vals)
        outputs.append(probs)
    return np.vstack(outputs)


def load_artifacts(model_dir: Path) -> tuple:
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "training_metadata.json"
    label_path = model_dir / "label_encoder.joblib"

    if not model_path.exists():
        raise SystemExit(f"{model_path} not found. Run train_model.py first.")
    if not metadata_path.exists():
        raise SystemExit(f"{metadata_path} not found. Run train_model.py first.")

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())

    label_classes = metadata.get("label_classes")
    if not label_classes and label_path.exists():
        encoder = joblib.load(label_path)
        label_classes = [str(c) for c in getattr(encoder, "classes_", [])]
    if not label_classes:
        raise SystemExit("Label classes are missing in metadata and encoder.")

    feature_names = metadata.get("feature_names")
    if not feature_names:
        raise SystemExit("Feature names are missing in metadata (training_metadata.json).")

    try:
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
    except KeyError:
        raise SystemExit("Expected pipeline with steps: imputer -> scaler -> clf (MLPClassifier).")

    clf_classes = [str(c) for c in getattr(clf, "classes_", [])]
    if clf_classes and clf_classes != label_classes:
        raise SystemExit(
            "Class ordering mismatch between metadata and classifier. Re-run training/export to refresh artifacts."
        )

    return pipeline, metadata, feature_names, label_classes, imputer, scaler, clf, clf_classes


def build_layer_params(clf) -> List[dict]:
    layers: List[dict] = []
    for idx, (weights, bias) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weight_matrix = np.asarray(weights, dtype=np.float32)
        bias_vec = np.asarray(bias, dtype=np.float32)
        in_dim, out_dim = weight_matrix.shape
        # Transpose to row-major [out][in] for straightforward C inference.
        flattened = weight_matrix.T.reshape(-1)
        layers.append(
            {
                "index": idx,
                "input_dim": int(in_dim),
                "output_dim": int(out_dim),
                "weights": flattened.tolist(),
                "bias": bias_vec.tolist(),
            }
        )
    return layers


def run_sanity_check(
    pipeline,
    sample_matrix: np.ndarray,
    imputer_stats: Sequence[float],
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
    layers: List[dict],
    output_type: str,
    num_classes: int,
    positive_class_index: int,
) -> tuple[int, float]:
    if sample_matrix.size == 0:
        raise SystemExit("Sanity check failed: no feature samples available to verify exported weights.")
    prepared_layers = reshape_layers_for_eval(layers)
    pipeline_probs = pipeline.predict_proba(sample_matrix)
    manual_probs = manual_predict_proba(
        sample_matrix,
        imputer_stats,
        scaler_mean,
        scaler_scale,
        prepared_layers,
        output_type,
        num_classes,
        positive_class_index,
    )
    diff = float(np.max(np.abs(pipeline_probs - manual_probs)))
    if not np.allclose(pipeline_probs, manual_probs, atol=SANITY_ABS_TOL, rtol=0.0):
        raise SystemExit(
            f"Sanity check failed: exported parameters mismatch sklearn predict_proba (max diff {diff:.2e})."
        )
    return sample_matrix.shape[0], diff


def generate_header(
    feature_names: List[str],
    label_classes: List[str],
    imputer_stats: List[float],
    scaler_mean: List[float],
    scaler_scale: List[float],
    layers: List[dict],
    output_type: str,
    positive_class_index: int,
) -> str:
    header_lines: List[str] = []
    header_lines.append("// Generated by scripts/export_model.py - do not edit by hand.")
    header_lines.append("#pragma once")
    header_lines.append("#include <cstddef>")
    header_lines.append("")
    header_lines.append("namespace model_weights {")
    header_lines.append(f"constexpr std::size_t kInputDim = {len(feature_names)};")
    header_lines.append(f"constexpr std::size_t kNumClasses = {len(label_classes)};")
    header_lines.append(f"constexpr std::size_t kNumLayers = {len(layers)};")
    header_lines.append("constexpr bool kIsTrained = true;")
    header_lines.append("")
    header_lines.append("// Output interpretation:")
    header_lines.append("// - kBinaryLogit: final neuron logistic → P(class[kLogitPositiveClass]), other class = 1 - P.")
    header_lines.append("// - kMultiClass: apply softmax to the final layer outputs to align with kClassNames order.")
    header_lines.append("enum class OutputType { kBinaryLogit, kMultiClass };")
    header_lines.append(f"constexpr OutputType kOutputType = OutputType::{output_type};")
    header_lines.append(
        f"constexpr std::size_t kLogitPositiveClass = {positive_class_index};  // valid only when kOutputType == OutputType::kBinaryLogit"
    )
    header_lines.append("")
    header_lines.append("static const char* const kClassNames[kNumClasses] = {")
    header_lines.append(format_array([f'"{c}"' for c in label_classes], indent="    ", per_line=4, formatter=str))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("static const char* const kFeatureNames[kInputDim] = {")
    header_lines.append(format_array([f'"{f}"' for f in feature_names], indent="    ", per_line=4, formatter=str))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("static const float kImputerMedian[kInputDim] = {")
    header_lines.append(format_array(imputer_stats))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("static const float kScalerMean[kInputDim] = {")
    header_lines.append(format_array(scaler_mean))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("static const float kScalerScale[kInputDim] = {")
    header_lines.append(format_array(scaler_scale))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("struct Layer {")
    header_lines.append("    std::size_t input_dim;")
    header_lines.append("    std::size_t output_dim;")
    header_lines.append("    const float* weights;  // row-major [output][input]")
    header_lines.append("    const float* biases;   // length == output_dim")
    header_lines.append("};")
    header_lines.append("")

    for layer in layers:
        idx = layer["index"]
        header_lines.append(f"static const float kLayer{idx}Weights[] = {{")
        header_lines.append(format_array(layer["weights"]))
        header_lines.append("};")
        header_lines.append(f"static const float kLayer{idx}Bias[] = {{")
        header_lines.append(format_array(layer["bias"]))
        header_lines.append("};")
        header_lines.append("")

    header_lines.append("static const Layer kLayers[kNumLayers] = {")
    layer_entries = []
    for layer in layers:
        idx = layer["index"]
        entry = (
            f"    {{ {layer['input_dim']}, {layer['output_dim']}, "
            f"kLayer{idx}Weights, kLayer{idx}Bias }}"
        )
        layer_entries.append(entry)
    header_lines.append(",\n".join(layer_entries))
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("}  // namespace model_weights")
    header_lines.append("")
    return "\n".join(header_lines)


def export_header(model_dir: Path, out_path: Path) -> None:
    pipeline, metadata, feature_names, label_classes, imputer, scaler, clf, clf_classes = load_artifacts(model_dir)

    imputer_stats = imputer.statistics_.tolist()
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()

    if not (
        len(feature_names)
        == len(imputer_stats)
        == len(scaler_mean)
        == len(scaler_scale)
    ):
        raise SystemExit("Feature count mismatch between metadata and preprocessing steps.")

    layers = build_layer_params(clf)
    if not layers:
        raise SystemExit("No layers found in classifier.")

    output_neurons = layers[-1]["output_dim"]
    num_classes = len(label_classes)
    if num_classes == 2 and output_neurons == 1:
        output_type = "kBinaryLogit"
        if len(clf_classes) != 2:
            raise SystemExit("Binary classifier missing class ordering information from clf.classes_.")
        positive_label = clf_classes[1]
        try:
            positive_class_index = label_classes.index(positive_label)
        except ValueError as exc:
            raise SystemExit("Positive class from classifier is missing in exported label list.") from exc
    elif output_neurons == num_classes:
        output_type = "kMultiClass"
        positive_class_index = 0
    else:
        raise SystemExit(
            "Unsupported classifier shape: final layer output does not align with number of classes. "
            "Ensure binary classifiers export a single logit and multi-class exports one logit per class."
        )

    feature_files_meta = metadata.get("feature_files") or []
    feature_paths = resolve_feature_paths(feature_files_meta, model_dir)
    if not feature_paths:
        glob_pattern = metadata.get("features_glob")
        if glob_pattern:
            feature_paths = list(Path(".").glob(glob_pattern))
    if not feature_paths:
        raise SystemExit(
            "Sanity check failed: unable to locate any feature CSVs referenced by training metadata."
        )

    sample_matrix = load_sample_matrix(feature_paths, feature_names, SANITY_SAMPLE_LIMIT)
    checked_samples, max_diff = run_sanity_check(
        pipeline,
        sample_matrix,
        imputer_stats,
        scaler_mean,
        scaler_scale,
        layers,
        output_type,
        num_classes,
        positive_class_index,
    )

    header = generate_header(
        feature_names,
        label_classes,
        imputer_stats,
        scaler_mean,
        scaler_scale,
        layers,
        output_type,
        positive_class_index,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header)

    print(f"Wrote header to {out_path}")
    print(f"Classes: {label_classes}")
    print(f"Feature order ({len(feature_names)}): {', '.join(feature_names)}")
    if output_type == "kBinaryLogit":
        print(f"Output mode: binary logit (probability corresponds to class '{label_classes[positive_class_index]}').")
    else:
        print("Output mode: multi-class softmax (probabilities follow kClassNames order).")
    print(f"Sanity check: {checked_samples} samples, max |Δ|={max_diff:.2e}")
    if "val_accuracy" in metadata:
        print(f"Validation accuracy recorded during training: {metadata['val_accuracy']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained MLP weights to a C header for Teensy inference.")
    parser.add_argument(
        "--modeldir",
        type=Path,
        default=Path("models") / "latest",
        help="Directory containing model.joblib and training_metadata.json (default: models/latest).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("include") / "model_weights.h",
        help="Path to write the generated header (default: include/model_weights.h).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_header(args.modeldir, args.out)

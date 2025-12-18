#!/usr/bin/env python3
"""
Export a trained scikit-learn MLP pipeline to a Teensy-friendly C header.

This script expects artifacts produced by scripts/train_model.py:
    - model.joblib
    - training_metadata.json

It exports:
    - feature order
    - class order (must match clf.classes_)
    - imputer medians
    - scaler mean/scale
    - MLP weights/biases (row-major [out][in])
    - output interpretation flags (single-logit case vs multi-logit case)
    - export-time sanity check to catch transpose/order bugs

Author:
    ChatGPT5: EAGER project context
    Reviewed and approved by Caleb Hottes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import joblib  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: joblib (installed with scikit-learn).") from exc


@dataclass(frozen=True)
class ExportConfig:
    modeldir: Path
    out: Path
    sanity_samples: int
    sanity_atol: float
    sanity_disable: bool


def format_float(val: float) -> str:
    if not np.isfinite(val):
        val = 0.0
    return f"{float(val):.8f}f"


def format_array(
    values: Iterable,
    *,
    indent: str = "    ",
    per_line: int = 8,
    formatter: Callable[[object], str] = format_float,
) -> str:
    vals = list(values)
    if not vals:
        return indent + "/* empty */"
    lines: List[str] = []
    for i in range(0, len(vals), per_line):
        chunk = vals[i : i + per_line]
        lines.append(indent + ", ".join(formatter(v) for v in chunk))
    return ",\n".join(lines)


def load_artifacts(modeldir: Path) -> Tuple[object, dict]:
    model_path = modeldir / "model.joblib"
    meta_path = modeldir / "training_metadata.json"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}. Run scripts/train_model.py first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}. Run scripts/train_model.py first.")
    pipeline = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return pipeline, metadata


def get_pipeline_steps(pipeline: object) -> Tuple[object, object, object]:
    try:
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Expected a Pipeline with steps: imputer, scaler, clf.") from exc
    return imputer, scaler, clf


def resolve_feature_paths(metadata: dict, modeldir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in metadata.get("feature_files", []) or []:
        candidate = Path(p)
        if candidate.exists():
            paths.append(candidate)
        else:
            rel = modeldir / p
            if rel.exists():
                paths.append(rel)
    if not paths:
        glob_pat = metadata.get("features_glob")
        if glob_pat:
            paths = sorted(Path(".").glob(glob_pat))
    return paths


def load_sample_matrix(
    csv_paths: Sequence[Path],
    feature_names: Sequence[str],
    limit: int,
) -> np.ndarray:
    rows: List[List[float]] = []
    for csv_path in csv_paths:
        if len(rows) >= limit:
            break
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample: List[float] = []
                for feat in feature_names:
                    s = row.get(feat)
                    if s is None or s == "":
                        sample.append(np.nan)
                    else:
                        try:
                            sample.append(float(s))
                        except ValueError:
                            sample.append(np.nan)
                rows.append(sample)
                if len(rows) >= limit:
                    break
    return np.asarray(rows, dtype=np.float64)


@dataclass
class Layer:
    in_dim: int
    out_dim: int
    w_out_in: np.ndarray  # [out, in]
    b: np.ndarray         # [out]


def extract_layers(clf: object) -> List[Layer]:
    coefs = getattr(clf, "coefs_", None)
    intercepts = getattr(clf, "intercepts_", None)
    if coefs is None or intercepts is None:
        raise SystemExit("Classifier does not expose coefs_ / intercepts_.")
    layers: List[Layer] = []
    for W_in_out, b_out in zip(coefs, intercepts):
        W = np.asarray(W_in_out, dtype=np.float64)  # [in, out]
        b = np.asarray(b_out, dtype=np.float64)     # [out]
        in_dim, out_dim = W.shape
        # Export as row-major [out][in] for straightforward C inference.
        layers.append(Layer(in_dim=in_dim, out_dim=out_dim, w_out_in=W.T.copy(), b=b.copy()))
    if not layers:
        raise SystemExit("No layers found in classifier.")
    return layers


def determine_output_mode(classes: List[str], layers: List[Layer]) -> Tuple[str, int]:
    """
    Returns: (output_type, positive_class_index)

    output_type:
      - "kBinaryLogit" if 2 classes and final layer has 1 neuron
      - "kMultiClass" if final layer has n_classes neurons

    positive_class_index:
      - only meaningful for kBinaryLogit; indicates which class index (in kClassNames)
        corresponds to sigmoid(logit).
    """
    n_classes = len(classes)
    out_neurons = layers[-1].out_dim
    if n_classes == 2 and out_neurons == 1:
        # predict_proba columns align with classes_ order; for a single-logit head,
        # sigmoid(logit) corresponds to the second class in that order.
        return "kBinaryLogit", 1
    if out_neurons == n_classes:
        return "kMultiClass", 0
    raise SystemExit(
        f"Unsupported output shape: {n_classes} class(es), final layer has {out_neurons} neuron(s)."
    )


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def manual_predict_proba(
    X_raw: np.ndarray,
    *,
    imputer_median: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    layers: List[Layer],
    output_type: str,
    positive_index: int,
) -> np.ndarray:
    X = X_raw.copy()
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        X[nan_mask] = np.take(imputer_median, np.where(nan_mask)[1])

    safe_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    X = (X - scaler_mean) / safe_scale

    A = X
    for li, layer in enumerate(layers):
        Z = A @ layer.w_out_in.T + layer.b
        A = relu(Z) if li < len(layers) - 1 else Z

    if output_type == "kBinaryLogit":
        p1 = sigmoid(A[:, 0])
        p0 = 1.0 - p1
        probs = np.stack([p0, p1], axis=1)
        if positive_index != 1:
            probs = probs[:, [positive_index ^ 1, positive_index]]
        return probs

    return softmax(A)


def run_sanity_check(
    *,
    pipeline: object,
    X_sample: np.ndarray,
    imputer_median: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    layers: List[Layer],
    output_type: str,
    positive_index: int,
    atol: float,
) -> Tuple[int, float]:
    if X_sample.shape[0] == 0:
        raise SystemExit("Sanity check failed: no samples loaded to validate export.")

    pipeline_probs = pipeline.predict_proba(X_sample)
    manual_probs = manual_predict_proba(
        X_sample,
        imputer_median=imputer_median,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        layers=layers,
        output_type=output_type,
        positive_index=positive_index,
    )

    if pipeline_probs.shape != manual_probs.shape:
        raise SystemExit(f"Sanity check failed: shape mismatch {pipeline_probs.shape} vs {manual_probs.shape}")

    diff = float(np.max(np.abs(pipeline_probs - manual_probs)))
    if not np.allclose(pipeline_probs, manual_probs, atol=atol, rtol=0.0):
        raise SystemExit(f"Sanity check failed: max |Δ|={diff:.2e} exceeds atol={atol:g}")
    return int(X_sample.shape[0]), diff


def generate_header(
    *,
    feature_names: List[str],
    class_names: List[str],
    imputer_median: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    layers: List[Layer],
    output_type: str,
    positive_index: int,
) -> str:
    # Export float32 constants to match embedded inference.
    imputer_f = imputer_median.astype(np.float32)
    mean_f = scaler_mean.astype(np.float32)
    scale_f = scaler_scale.astype(np.float32)

    hdr: List[str] = []
    hdr.append("// Generated by scripts/export_model.py - do not edit by hand.")
    hdr.append("#pragma once")
    hdr.append("#include <cstddef>")
    hdr.append("")
    hdr.append("namespace model_weights {")
    hdr.append("constexpr bool kIsTrained = true;")
    hdr.append(f"constexpr std::size_t kInputDim = {len(feature_names)};")
    hdr.append(f"constexpr std::size_t kNumClasses = {len(class_names)};")
    hdr.append(f"constexpr std::size_t kNumLayers = {len(layers)};")
    hdr.append("")
    hdr.append("// Output interpretation:")
    hdr.append("// - kBinaryLogit: final neuron logistic -> P(class[kLogitPositiveClass]); other class = 1 - P.")
    hdr.append("// - kMultiClass: apply softmax to final layer logits; probabilities align with kClassNames order.")
    hdr.append("enum class OutputType { kBinaryLogit, kMultiClass };")
    hdr.append(f"constexpr OutputType kOutputType = OutputType::{output_type};")
    hdr.append(f"constexpr std::size_t kLogitPositiveClass = {positive_index};")
    hdr.append("")
    hdr.append("static const char* const kClassNames[kNumClasses] = {")
    hdr.append(format_array([f'"{c}"' for c in class_names], indent="    ", per_line=4, formatter=str))
    hdr.append("};")
    hdr.append("")
    hdr.append("static const char* const kFeatureNames[kInputDim] = {")
    hdr.append(format_array([f'"{f}"' for f in feature_names], indent="    ", per_line=4, formatter=str))
    hdr.append("};")
    hdr.append("")
    hdr.append("static const float kImputerMedian[kInputDim] = {")
    hdr.append(format_array(imputer_f.tolist()))
    hdr.append("};")
    hdr.append("")
    hdr.append("static const float kScalerMean[kInputDim] = {")
    hdr.append(format_array(mean_f.tolist()))
    hdr.append("};")
    hdr.append("")
    hdr.append("static const float kScalerScale[kInputDim] = {")
    hdr.append(format_array(scale_f.tolist()))
    hdr.append("};")
    hdr.append("")
    hdr.append("struct Layer {")
    hdr.append("    std::size_t input_dim;")
    hdr.append("    std::size_t output_dim;")
    hdr.append("    const float* weights; // row-major [output][input]")
    hdr.append("    const float* biases;  // length == output_dim")
    hdr.append("};")
    hdr.append("")

    for li, layer in enumerate(layers):
        W = layer.w_out_in.astype(np.float32)  # [out, in]
        b = layer.b.astype(np.float32)
        hdr.append(f"static const float kLayer{li}Weights[] = {{")
        hdr.append(format_array(W.reshape(-1).tolist()))
        hdr.append("};")
        hdr.append(f"static const float kLayer{li}Bias[] = {{")
        hdr.append(format_array(b.tolist()))
        hdr.append("};")
        hdr.append("")

    hdr.append("static const Layer kLayers[kNumLayers] = {")
    hdr.append(
        ",\n".join(
            f"    {{ {layer.in_dim}, {layer.out_dim}, kLayer{li}Weights, kLayer{li}Bias }}"
            for li, layer in enumerate(layers)
        )
    )
    hdr.append("};")
    hdr.append("")
    hdr.append("} // namespace model_weights")
    hdr.append("")
    return "\n".join(hdr)


def export_model(cfg: ExportConfig) -> None:
    pipeline, metadata = load_artifacts(cfg.modeldir)
    imputer, scaler, clf = get_pipeline_steps(pipeline)

    feature_names = metadata.get("feature_names")
    if not feature_names:
        raise SystemExit("training_metadata.json missing feature_names.")
    feature_names = [str(f) for f in feature_names]

    classes = [str(c) for c in getattr(clf, "classes_", [])]
    if not classes:
        raise SystemExit("Classifier has no classes_. Train the model first.")

    meta_classes = metadata.get("classes")
    if meta_classes and [str(c) for c in meta_classes] != classes:
        raise SystemExit("Class ordering mismatch between metadata and classifier. Re-train/export.")

    imputer_median = np.asarray(getattr(imputer, "statistics_", None), dtype=np.float64)
    scaler_mean = np.asarray(getattr(scaler, "mean_", None), dtype=np.float64)
    scaler_scale = np.asarray(getattr(scaler, "scale_", None), dtype=np.float64)

    if not (len(imputer_median) == len(scaler_mean) == len(scaler_scale) == len(feature_names)):
        raise SystemExit("Preprocessing arrays do not match feature_names length. Re-train.")

    layers = extract_layers(clf)
    output_type, positive_index = determine_output_mode(classes, layers)

    if not cfg.sanity_disable:
        feature_paths = resolve_feature_paths(metadata, cfg.modeldir)
        if not feature_paths:
            raise SystemExit("Sanity check failed: could not locate feature CSVs.")
        X_sample = load_sample_matrix(feature_paths, feature_names, cfg.sanity_samples)
        checked_n, max_diff = run_sanity_check(
            pipeline=pipeline,
            X_sample=X_sample,
            imputer_median=imputer_median,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            layers=layers,
            output_type=output_type,
            positive_index=positive_index,
            atol=cfg.sanity_atol,
        )
    else:
        checked_n, max_diff = 0, math.nan

    header = generate_header(
        feature_names=feature_names,
        class_names=classes,
        imputer_median=imputer_median,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        layers=layers,
        output_type=output_type,
        positive_index=positive_index,
    )

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(header)

    print(f"Wrote header: {cfg.out}")
    print(f"Classes (kClassNames): {classes}")
    print(f"Feature order ({len(feature_names)}): {', '.join(feature_names)}")
    if output_type == "kBinaryLogit":
        print(f"Output mode: single-logit head; sigmoid(logit) is P('{classes[positive_index]}').")
    else:
        print("Output mode: multi-logit head; apply softmax; probs align with kClassNames.")
    if not cfg.sanity_disable:
        print(f"Sanity check: {checked_n} sample(s), max |Δ|={max_diff:.2e}, atol={cfg.sanity_atol:g}")


def parse_args() -> ExportConfig:
    p = argparse.ArgumentParser(
        description="Export trained MLP weights to a C header for Teensy inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/export_model.py --modeldir models/latest --out include/model_weights.h\n"
            "  (add --sanity-disable if you need to skip the forward-pass sanity check)"
        ),
    )
    p.add_argument("--modeldir", type=Path, default=Path("models") / "latest", help="Model directory.")
    p.add_argument("--out", type=Path, default=Path("include") / "model_weights.h", help="Output header path.")
    p.add_argument("--sanity-samples", type=int, default=10, help="Number of CSV rows to sanity-check.")
    p.add_argument("--sanity-atol", type=float, default=1e-4, help="Absolute tolerance for sanity check.")
    p.add_argument("--sanity-disable", action="store_true", help="Disable export sanity check.")
    args = p.parse_args()
    if args.sanity_samples <= 0:
        raise SystemExit("--sanity-samples must be > 0")
    if args.sanity_atol <= 0:
        raise SystemExit("--sanity-atol must be > 0")
    return ExportConfig(
        modeldir=args.modeldir,
        out=args.out,
        sanity_samples=args.sanity_samples,
        sanity_atol=args.sanity_atol,
        sanity_disable=bool(args.sanity_disable),
    )


def main() -> None:
    export_model(parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a tiny MLP on collected impact feature CSV files and save artifacts for Teensy export.

Typical use:
python3 scripts/train_model.py --features-glob "data/run_*/features.csv" --outdir models/latest
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that are metadata, identifiers, or config (not used as ML features)
NON_FEATURE_COLUMNS = {
    "impact_id",
    "label",
    "timestamp_us",
    "odr_hz",
    "fs_g",
    "pretrigger_recorded",
    "stage1_count",
    "stage2_count",
    "stage2_decimation",
    "filename",
    "file",
    "npz_file",
    "npz_path",
    "run_dir",
    "config",
}

# Case-insensitive substrings to map raw labels into categories
CATEGORY_RULES: Dict[str, Tuple[str, ...]] = {
    "erasor": ("erasor",),
    "screw": ("screw",),
}

DEFAULT_FEATURE_GLOB = "data/run_*/features.csv"


@dataclass
class LoadedDataset:
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    categories: List[str]
    counts: Dict[str, int]
    dropped_rows: int
    source_files: List[str]


def map_label_to_category(label: str | None) -> str | None:
    """Apply simple substring rules to map the raw label to a category."""
    if not label:
        return None
    lower = label.lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(token in lower for token in keywords):
            return category
    return None


def find_feature_files(pattern: str) -> List[Path]:
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise SystemExit(f"No feature CSV files matched pattern: {pattern}")
    return paths


def load_dataset(csv_files: Sequence[Path]) -> LoadedDataset:
    feature_order: List[str] = []
    rows: List[Dict[str, float]] = []
    labels: List[str] = []
    dropped_rows = 0

    for path in csv_files:
        with path.open() as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            for row in reader:
                category = map_label_to_category(row.get("label"))
                if category is None:
                    dropped_rows += 1
                    continue

                feats: Dict[str, float] = {}
                for key in reader.fieldnames:
                    if key is None or key in NON_FEATURE_COLUMNS:
                        continue
                    val = row.get(key, "")
                    if val is None or val == "":
                        continue
                    try:
                        feat_val = float(val)
                    except (TypeError, ValueError):
                        continue
                    feats[key] = feat_val
                    if key not in feature_order:
                        feature_order.append(key)

                if not feats:
                    continue

                rows.append(feats)
                labels.append(category)

    if not rows:
        raise SystemExit("No usable rows were found after filtering categories and features.")

    feature_names = feature_order
    feature_matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float32)
    for idx, feats in enumerate(rows):
        for col_idx, name in enumerate(feature_names):
            if name in feats:
                feature_matrix[idx, col_idx] = np.float32(feats[name])

    counts = Counter(labels)
    categories = sorted(counts.keys())

    return LoadedDataset(
        features=feature_matrix,
        labels=np.array(labels),
        feature_names=feature_names,
        categories=categories,
        counts=dict(counts),
        dropped_rows=dropped_rows,
        source_files=[str(p) for p in csv_files],
    )


def build_model(hidden_sizes: Tuple[int, ...], random_seed: int, max_iter: int) -> Pipeline:
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_seed,
        learning_rate_init=1e-3,
        verbose=False,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def train(args: argparse.Namespace) -> None:
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val-ratio must be between 0 and 1.")
    if any(h <= 0 for h in args.hidden_sizes):
        raise SystemExit("Hidden layer sizes must be positive integers.")

    feature_files = find_feature_files(args.features_glob)
    dataset = load_dataset(feature_files)

    if len(dataset.categories) < 2:
        raise SystemExit("Need at least two categories after grouping to train a classifier.")

    print(f"Loaded {len(dataset.labels)} samples from {len(feature_files)} files.")
    print(f"Dropped rows (labels outside current categories): {dataset.dropped_rows}")
    print("Per-category counts:")
    for cat, cnt in dataset.counts.items():
        print(f"  {cat}: {cnt}")
    print(f"Using features ({len(dataset.feature_names)}): {', '.join(dataset.feature_names)}")

    if dataset.features.shape[0] < 2:
        raise SystemExit("Need at least two samples to run train/validation split.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(dataset.labels)
    class_names = list(label_encoder.classes_)

    model = build_model(tuple(args.hidden_sizes), args.random_seed, args.max_iter)

    val_size = max(1, int(round(len(dataset.labels) * args.val_ratio)))
    if val_size >= len(dataset.labels):
        val_size = len(dataset.labels) - 1
    print(f"Validation holdout: {val_size} samples (~{val_size / len(dataset.labels):.1%})")

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.features,
            y_encoded,
            test_size=val_size,
            random_state=args.random_seed,
            stratify=y_encoded,
        )
    except ValueError:
        # Fallback without stratification for very small datasets
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.features,
            y_encoded,
            test_size=val_size,
            random_state=args.random_seed,
            stratify=None,
        )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nValidation accuracy: {val_acc:.3f}")
    cm = confusion_matrix(y_val, val_pred, labels=list(range(len(class_names))))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_val, val_pred, labels=list(range(len(class_names))), target_names=class_names))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.joblib"
    label_path = outdir / "label_encoder.joblib"
    metadata_path = outdir / "training_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_path)

    metadata = {
        "created": datetime.now().isoformat(),
        "features_glob": args.features_glob,
        "feature_files": dataset.source_files,
        "feature_names": dataset.feature_names,
        "label_classes": class_names,
        "category_rules": {k: list(v) for k, v in CATEGORY_RULES.items()},
        "hidden_layer_sizes": list(args.hidden_sizes),
        "val_ratio": args.val_ratio,
        "val_size": val_size,
        "val_accuracy": val_acc,
        "train_samples": int(y_train.shape[0]),
        "val_samples": int(y_val.shape[0]),
        "counts_per_category": dataset.counts,
        "dropped_rows": dataset.dropped_rows,
        "model_params": {
            "activation": "relu",
            "solver": "adam",
            "max_iter": args.max_iter,
            "learning_rate_init": 1e-3,
            "random_seed": args.random_seed,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model artifacts to {outdir}")
    print(f"  model:          {model_path}")
    print(f"  label encoder:  {label_path}")
    print(f"  metadata:       {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small MLP on impact feature CSV files.")
    parser.add_argument(
        "--features-glob",
        default=DEFAULT_FEATURE_GLOB,
        help="Glob pattern for feature CSV files (default: data/run_*/features.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("models") / "latest",
        help="Directory to write model artifacts (default: models/latest)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio between 0 and 1 (default: 0.2).",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="Hidden layer sizes for the MLP (default: 16). Example: --hidden-sizes 24 12",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Max training iterations for the MLP (default: 300).")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for splitting and training (default: 42).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

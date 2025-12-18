#!/usr/bin/env python3
"""
Train a small MLP classifier on impact feature CSV files and save artifacts for embedded export.

Pipeline (scikit-learn):
    SimpleImputer(median) -> StandardScaler -> MLPClassifier

Label grouping:
    Raw CSV `label` strings are mapped into categories using substring rules
    (default: "eraser" and "screw"). Rows that do not match any category are dropped.

Artifacts written to --outdir:
    - model.joblib              (sklearn Pipeline)
    - training_metadata.json    (feature order, class order, sources, metrics, etc.)

Author:
    ChatGPT5: EAGER project context
    Reviewed and approved by Caleb Hottes.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import joblib  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: joblib (installed with scikit-learn).") from exc

try:
    import sklearn  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: scikit-learn. Install requirements.txt.") from exc

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Columns that are metadata/IDs/config and should not be used as ML features.
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

DEFAULT_FEATURES_GLOB = "data/run_*/features.csv"

# Default substring rules: category -> list of tokens (case-insensitive).
DEFAULT_CATEGORY_RULES: Dict[str, Tuple[str, ...]] = {
    "eraser": ("eraser",),
    "screw": ("screw",),
}


@dataclass(frozen=True)
class Dataset:
    """In-memory dataset built from many CSV files."""
    X: np.ndarray  # [n_samples, n_features]
    y: np.ndarray  # [n_samples], dtype=str
    feature_names: List[str]
    source_files: List[str]
    dropped_rows: int
    counts: Dict[str, int]


def _natural_key(s: str) -> Tuple:
    """Sort key that keeps band_2 before band_10."""
    import re

    return tuple(int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", s))


def parse_category_rules(spec: Optional[str]) -> Dict[str, Tuple[str, ...]]:
    """
    Parse a simple rule spec like:
        "eraser=eraser,eraser;screw=screw,wood_screw"
    Returns: dict category -> tuple(tokens)
    """
    if not spec:
        return dict(DEFAULT_CATEGORY_RULES)

    rules: Dict[str, Tuple[str, ...]] = {}
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Invalid --category-rules entry (missing '='): {part!r}")
        cat, toks = part.split("=", 1)
        cat = cat.strip()
        tokens = tuple(t.strip() for t in toks.split(",") if t.strip())
        if not cat or not tokens:
            raise SystemExit(f"Invalid --category-rules entry: {part!r}")
        rules[cat] = tokens
    return rules


def map_label_to_category(label: Optional[str], rules: Dict[str, Tuple[str, ...]]) -> Optional[str]:
    """Map a raw label string into a category via substring rules."""
    if not label:
        return None
    low = label.lower()
    for category, tokens in rules.items():
        if any(tok.lower() in low for tok in tokens):
            return category
    return None


def find_feature_csvs(glob_pattern: str) -> List[Path]:
    files = sorted(Path(".").glob(glob_pattern))
    if not files:
        raise SystemExit(f"No CSVs matched --features-glob {glob_pattern!r}")
    return files


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        for row in reader:
            yield row


def load_dataset(csv_files: Sequence[Path], rules: Dict[str, Tuple[str, ...]]) -> Dataset:
    """
    Load and merge multiple features.csv files.

    Strategy:
    - Discover feature names by finding numeric columns (excluding NON_FEATURE_COLUMNS)
      that successfully parse at least once across all files.
    - Build a dense matrix with NaN for missing entries, then let the imputer handle NaNs.
    """
    numeric_columns: Dict[str, int] = {}  # name -> count of successful parses
    kept_rows: List[Dict[str, float]] = []
    labels: List[str] = []
    dropped_rows = 0

    for csv_path in csv_files:
        for row in _iter_csv_rows(csv_path):
            category = map_label_to_category(row.get("label"), rules)
            if category is None:
                dropped_rows += 1
                continue

            feats: Dict[str, float] = {}
            for k, v in row.items():
                if not k or k in NON_FEATURE_COLUMNS:
                    continue
                if v is None or v == "":
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                feats[k] = fv
                numeric_columns[k] = numeric_columns.get(k, 0) + 1

            if not feats:
                dropped_rows += 1
                continue

            kept_rows.append(feats)
            labels.append(category)

    if not kept_rows:
        raise SystemExit("No usable rows found after label grouping and numeric feature filtering.")

    feature_names = sorted(numeric_columns.keys(), key=_natural_key)
    X = np.full((len(kept_rows), len(feature_names)), np.nan, dtype=np.float64)

    col_index = {name: i for i, name in enumerate(feature_names)}
    for i, feats in enumerate(kept_rows):
        for name, val in feats.items():
            j = col_index.get(name)
            if j is not None:
                X[i, j] = float(val)

    y = np.asarray(labels, dtype=str)
    counts = dict(Counter(labels))

    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        source_files=[str(p) for p in csv_files],
        dropped_rows=dropped_rows,
        counts=counts,
    )


def build_pipeline(hidden_sizes: Tuple[int, ...], seed: int, max_iter: int) -> Pipeline:
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=1e-3,
        max_iter=max_iter,
        shuffle=True,
        random_state=seed,
        tol=1e-4,
        n_iter_no_change=20,
        verbose=False,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_and_save(
    *,
    features_glob: str,
    outdir: Path,
    category_rules: Dict[str, Tuple[str, ...]],
    val_ratio: float,
    seed: int,
    hidden_sizes: Tuple[int, ...],
    max_iter: int,
) -> None:
    csv_files = find_feature_csvs(features_glob)
    ds = load_dataset(csv_files, category_rules)

    if len(set(ds.y.tolist())) < 2:
        raise SystemExit("Need at least two categories after grouping to train a classifier.")

    print(f"Loaded {ds.X.shape[0]} samples from {len(csv_files)} CSV(s).")
    print(f"Dropped rows (unmatched label or empty features): {ds.dropped_rows}")
    print("Per-category counts:", ", ".join(f"{k}={v}" for k, v in sorted(ds.counts.items())))
    print(f"Features ({len(ds.feature_names)}): {', '.join(ds.feature_names)}")

    if not (0.0 < val_ratio < 1.0):
        raise SystemExit("--val-ratio must be between 0 and 1.")

    n = ds.X.shape[0]
    val_size = max(1, int(round(n * val_ratio)))
    if val_size >= n:
        val_size = n - 1

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            ds.X,
            ds.y,
            test_size=val_size,
            random_state=seed,
            stratify=ds.y,
        )
    except ValueError:
        print("[warn] stratified split failed; falling back to unstratified split.")
        X_train, X_val, y_train, y_val = train_test_split(
            ds.X,
            ds.y,
            test_size=val_size,
            random_state=seed,
            stratify=None,
        )

    pipe = build_pipeline(hidden_sizes=hidden_sizes, seed=seed, max_iter=max_iter)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    classes = [str(c) for c in pipe.named_steps["clf"].classes_]

    cm = confusion_matrix(y_val, y_pred, labels=classes)
    report = classification_report(y_val, y_pred, labels=classes, target_names=classes, zero_division=0)

    print(f"Validation accuracy: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Classification report:")
    print(report)

    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.joblib"
    meta_path = outdir / "training_metadata.json"

    joblib.dump(pipe, model_path)

    metadata = {
        "created_utc": _utc_now_iso(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "features_glob": features_glob,
        "feature_files": ds.source_files,
        "feature_names": ds.feature_names,
        "category_rules": {k: list(v) for k, v in category_rules.items()},
        "classes": classes,  # authoritative predict_proba ordering
        "model_type": "sklearn.Pipeline(imputer->scaler->MLPClassifier)",
        "mlp_hidden_sizes": list(hidden_sizes),
        "mlp_max_iter": int(max_iter),
        "random_seed": int(seed),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "val_ratio": float(val_ratio),
        "val_accuracy": acc,
        "counts_per_category": ds.counts,
        "dropped_rows": int(ds.dropped_rows),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved artifacts to: {outdir}")
    print(f"  {model_path}")
    print(f"  {meta_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a small MLP on impact feature CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/train_model.py --features-glob \"data/run_*/features.csv\" "
            "--outdir models/latest --hidden-sizes 16 8 --val-ratio 0.25\n"
            "Category rule override example:\n"
            "  --category-rules \"eraser=eraser,erasor;screw=screw,wood_screw\""
        ),
    )
    p.add_argument("--features-glob", default=DEFAULT_FEATURES_GLOB, help="Glob for features.csv files.")
    p.add_argument("--outdir", type=Path, default=Path("models") / "latest", help="Output directory for artifacts.")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio in (0,1).")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="MLP hidden layer sizes, e.g. --hidden-sizes 16 or --hidden-sizes 24 12",
    )
    p.add_argument("--max-iter", type=int, default=300, help="MLP max iterations.")
    p.add_argument(
        "--category-rules",
        type=str,
        default=None,
        help='Override category rules. Example: "eraser=eraser,eraser;screw=screw,wood_screw"',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rules = parse_category_rules(args.category_rules)
    train_and_save(
        features_glob=args.features_glob,
        outdir=args.outdir,
        category_rules=rules,
        val_ratio=args.val_ratio,
        seed=args.random_seed,
        hidden_sizes=tuple(args.hidden_sizes),
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()

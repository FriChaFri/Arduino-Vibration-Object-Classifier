#!/usr/bin/env python3
"""
inspect_dataset.py

Quick summary of a run directory produced by collect_impacts.py:
 - counts per label
 - per-label mean/std for each feature
 - simple effect size between the first two labels (if available)
 - optional histograms per feature
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_features(csv_path: Path) -> tuple[list[dict], list[str]]:
    rows: List[Dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k == "label":
                    parsed[k] = v
                elif v is None or v == "":
                    parsed[k] = np.nan
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows, headers


def summarize(rows: List[Dict], headers: List[str]):
    if not rows:
        print("No rows to summarize.")
        return

    labels = sorted(set(r["label"] for r in rows))
    features = [h for h in headers if h not in ("impact_id", "label", "timestamp_us")]

    print("Counts per label:")
    for label in labels:
        cnt = sum(1 for r in rows if r["label"] == label)
        print(f"  {label}: {cnt}")

    print("\nPer-label mean / std:")
    for label in labels:
        subset = [r for r in rows if r["label"] == label]
        print(f"Label {label}:")
        for feat in features:
            vals = np.array([r[feat] for r in subset], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            print(f"  {feat:16s} mean={vals.mean():8.3f} std={vals.std():8.3f}")

    if len(labels) >= 2:
        l1, l2 = labels[:2]
        print(f"\nEffect size (d) between '{l1}' and '{l2}':")
        s1 = [r for r in rows if r["label"] == l1]
        s2 = [r for r in rows if r["label"] == l2]
        for feat in features:
            a = np.array([r[feat] for r in s1], dtype=np.float64)
            b = np.array([r[feat] for r in s2], dtype=np.float64)
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]
            if a.size < 2 or b.size < 2:
                continue
            pooled = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
            if pooled == 0:
                continue
            d = (a.mean() - b.mean()) / pooled
            print(f"  {feat:16s} d={d:6.3f}")


def plot_features(rows: List[Dict], headers: List[str], out_dir: Path):
    labels = sorted(set(r["label"] for r in rows))
    features = [h for h in headers if h not in ("impact_id", "label", "timestamp_us")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for feat in features:
        plt.figure(figsize=(6, 4))
        for label in labels:
            vals = np.array([r[feat] for r in rows if r["label"] == label], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            plt.hist(vals, bins=30, alpha=0.6, label=f"{label}")
        plt.title(feat)
        plt.xlabel(feat)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"{feat}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a collected impact dataset run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory (contains features.csv)")
    parser.add_argument("--plots", action="store_true", help="Save feature histograms to run_dir/plots")
    args = parser.parse_args()

    features_path = args.run_dir / "features.csv"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found")

    rows, headers = load_features(features_path)
    summarize(rows, headers)

    if args.plots:
        plot_dir = args.run_dir / "plots"
        plot_features(rows, headers, plot_dir)


if __name__ == "__main__":
    main()

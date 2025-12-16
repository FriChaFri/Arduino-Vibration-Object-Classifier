#!/usr/bin/env python3
"""
plot_waveforms.py

Load saved waveforms (CSV files produced by impact_logger.py) and generate plots.
Requires ``matplotlib`` (``pip install matplotlib``).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_waveform(csv_path: Path) -> List[float]:
    values: List[float] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except ValueError:
                continue
    return values


def plot_waveform(csv_path: Path, output_dir: Path | None = None, show: bool = False):
    values = load_waveform(csv_path)
    if not values:
        print(f"Skipping {csv_path} (no data)")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(csv_path.stem)
    plt.xlabel("Sample")
    plt.ylabel("Magnitude (mg)")
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{csv_path.stem}.png"
        plt.savefig(out_path)
        print(f"Saved plot -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot logged impact waveforms")
    parser.add_argument("input", default="data/raw_waveforms", nargs="?", help="Directory containing waveform CSV files")
    parser.add_argument("--output", help="Where to save plots (if omitted, plots are only shown)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of waveforms to plot")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise SystemExit(f"Input directory {input_dir} does not exist")

    files = sorted(input_dir.glob("impact_*.csv"))
    if args.limit is not None:
        files = files[: args.limit]

    output_dir = Path(args.output) if args.output else None

    for csv_path in files:
        plot_waveform(csv_path, output_dir=output_dir, show=args.show)

    if not files:
        print("No waveform CSV files found.")


if __name__ == "__main__":
    main()

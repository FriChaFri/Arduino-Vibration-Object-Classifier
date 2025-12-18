#!/usr/bin/env python3
"""
plot_waveforms.py

Load collected waveforms (NPZ files from collect_impacts.py or legacy CSV dumps)
and generate both per-wave and stacked overview plots.

example use case
python ./scripts/plot_waveforms.py --output assets/eraser_top/ --limit 30 ././data/run_20251216_172729/
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class WaveformRecord:
    path: Path
    time_s: np.ndarray
    magnitude_mg: np.ndarray
    label: str | None = None
    impact_id: int | None = None

    @property
    def legend_label(self) -> str:
        parts: List[str] = []
        if self.impact_id is not None:
            parts.append(f"id={self.impact_id}")
        if self.label is not None:
            parts.append(f"label={self.label}")
        if parts:
            return " ".join(parts)
        return self.path.stem


def extract_id_and_label(path: Path) -> tuple[int | None, str | None]:
    impact_id: int | None = None
    label: str | None = None
    tokens = path.stem.split("_")
    for idx, token in enumerate(tokens):
        if token == "impact" and idx + 1 < len(tokens):
            try:
                impact_id = int(tokens[idx + 1])
            except ValueError:
                pass
        if token == "label" and idx + 1 < len(tokens):
            label = tokens[idx + 1]
    return impact_id, label


def load_legacy_csv_waveform(csv_path: Path) -> WaveformRecord:
    values: List[float] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                values.append(float(row[1]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"{csv_path} did not contain numeric samples")
    magnitude = np.array(values, dtype=np.float32)
    time = np.arange(magnitude.size, dtype=np.float32)
    impact_id, label = extract_id_and_label(csv_path)
    return WaveformRecord(path=csv_path, time_s=time, magnitude_mg=magnitude, label=label, impact_id=impact_id)


def _load_metadata(npz_obj) -> dict:
    if "metadata" not in npz_obj.files:
        return {}
    meta_arr = npz_obj["metadata"]
    meta: object
    if isinstance(meta_arr, np.ndarray):
        if meta_arr.shape == ():
            meta = meta_arr.item()
        else:
            meta = meta_arr.tolist()
    else:
        meta = meta_arr
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    if isinstance(meta, dict):
        return meta
    return {}


def load_npz_waveform(npz_path: Path) -> WaveformRecord:
    with np.load(npz_path, allow_pickle=False) as npz:
        metadata = _load_metadata(npz)
        if "magnitude_mg" in npz.files:
            magnitude = np.array(npz["magnitude_mg"], dtype=np.float32)
        elif "accel_raw" in npz.files:
            accel = np.array(npz["accel_raw"], dtype=np.float32)
            mg_per_lsb = None
            cfg = metadata.get("config")
            if isinstance(cfg, dict):
                mg_per_lsb = cfg.get("mg_per_lsb")
            scale = float(mg_per_lsb) if mg_per_lsb is not None else 1.0
            magnitude = np.linalg.norm(accel * scale, axis=1)
        else:
            raise ValueError(f"{npz_path} missing magnitude data")

        if "time_s" in npz.files:
            time = np.array(npz["time_s"], dtype=np.float32)
        else:
            dt = None
            cfg = metadata.get("config")
            if isinstance(cfg, dict):
                odr = cfg.get("odr_hz")
                if odr:
                    dt = 1.0 / float(odr)
            if dt is not None:
                time = np.arange(magnitude.size, dtype=np.float32) * dt
            else:
                time = np.arange(magnitude.size, dtype=np.float32)

    impact_id = metadata.get("impact_id")
    if impact_id is None:
        impact_id, _ = extract_id_and_label(npz_path)
    label = metadata.get("label")
    if label is None:
        _, label = extract_id_and_label(npz_path)

    return WaveformRecord(path=npz_path, time_s=time, magnitude_mg=magnitude, label=label, impact_id=impact_id)


def load_waveform(path: Path) -> WaveformRecord:
    if path.suffix.lower() == ".npz":
        return load_npz_waveform(path)
    return load_legacy_csv_waveform(path)


def iter_waveform_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise SystemExit(f"Input path {input_path} does not exist")

    search_dirs: List[Path] = []
    wave_dir = input_path / "waves"
    if wave_dir.is_dir():
        search_dirs.append(wave_dir)
    if input_path.is_dir():
        search_dirs.append(input_path)

    seen: set[Path] = set()
    npz_files: List[Path] = []
    for directory in search_dirs:
        for path in sorted(directory.glob("impact_*.npz")):
            if path not in seen:
                npz_files.append(path)
                seen.add(path)
    if npz_files:
        return npz_files

    csv_files: List[Path] = []
    for directory in search_dirs:
        for path in sorted(directory.glob("impact_*.csv")):
            if path not in seen:
                csv_files.append(path)
                seen.add(path)
    return csv_files


def plot_individual_waveforms(
    waves: Iterable[WaveformRecord],
    output_dir: Path | None,
    show: bool,
    dpi: int,
) -> None:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for record in waves:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
        ax.plot(record.time_s, record.magnitude_mg)
        ax.set_title(record.legend_label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnitude (mg)")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        if output_dir:
            out_path = output_dir / f"{record.path.stem}.png"
            fig.savefig(out_path)
            print(f"Saved per-wave plot -> {out_path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_stacked_waveforms(
    waves: List[WaveformRecord],
    output_path: Path | None,
    show: bool,
    dpi: int,
) -> None:
    if not waves:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    cmap = plt.get_cmap("viridis")
    for idx, record in enumerate(waves):
        color = cmap(idx / max(len(waves) - 1, 1))
        ax.plot(record.time_s, record.magnitude_mg, color=color, alpha=0.75, label=record.legend_label)
    ax.set_title("Stacked waveforms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (mg)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved stacked plot -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot collected impact waveforms (NPZ or CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/plot_waveforms.py data/run_20240101_120000 --output assets/plots "
            "--stacked-output assets/plots/stacked.png --limit 32"
        ),
    )
    parser.add_argument(
        "input",
        default="data",
        nargs="?",
        help="Wave directory, run directory, or single waveform file.",
    )
    parser.add_argument("--output", type=Path, help="Directory to store per-wave PNGs.")
    parser.add_argument(
        "--stacked-output",
        type=Path,
        help="Path for the stacked plot (default: <output>/stacked_waveforms.png or cwd).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of waveforms to plot.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI (default: 120).")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = iter_waveform_files(input_path)
    if not files:
        raise SystemExit(f"No waveforms found under {input_path}")

    if args.limit is not None:
        files = files[: args.limit]

    waveforms: List[WaveformRecord] = []
    for path in files:
        try:
            waveforms.append(load_waveform(path))
        except (OSError, ValueError) as exc:
            print(f"[warn] skipping {path}: {exc}")

    if not waveforms:
        raise SystemExit("No valid waveforms to plot.")

    if args.output:
        plot_individual_waveforms(waveforms, args.output, show=args.show, dpi=args.dpi)
    else:
        plot_individual_waveforms(waveforms, None, show=args.show, dpi=args.dpi)

    stacked_path: Path | None = args.stacked_output
    if stacked_path is None and args.output:
        stacked_path = args.output / "stacked_waveforms.png"
    plot_stacked_waveforms(waveforms, stacked_path, show=args.show, dpi=args.dpi)


if __name__ == "__main__":
    main()

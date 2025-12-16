#!/usr/bin/env python3
"""
collect_impacts.py

Batch impact logger with labeling support. Listens for IMPACT packets, saves each
waveform to NPZ, and appends feature rows to features.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - host utility
    raise SystemExit("pyserial is required. pip install pyserial") from exc

from protocol import parse_impact_frame


def timestamp_dir(base: Path | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (base or Path("data")) / f"run_{stamp}"


def ensure_dirs(run_dir: Path) -> Path:
    waves = run_dir / "waves"
    waves.mkdir(parents=True, exist_ok=True)
    return waves


def feature_header(num_bands: int) -> List[str]:
    head = [
        "impact_id",
        "label",
        "timestamp_us",
        "odr_hz",
        "fs_g",
        "pretrigger_recorded",
        "stage1_count",
        "stage2_count",
        "peak_mag_mg",
        "peak_dev_mg",
        "rms_dev_mg",
        "decay_ms",
    ]
    head.extend([f"band_{i}" for i in range(num_bands)])
    return head


def append_feature_row(csv_path: Path, header: List[str], row: Dict):
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_time_axis(pkt: Dict) -> np.ndarray:
    cfg = pkt["config"]
    dt = 1.0 / float(cfg["odr_hz"])
    n1 = int(pkt["stage1_count"])
    n2 = int(pkt["stage2_count"])
    pre = int(cfg["pretrigger_recorded"])
    d2 = int(cfg["stage2_decimation"])
    total = n1 + n2
    times = np.zeros(total, dtype=np.float32)
    if n1:
        idx = np.arange(n1, dtype=np.int32)
        times[:n1] = (idx - pre) * dt
    if n2:
        j = np.arange(n2, dtype=np.int32)
        times[n1:] = (n1 + j * d2 - pre) * dt
    return times


def save_npz(wave_dir: Path, pkt: Dict, label: str):
    cfg = pkt["config"]
    time_s = build_time_axis(pkt)
    mg_per_lsb = float(cfg["mg_per_lsb"])
    magnitude_mg = np.linalg.norm(pkt["waveform"].astype(np.float32) * mg_per_lsb, axis=1)
    filename = wave_dir / f"impact_{pkt['impact_id']:06d}_label_{label}.npz"
    np.savez(
        filename,
        accel_raw=pkt["waveform"],
        time_s=time_s,
        magnitude_mg=magnitude_mg,
        metadata=json.dumps(
            {
                "impact_id": pkt["impact_id"],
                "timestamp_us": pkt["timestamp_us"],
                "label": label,
                "config": cfg,
                "features": pkt["features"],
            }
        ),
    )
    return filename


def log_run_meta(run_dir: Path, args, cfg: Dict):
    meta = {
        "label": args.label,
        "count": args.count,
        "port": args.port,
        "baud": args.baud,
        "started": datetime.now().isoformat(),
        "config": cfg,
        "features": ["peak_mag_mg", "peak_dev_mg", "rms_dev_mg", "decay_ms", "band_energy"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Collect labeled impact packets and save to disk.")
    parser.add_argument("--port", required=True, help="Serial port, e.g., /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate (default 921600)")
    parser.add_argument("--label", required=True, help="Label applied to this batch (string or int)")
    parser.add_argument("--count", type=int, default=30, help="Number of impacts to record")
    parser.add_argument("--out", type=Path, help="Run directory (default: data/run_<timestamp>)")
    args = parser.parse_args()

    run_dir = timestamp_dir(args.out)
    wave_dir = ensure_dirs(run_dir)
    features_path = run_dir / "features.csv"

    print(f"Run directory: {run_dir}")
    print(f"Listening on {args.port} @ {args.baud} baud for {args.count} impacts labeled '{args.label}'")

    seen_packets = 0
    header: List[str] | None = None
    run_meta_written = False

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:  # type: ignore[arg-type]
            while seen_packets < args.count:
                frame = ser.read_until(b"\x00")
                if not frame:
                    continue
                try:
                    pkt = parse_impact_frame(frame)
                except ValueError as exc:
                    print(f"[warn] drop frame: {exc}")
                    continue

                if not run_meta_written:
                    log_run_meta(run_dir, args, pkt["config"])
                    run_meta_written = True

                if header is None:
                    header = feature_header(len(pkt["features"]["band_energy"]))

                save_npz(wave_dir, pkt, args.label)

                row = {
                    "impact_id": pkt["impact_id"],
                    "label": args.label,
                    "timestamp_us": pkt["timestamp_us"],
                    "odr_hz": pkt["config"]["odr_hz"],
                    "fs_g": pkt["config"]["fs_g"],
                    "pretrigger_recorded": pkt["config"]["pretrigger_recorded"],
                    "stage1_count": pkt["stage1_count"],
                    "stage2_count": pkt["stage2_count"],
                    "peak_mag_mg": pkt["features"]["peak_mag_mg"],
                    "peak_dev_mg": pkt["features"]["peak_dev_mg"],
                    "rms_dev_mg": pkt["features"]["rms_dev_mg"],
                    "decay_ms": pkt["features"]["decay_ms"],
                }
                for idx, val in enumerate(pkt["features"]["band_energy"]):
                    row[f"band_{idx}"] = val

                append_feature_row(features_path, header, row)
                seen_packets += 1
                print(f"[saved] impact {pkt['impact_id']} ({seen_packets}/{args.count})")
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()

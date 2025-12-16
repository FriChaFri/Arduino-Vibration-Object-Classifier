#!/usr/bin/env python3
"""
impact_logger.py

Host-side helper that listens to the Teensy serial output, saves per-impact
features to a CSV file, and stores each raw waveform as its own CSV file.
Requires the ``pyserial`` package (``pip install pyserial``).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - host utility
    print("pyserial is required. Install it via 'pip install pyserial'.", file=sys.stderr)
    raise

FEATURE_HEADER = [
    "impact_id",
    "t_ms",
    "window_samples",
    "peak_mg",
    "rms_mg",
    "p2p_mg",
    "decay_ratio",
]


@dataclass
class WaveformRecord:
    impact_id: int
    timestamp_ms: int
    samples: List[float]


def parse_feature_line(parts: List[str]):
    if len(parts) != len(FEATURE_HEADER):
        return None
    try:
        impact_id = int(parts[0])
        timestamp_ms = int(parts[1])
        window_samples = int(parts[2])
        peak = float(parts[3])
        rms = float(parts[4])
        p2p = float(parts[5])
        decay = float(parts[6])
    except ValueError:
        return None
    return {
        "impact_id": impact_id,
        "t_ms": timestamp_ms,
        "window_samples": window_samples,
        "peak_mg": peak,
        "rms_mg": rms,
        "p2p_mg": p2p,
        "decay_ratio": decay,
    }


def parse_waveform_line(parts: List[str]):
    if len(parts) < 4:
        return None
    if parts[0].lower() != "waveform":
        return None
    try:
        impact_id = int(parts[1])
        timestamp_ms = int(parts[2])
        samples = [float(v) for v in parts[3:]]
    except ValueError:
        return None
    return WaveformRecord(impact_id, timestamp_ms, samples)


def ensure_dirs(base_dir: Path) -> Path:
    wave_dir = base_dir / "raw_waveforms"
    wave_dir.mkdir(parents=True, exist_ok=True)
    return wave_dir


def append_feature_row(path: Path, row: dict):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_waveform(wave_dir: Path, record: WaveformRecord):
    filename = f"impact_{record.impact_id:05d}.csv"
    path = wave_dir / filename
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "magnitude_mg"])
        for idx, value in enumerate(record.samples):
            writer.writerow([idx, value])


def read_serial(ser, wave_dir: Path, feature_path: Path):
    print("Logging impacts. Press Ctrl+C to stop.")
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(',')
        if parts[0].lower() == "waveform":
            record = parse_waveform_line(parts)
            if record:
                save_waveform(wave_dir, record)
                print(f"Saved waveform impact_id={record.impact_id} samples={len(record.samples)}")
            continue
        row = parse_feature_line(parts)
        if row:
            append_feature_row(feature_path, row)
            print(f"Logged features impact_id={row['impact_id']}")


def main():
    parser = argparse.ArgumentParser(description="Log impact features and waveforms from Teensy serial output")
    parser.add_argument("port", help="Serial port name (e.g. /dev/ttyACM0 or COM5)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument(
        "--output",
        default="data",
        help="Directory to store CSV outputs (default: data)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)
    wave_dir = ensure_dirs(base_dir)
    feature_path = base_dir / "impact_features.csv"

    with serial.Serial(args.port, args.baud, timeout=1) as ser:  # type: ignore[arg-type]
        try:
            read_serial(ser, wave_dir, feature_path)
        except KeyboardInterrupt:
            print("\nStopping logger.")


if __name__ == "__main__":
    main()

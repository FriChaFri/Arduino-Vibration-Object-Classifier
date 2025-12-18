# Vibration Classify

## Project Overview

This project is a proof-of-concept **embedded AI object classifier** that distinguishes objects based on the vibration response produced when they are dropped onto a platform.

A thin steel plate mounted on compliant TPU feet is instrumented with an **LSM6DS3TR-C IMU**. When an object impacts the plate, the resulting vibration signature is captured at high rate by the accelerometer. These signals are processed on-device by a **Teensy microcontroller**, with the long-term goal of performing real-time classification using a very small neural network.

The design prioritizes:

* Mechanical simplicity and repeatability
* High-rate, low-latency sensing
* Lightweight signal processing suitable for microcontrollers
* A clear end-to-end pipeline from physics → data → inference
* Clean separation between embedded firmware and host-side analysis

---

## Mechanical Setup

* **Platform:** Thin steel sheet metal plate
* **Supports:** Four 3D-printed TPU feet (corner-mounted)

  * Provide controlled compliance
  * Limit energy loss to the table
* **IMU Mounting:**

  * LSM6DS3TR-C mounted upside-down
  * Rigid 3D-printed mount
  * Mount glued directly to the steel plate for strong vibration coupling
* **Electronics:**

  * IMU on rigid mount
  * Teensy and breadboard mounted off to the side to avoid mechanical damping

This configuration allows the plate to ring after impact, producing distinct vibration signatures for different objects.

---

## Sensing and Data Capture

* **Sensor:** LSM6DS3TR-C 3-axis accelerometer over I2C
* **Sampling:**

  * Accelerometer configured for **1660 Hz ODR** (1.66 kHz)
  * ±8 g full-scale to avoid clipping
  * High-performance mode, gyro powered down
* **Trigger:**

  * Magnitude deviation from a drifting baseline with hysteresis + refractory
* **Capture Window (staged):**

  * Pre-trigger buffer: 128 samples (full-rate)
  * Stage A: 1024 samples at full ODR (contains pre-trigger)
  * Stage B: 1024 samples stored every 4th sample (decimated tail)
  * Total stored samples: 2048 (covers ~3 seconds at 1.66 kHz)

---

## Firmware Behavior

Two modes are supported (set `RUN_MODE` in `include/config.h`):

* **MODE_MONITOR:** stream summary stats at `PRINT_HZ` to confirm ODR / baseline stability.
* **MODE_COLLECT:** run the trigger + staged capture and emit binary **IMPACT** packets over serial.

On-device pipeline:

* Poll data-ready and read accel at ~1660 Hz, ±8 g.
* Maintain a baseline EMA of |a|, trigger on deviation with hysteresis + refractory.
* Capture staged window: pre-trigger + Stage A (1024 samples) + Stage B (decimated tail, 1024 samples @ /4).
* Extract features:
  * peak magnitude + deviation
  * RMS deviation (stage A)
  * decay time to 20% of peak
  * Goertzel band energies at 80 / 160 / 320 / 640 Hz
* Serialize as binary packet (COBS framed + CRC16) containing metadata + features + raw int16 XYZ samples.

---

## Host-Side Python Tooling

All non-embedded tasks are handled **off-device using Python**. This keeps the firmware simple and deterministic while enabling rapid experimentation.

Python tooling is used for:

* Capturing and parsing serial logs
* Storing datasets (features and raw waveforms)
* Visualizing vibration responses
* Training and evaluating machine learning models
* Exporting trained model weights to C/C++ headers for Teensy deployment

Python code lives outside the firmware build system and does **not** run on the Teensy.

---

## Machine Learning Approach

* **Model:** Very small neural network (e.g., shallow MLP)
* **Training:** Offline in Python using collected vibration datasets
* **Deployment:**

  * Trained weights embedded directly into firmware
  * Forward pass runs entirely on the Teensy
* **Classes:** Initially binary (e.g., hard vs. soft impact objects)
* **Output:**

  * Serial classification result
  * Optional LEDs for live demonstration

The emphasis is on:

* Tiny model size
* Predictable execution time
* Feasibility on constrained hardware

---

## Project Structure

```
Vibration Classify/
├── platformio.ini
├── AGENTS.md              # Codex workflow and constraints
├── include/
│   └── config.h           # Pins, constants, modes, tunables
├── src/
│   ├── main.cpp           # Entry point (setup / loop)
│   ├── imu_lsm6ds3trc.*   # IMU driver and configuration
│   ├── impact_capture.*   # Trigger + staged capture
│   ├── features.*         # Feature extraction
│   └── protocol.*         # Packet framing (COBS + CRC)
├── scripts/               # Host-side Python utilities
│   ├── collect_impacts.py # Labeled dataset collection
│   ├── protocol.py        # Packet decode helpers
│   ├── inspect_dataset.py # Quick stats / plots
│   ├── impact_logger.py   # Legacy CSV logger
│   └── plot_waveforms.py  # Legacy plotting helper
├── python/                # ML and data-processing code
│   ├── data_processing.py
│   ├── model_training.py
│   └── model_export.py
└── data/                  # Collected datasets (CSV/NPZ)
```

The structure is intentionally minimal while remaining scalable.

---

## Build and Run (Firmware)

This project uses **PlatformIO**.

Build / flash:

```bash
pio run
pio run -t upload
pio device monitor -b 921600   # monitor mode only
```

Key firmware configuration lives in `include/config.h` (modes, ODR/FS, trigger, window sizes).

**Monitor mode**
1) Set `RUN_MODE` to `MODE_MONITOR` in `include/config.h`.
2) Build + upload.
3) `pio device monitor -b 921600` to view per-axis stats and approx sample rate.

**Collect mode**
1) Set `RUN_MODE` to `MODE_COLLECT` in `include/config.h` (default).
2) Build + upload.
3) Use `scripts/collect_impacts.py` on the host to capture labeled batches:

```bash
python3 scripts/collect_impacts.py --port /dev/ttyACM0 --baud 921600 --label 0 --count 30 --out data/run_$(date +%Y%m%d_%H%M%S)
python3 scripts/collect_impacts.py --port /dev/ttyACM0 --baud 921600 --label 1 --count 30 --out data/run_$(date +%Y%m%d_%H%M%S)
```

Each run directory contains:
* `waves/impact_<id>_label_<x>.npz` with raw int16 XYZ samples, time axis, magnitude.
* `features.csv` with metadata + features for every impact.
* `meta.json` capturing run configuration.

Validate separability with:

```bash
python3 scripts/inspect_dataset.py data/run_YYYYMMDD_HHMMSS --plots
```

## Offline Training + Export

Train on CSV feature logs (labels grouped to categories) and export a Teensy-ready header:

```bash
# Train from all collected runs (CSV only), writes model artifacts under models/latest
python3 scripts/train_model.py --features-glob "data/run_*/features.csv" --outdir models/latest

# Export scaler + MLP weights/biases + ordered class labels to firmware header
python3 scripts/export_model.py --modeldir models/latest --out include/model_weights.h
```

Details:
* Label grouping: raw labels containing "eraser" → category `eraser`; labels containing "screw" → category `screw`; everything else is dropped (easy to extend later).
* Feature handling: uses numeric feature columns and ignores IDs/timestamps/config fields (impact_id, timestamp_us, odr_hz, fs_g, stage counts, filenames).
* Artifacts: `model.joblib`, `label_encoder.joblib`, and `training_metadata.json` in the chosen `models/` subdir; export writes `include/model_weights.h` with scaler stats, weights/bias, feature order, and class names.

---

## Current Status

* [x] IMU bring-up and verification
* [x] High-rate accelerometer configuration
* [x] Impact trigger with hysteresis and cooldown
* [x] Fixed-window waveform capture
* [x] Per-impact feature extraction
* [x] Structured serial output (features + optional waveforms)
* [ ] Dataset labeling
* [x] Offline model training in Python (CSV pipeline + grouped labels)
* [x] Model export to embedded firmware (header generator)
* [ ] On-device classifier inference

---

## Project Goal

Demonstrate a complete, real-time **embedded machine learning pipeline** on a microcontroller:

**mechanical system → sensing → feature extraction → neural network → classification**

with an emphasis on clarity, repeatability, and practical embedded constraints.

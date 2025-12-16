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

* **Sensor:** LSM6DS3TR-C 3-axis accelerometer
* **Interface:** I2C
* **Sampling:**

  * Accelerometer configured for **high-rate operation (~kHz ODR)**
  * ±2g full-scale for maximum sensitivity
  * High-performance mode enabled
* **Trigger:**

  * Impact detected via acceleration magnitude threshold
* **Capture Window:**

  * Fixed-length post-impact time window (configurable)

---

## Firmware Behavior (Current)

The firmware currently supports **impact-triggered capture** and structured serial output.

On-device behavior includes:

* Continuous accelerometer sampling
* Acceleration magnitude computation
* Impact detection with hysteresis and cooldown
* Fixed-length waveform capture after impact
* Lightweight feature extraction per impact:

  * Peak acceleration
  * RMS acceleration
  * Peak-to-peak
  * Simple ring-down decay metric
* Structured serial output:

  * One CSV feature line per impact
  * Optional raw waveform output for offline analysis

This validates:

* IMU configuration and timing
* Trigger robustness
* Data integrity and repeatability
* Suitability of signals for classification

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
│   ├── imu_lsm6ds3.h      # IMU driver and configuration
│   ├── modes.h            # Monitor / capture / infer modes
│   ├── features.h         # Feature extraction
│   └── model.h            # Classifier / NN inference (future)
├── scripts/               # Host-side Python utilities
│   ├── impact_logger.py
│   ├── plot_waveforms.py
│   └── parse_serial_log.py
├── python/                # ML and data-processing code
│   ├── dataset.py
│   ├── train_mlp.py
│   └── export_c_header.py
├── data/                  # Collected datasets (CSV)
│   └── README.md
└── test/
    └── README.md
```

The structure is intentionally minimal while remaining scalable.

---

## Build and Run (Firmware)

This project uses **PlatformIO**.

Typical workflow:

```bash
pio run
pio run -t upload
pio device monitor -b 115200
```

Key firmware configuration options (modes, thresholds, window length, output flags) are centralized in:

```
include/config.h
```

---

## Current Status

* [x] IMU bring-up and verification
* [x] High-rate accelerometer configuration
* [x] Impact trigger with hysteresis and cooldown
* [x] Fixed-window waveform capture
* [x] Per-impact feature extraction
* [x] Structured serial output (features + optional waveforms)
* [ ] Dataset labeling
* [ ] Offline model training in Python
* [ ] Model export to embedded firmware
* [ ] On-device classifier inference

---

## Project Goal

Demonstrate a complete, real-time **embedded machine learning pipeline** on a microcontroller:

**mechanical system → sensing → feature extraction → neural network → classification**

with an emphasis on clarity, repeatability, and practical embedded constraints.

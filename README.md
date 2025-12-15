# Vibration Classify

## Project Overview

This project is a proof-of-concept **embedded AI object classifier** that distinguishes objects based on the vibration response produced when they are dropped onto a platform.

A thin steel plate mounted on compliant TPU feet is instrumented with an **LSM6DS3TR-C IMU**. When an object impacts the plate, the resulting vibration signature is captured at high rate by the accelerometer. These signals are processed on-device by a **Teensy microcontroller**, with the long-term goal of performing real-time classification using a very small neural network.

The design prioritizes:

* Mechanical simplicity and repeatability
* High-rate, low-latency sensing
* Lightweight signal processing suitable for microcontrollers
* A clear end-to-end pipeline from physics → data → inference

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

  * Accelerometer configured for **6.66 kHz ODR**
  * ±2g full-scale for maximum sensitivity
  * High-performance mode enabled
* **Trigger (future stage):**

  * Impact detected via acceleration magnitude threshold
* **Capture Window (future stage):**

  * Fixed-length post-impact time window

---

## Signal Processing (Current)

The current firmware operates in **monitor mode**, continuously:

* Reading raw accelerometer data
* Converting raw values to physical units (mg, g, m/s²)
* Accumulating statistics per axis:

  * Mean
  * RMS
  * Peak-to-peak
* Printing summary statistics at a controlled rate (default: 50 Hz)
* Printing approximate sample throughput once per second

This validates:

* IMU configuration
* Data integrity
* Vibration observability

---

## Planned Feature Extraction

Future stages will extract low-cost features from the post-impact window, such as:

* Peak acceleration magnitude
* RMS energy
* Ring-down decay characteristics
* Coarse frequency-band energy

Features are chosen to be:

* Computationally inexpensive
* Robust to noise
* Suitable for real-time embedded inference

---

## Machine Learning Approach

* **Model:** Very small neural network (e.g., shallow MLP)
* **Training:** Offline on collected vibration datasets
* **Deployment:**

  * Trained weights embedded directly into firmware
  * Forward pass runs entirely on the Teensy
* **Classes:** Initially binary (e.g., hard vs soft object)
* **Output:**

  * Serial output
  * Optional LEDs for live demo

---

## Project Structure

```
Vibration Classify/
├── platformio.ini
├── include/
│   └── config.h          # Pins, constants, modes
├── src/
│   ├── main.cpp          # Entry point (setup/loop)
│   ├── imu_lsm6ds3.h     # IMU driver and configuration
│   ├── modes.h           # Monitor / collect / infer modes
│   ├── features.h        # Feature extraction (future)
│   └── model.h           # Classifier / NN inference (future)
├── lib/
│   └── README
├── test/
│   └── README
└── data/
    └── README.md         # Host-side datasets (CSV logs)
```

The structure is intentionally minimal to reduce overhead while remaining scalable.

---

## Build and Run

This project uses **PlatformIO**.

Typical workflow:

```bash
pio run
pio run -t upload
pio device monitor
```

Key configuration options (modes, rates, thresholds) are centralized in:

```
include/config.h
```

---

## Current Status

* [x] IMU bring-up and verification
* [x] High-rate accelerometer configuration
* [x] Continuous vibration monitoring with live statistics
* [ ] Impact trigger + capture window
* [ ] Dataset collection (CSV)
* [ ] Feature extraction pipeline
* [ ] On-device classifier inference

---

## Project Goal

Demonstrate a complete, real-time **embedded machine learning pipeline** on a microcontroller:

**mechanical system → sensing → feature extraction → neural network → classification**

with an emphasis on clarity, repeatability, and practical embedded constraints.

# Teensy Vibration Classifier (Codex instructions)

## Project Scope and Goal

Implement a proof-of-concept object classifier using post-impact accelerometer vibration data collected from a Teensy-based system.

End-to-end pipeline:

* Capture a fixed-length vibration window after an impact trigger
* Extract lightweight, embedded-friendly features
* Run a tiny neural network inference **on the Teensy**
* Emit results over Serial for validation and logging

In addition, use **Python on the host computer** for all non-Teensy tasks such as:

* Data logging and parsing of serial output
* Visualization and plotting of vibration waveforms
* Dataset preparation and labeling
* Training and evaluation of machine learning models
* Exporting trained model weights to Teensy-compatible C/C++ headers

The Teensy firmware must remain lightweight and deterministic; heavy computation and experimentation belongs in Python.

---

## How to Build / Run (PlatformIO)

All firmware builds use PlatformIO.

* Build (firmware only):

  ```bash
  pio run
  ```

* Upload to device:

  ```bash
  pio run -t upload
  ```

* Serial monitor:

  ```bash
  pio device monitor -b 115200
  ```

Rules:

* If the **build** command fails, stop immediately and report the exact error output.
* If the **upload** command fails, retry up to **two additional times**. If it still fails, stop and report the exact error output.
* Do **not** assume access to physical hardware unless explicitly stated; serial monitoring may not be available in all environments.

---

## Python Usage Policy

Python is used exclusively for **host-side tooling**, not for embedded firmware.

Allowed Python responsibilities:

* Serial log capture and offline parsing
* Feature inspection and visualization
* Training ML models (e.g., small MLPs)
* Model evaluation and iteration
* Exporting trained parameters into static C/C++ representations for Teensy

Constraints:

* Python code should be runnable on a standard desktop environment
* Prefer simple, explicit scripts over complex frameworks
* Training code should be clearly separated from export/deployment code

---

## Workflow Rules

1. **Plan first**: summarize intended file changes and commands before editing.
2. **Small diffs**: prefer incremental, reviewable changes.
3. **Verify firmware**: after embedded edits, run `pio run` and report results.
4. **No silent regressions**: never delete or disable working code without explanation and a clear revert path.
5. **Environment safety**: do not modify user shell configuration, permissions, or system packages unless explicitly instructed.

---

## Embedded Constraints (Teensy)

* Avoid dynamic memory allocation in hot paths
* Keep computation cheap and bounded (small feature sets, small NN)
* Use clear, named constants (`#define` or `constexpr`) for all tunables
* Favor deterministic control flow and predictable timing
* Serial output must be structured, machine-parseable, and rate-limited

---

## Expected Codex Behavior

* Treat this repository as the single source of truth
* Assume PlatformIO is the authoritative build system for firmware
* Assume Python scripts are auxiliary tooling, not part of the firmware build
* Clearly state when hardware access limits what can be tested
* Stop immediately on build or environment errors and report verbatim output

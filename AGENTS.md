# Teensy Vibration Classifier (Codex instructions)

## Goal
Implement a proof-of-concept object classifier using post-impact accelerometer vibration data:
- Capture a fixed window after trigger
- Extract lightweight features
- Run a tiny NN inference on Teensy
- Print class over Serial

## How to build / run (PlatformIO)
- Build: `pio run`
- Upload: `pio run -t upload`
- Serial monitor: `pio device monitor -b 115200`

If any command fails, stop and report the exact error output.

## Workflow rules
1. Plan first: summarize your intended file changes and command(s) you will run.
2. Make small diffs: prefer incremental commits/patches.
3. After edits: run `pio run` and report results.
4. Never delete working code without explaining why and offering a revert path.

## Embedded constraints
- Avoid dynamic allocation in hot paths.
- Keep compute cheap (small feature set, small NN).
- Prefer clear named constants (`#define` / `constexpr`)
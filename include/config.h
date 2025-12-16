/*
 * config.h
 * Project-wide configuration constants (pins, IMU address, printing rates).
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

// ---------------- Pins ----------------
#define PIN_IMU_SDA 18
#define PIN_IMU_SCL 19

// ---------------- LSM6DS3TR-C I2C address ----------------
#define IMU_ADDR 0x6A

// ---------------- Output rate (summary print) ----------------
static const uint32_t PRINT_HZ = 50; // lines/sec
static const uint32_t PRINT_INTERVAL_US = 1000000UL / PRINT_HZ;

// ---------------- Unit conversions ----------------
static const float ACCEL_MG_PER_LSB = 0.061f;  // ±2g sensitivity (mg/LSB)
static const float G_TO_MPS2        = 9.80665f;

// ---------------- Modes ----------------
#define MODE_MONITOR  0
#define MODE_COLLECT  1
#define MODE_INFER    2

// Select what the firmware does:
#define RUN_MODE MODE_MONITOR

// ---------------- Feature extractor controls ----------------
#ifndef ENABLE_CONTINUOUS_STATS
#define ENABLE_CONTINUOUS_STATS 1
#endif

#ifndef ENABLE_WAVEFORM_OUTPUT
#define ENABLE_WAVEFORM_OUTPUT 1
#endif

#ifndef WAVEFORM_DECIMATE
#define WAVEFORM_DECIMATE 1
#endif

static const float IMPACT_BASELINE_MG = 1000.0f;  // approx 1 g
static const float TRIGGER_MG        = 250.0f;    // |a|-1 g trigger level
static const float HYST_MG           = 50.0f;     // hysteresis band
static const uint32_t COOLDOWN_MS    = 200;       // lockout after impact
static const size_t WINDOW_SAMPLES   = 256;       // capture window length
static const float DECAY_EPSILON     = 1e-6f;     // protect divide-by-zero

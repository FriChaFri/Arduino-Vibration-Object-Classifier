/*
 * config.h
 * Project-wide configuration constants (pins, IMU settings, capture sizes).
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

// ---------------- Pins ----------------
#define PIN_IMU_SDA 18
#define PIN_IMU_SCL 19

// ---------------- Serial ----------------
static const uint32_t SERIAL_BAUD = 921600;

// ---------------- LSM6DS3TR-C I2C address ----------------
#define IMU_ADDR      0x6A
#define IMU_ADDR_ALT  0x69

// ---------------- IMU configuration ----------------
static const uint16_t IMU_ODR_HZ         = 1660;      // target accel ODR
static const uint8_t  IMU_FS_G           = 8;         // ±8 g full-scale
static const float    ACCEL_MG_PER_LSB   = 0.244f;    // sensitivity at ±8 g
static const uint32_t IMU_I2C_FAST_HZ    = 1000000UL; // try 1 MHz first
static const uint32_t IMU_I2C_SLOW_HZ    = 400000UL;  // fallback if needed
static const uint8_t  IMU_I2C_MAX_RETRY  = 3;

// ---------------- Capture configuration ----------------
static const size_t   PRETRIGGER_SAMPLES = 128;   // stored inside stage 1
static const size_t   STAGE1_SAMPLES     = 1024;  // full-rate samples
static const size_t   STAGE2_SAMPLES     = 1024;  // decimated tail samples
static const uint8_t  STAGE2_DECIMATION  = 4;     // keep 1 every D samples

// ---------------- Trigger / refractory ----------------
static const float    BASELINE_INIT_MG   = 1000.0f;   // approx 1 g
static const float    BASELINE_ALPHA     = 0.0015f;   // EMA update when idle
static const float    TRIGGER_MG         = 250.0f;    // |a|-baseline trigger
static const float    HYST_MG            = 50.0f;     // hysteresis band
static const uint32_t REFRACTORY_MS      = 300;       // lockout after capture

// ---------------- Feature extractor ----------------
static const float    DECAY_FRACTION     = 0.20f;     // time to peak*frac
static const uint8_t  NUM_BAND_FEATURES  = 4;
static const float    BAND_FREQS_HZ[NUM_BAND_FEATURES] = {
    80.0f, 160.0f, 320.0f, 640.0f
};
static const float    G_TO_MPS2          = 9.80665f;

// ---------------- Output rate (summary print) ----------------
static const uint32_t PRINT_HZ = 25; // lines/sec
static const uint32_t PRINT_INTERVAL_US = 1000000UL / PRINT_HZ;

// ---------------- Modes ----------------
#define MODE_MONITOR  0
#define MODE_COLLECT  1
#define MODE_INFER    2

// Select what the firmware does (monitoring vs dataset capture)
#define RUN_MODE MODE_COLLECT

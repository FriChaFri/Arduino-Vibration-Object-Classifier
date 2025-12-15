/*
 * config.h
 * Project-wide configuration constants (pins, IMU address, printing rates).
 */

#pragma once
#include <stdint.h>

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

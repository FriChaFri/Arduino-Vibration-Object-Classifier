/*
 * config.h
 * Central configuration for pins, thresholds, and capture parameters.
 */

#pragma once

// I2C pins
#define PIN_IMU_SDA 18
#define PIN_IMU_SCL 19

// IMU I2C address
#define IMU_ADDR 0x6A

// Impact detection
#define IMPACT_THRESHOLD_MG 200.0f   // trigger threshold (tune later)

// Capture window
#define CAPTURE_SAMPLES 512          // samples per impact window

// Modes
#define MODE_COLLECT 1
#define MODE_INFER   2

// Select mode here
#define RUN_MODE MODE_COLLECT

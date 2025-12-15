/*
 * imu_lsm6ds3.h
 * Low-level IMU configuration and accel read helpers.
 */

#pragma once
#include <Arduino.h>
#include <Wire.h>
#include "config.h"

inline void imuInit() {
    Wire.setSDA(PIN_IMU_SDA);
    Wire.setSCL(PIN_IMU_SCL);
    Wire.begin();
    delay(50);
    // TODO: write register configuration
}

inline void imuReadAccelMg(float &ax, float &ay, float &az) {
    // TODO: read raw accel and convert to mg
    ax = ay = az = 0.0f;
}

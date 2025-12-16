/*
 * imu_lsm6ds3trc.h
 * Robust LSM6DS3TR-C accelerometer bring-up for Teensy 4.1 (I2C).
 */

#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <stdint.h>

#include "config.h"

struct AccelSample {
    int16_t x;
    int16_t y;
    int16_t z;
};

bool imuBegin();
bool imuConfigureAccel();
bool imuDataReady();
bool imuReadSample(AccelSample &sample);
uint8_t imuActiveAddress();
uint8_t imuWhoAmI();


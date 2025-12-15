/*
 * imu_lsm6ds3.h
 * Minimal, low-level LSM6DS3TR-C support:
 *  - I2C register read/write
 *  - accel raw reads
 *  - “fast accel” configuration matching the old file
 */

#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <stdint.h>
#include "config.h"

// ---------------- Registers ----------------
#define REG_WHO_AM_I    0x0F

#define REG_CTRL1_XL    0x10  // accel ODR/FS/BW
#define REG_CTRL2_G     0x11  // gyro ODR/FS (we'll power it down)
#define REG_CTRL3_C     0x12  // BDU, IF_INC, etc.
#define REG_CTRL6_C     0x15  // includes XL_HM_MODE (high-perf disable bit)
#define REG_CTRL8_XL    0x17  // accel filtering options

#define REG_OUTX_L_XL   0x28  // accel output start

static const uint8_t WHO_AM_I_EXPECTED = 0x6A;

// ---------------- Low-level I2C helpers ----------------
static inline void imuWriteReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

static inline uint8_t imuReadReg(uint8_t reg) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(IMU_ADDR, (uint8_t)1);
    if (Wire.available()) return Wire.read();
    return 0xFF;
}

static inline void imuReadAccelRaw(int16_t &ax, int16_t &ay, int16_t &az) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(REG_OUTX_L_XL);
    Wire.endTransmission(false);

    Wire.requestFrom(IMU_ADDR, (uint8_t)6);
    if (Wire.available() >= 6) {
        uint8_t xl = Wire.read();
        uint8_t xh = Wire.read();
        uint8_t yl = Wire.read();
        uint8_t yh = Wire.read();
        uint8_t zl = Wire.read();
        uint8_t zh = Wire.read();

        ax = (int16_t)((xh << 8) | xl);
        ay = (int16_t)((yh << 8) | yl);
        az = (int16_t)((zh << 8) | zl);
    } else {
        ax = ay = az = 0;
    }
}

// Convenience: raw -> mg
static inline void imuReadAccelMg(float &ax_mg, float &ay_mg, float &az_mg) {
    int16_t ax_raw, ay_raw, az_raw;
    imuReadAccelRaw(ax_raw, ay_raw, az_raw);
    ax_mg = (float)ax_raw * ACCEL_MG_PER_LSB;
    ay_mg = (float)ay_raw * ACCEL_MG_PER_LSB;
    az_mg = (float)az_raw * ACCEL_MG_PER_LSB;
}

// ---------------- Configuration: match old file ----------------
//
// - CTRL3_C: BDU=1, IF_INC=1
// - CTRL2_G: gyro power-down
// - CTRL6_C: high-performance enabled (XL_HM_MODE = 0 via writing 0x00)
// - CTRL1_XL: ODR=6.66kHz, FS=±2g  => 0xA0
// - CTRL8_XL: LPF2 enable + HPCF settings => 0b11000000
//
static inline void imuConfigureFastAccel() {
    imuWriteReg(REG_CTRL3_C, 0b01000100); // BDU=1, IF_INC=1
    imuWriteReg(REG_CTRL2_G, 0x00);       // gyro off
    imuWriteReg(REG_CTRL6_C, 0x00);       // accel high-performance
    imuWriteReg(REG_CTRL1_XL, 0xA0);      // ODR=6.66kHz, FS=±2g
    imuWriteReg(REG_CTRL8_XL, 0b11000000);// optional filtering
}

// Wire bring-up helper
static inline void imuWireBegin() {
    Wire.setSDA(PIN_IMU_SDA);
    Wire.setSCL(PIN_IMU_SCL);
    Wire.begin();
    delay(50);
}

static inline uint8_t imuWhoAmI() {
    return imuReadReg(REG_WHO_AM_I);
}

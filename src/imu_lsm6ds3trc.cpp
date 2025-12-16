/*
 * imu_lsm6ds3trc.cpp
 * Low-level I2C helpers for LSM6DS3TR-C on Teensy 4.1.
 */

#include "imu_lsm6ds3trc.h"

#include <Arduino.h>
#include <Wire.h>

#include "config.h"

namespace {
// Registers
constexpr uint8_t REG_WHO_AM_I   = 0x0F;
constexpr uint8_t REG_CTRL1_XL   = 0x10;
constexpr uint8_t REG_CTRL2_G    = 0x11;
constexpr uint8_t REG_CTRL3_C    = 0x12;
constexpr uint8_t REG_CTRL6_C    = 0x15;
constexpr uint8_t REG_CTRL8_XL   = 0x17;
constexpr uint8_t REG_STATUS     = 0x1E;
constexpr uint8_t REG_OUTX_L_XL  = 0x28; // 6 bytes starting here

constexpr uint8_t WHOAMI_MAIN = 0x6A;
constexpr uint8_t WHOAMI_ALT  = 0x69;

uint8_t activeAddr = IMU_ADDR;
uint32_t i2cClockHz = IMU_I2C_FAST_HZ;

bool writeReg(uint8_t reg, uint8_t value) {
    for (uint8_t attempt = 0; attempt < IMU_I2C_MAX_RETRY; ++attempt) {
        Wire.beginTransmission(activeAddr);
        Wire.write(reg);
        Wire.write(value);
        uint8_t rc = Wire.endTransmission();
        if (rc == 0) {
            return true;
        }
        delayMicroseconds(200);
    }
    return false;
}

bool readReg(uint8_t reg, uint8_t &value) {
    for (uint8_t attempt = 0; attempt < IMU_I2C_MAX_RETRY; ++attempt) {
        Wire.beginTransmission(activeAddr);
        Wire.write(reg);
        uint8_t rc = Wire.endTransmission(false);
        if (rc != 0) {
            delayMicroseconds(200);
            continue;
        }
        uint8_t requested = Wire.requestFrom(activeAddr, (uint8_t)1);
        if (requested == 1 && Wire.available()) {
            value = Wire.read();
            return true;
        }
        delayMicroseconds(200);
    }
    return false;
}

bool readMulti(uint8_t startReg, uint8_t *buf, size_t len) {
    for (uint8_t attempt = 0; attempt < IMU_I2C_MAX_RETRY; ++attempt) {
        Wire.beginTransmission(activeAddr);
        Wire.write(startReg);
        uint8_t rc = Wire.endTransmission(false);
        if (rc != 0) {
            delayMicroseconds(200);
            continue;
        }
        uint8_t requested = Wire.requestFrom(activeAddr, (uint8_t)len);
        if (requested == len) {
            for (size_t i = 0; i < len && Wire.available(); ++i) {
                buf[i] = Wire.read();
            }
            return true;
        }
        delayMicroseconds(200);
    }
    return false;
}

bool detectAddress() {
    uint8_t who = 0;
    activeAddr = IMU_ADDR;
    if (readReg(REG_WHO_AM_I, who) && (who == WHOAMI_MAIN || who == WHOAMI_ALT)) {
        return true;
    }
    activeAddr = IMU_ADDR_ALT;
    if (readReg(REG_WHO_AM_I, who) && (who == WHOAMI_MAIN || who == WHOAMI_ALT)) {
        return true;
    }
    activeAddr = IMU_ADDR;
    return false;
}

void setI2CClock(uint32_t hz) {
    i2cClockHz = hz;
    Wire.setClock(hz);
    delay(2);
}
} // namespace

bool imuBegin() {
    Wire.setSDA(PIN_IMU_SDA);
    Wire.setSCL(PIN_IMU_SCL);
    Wire.begin();
    setI2CClock(IMU_I2C_FAST_HZ);

    if (!detectAddress()) {
        // Fallback to slower clock and retry detection.
        setI2CClock(IMU_I2C_SLOW_HZ);
        if (!detectAddress()) {
            return false;
        }
    }
    return true;
}

bool imuConfigureAccel() {
    auto applyCfg = []() -> bool {
        // CTRL3_C: BDU=1, IF_INC=1
        if (!writeReg(REG_CTRL3_C, 0b01000100)) {
            return false;
        }
        // CTRL2_G: gyro power-down
        if (!writeReg(REG_CTRL2_G, 0x00)) {
            return false;
        }
        // CTRL6_C: ensure high-performance (XL_HM_MODE=0)
        if (!writeReg(REG_CTRL6_C, 0x00)) {
            return false;
        }
        // CTRL1_XL: ODR=1.66 kHz, FS=±8 g, BW=ODR/2
        const uint8_t odr_bits = 0b1001 << 4; // 1.66 kHz
        const uint8_t fs_bits  = 0b10 << 2;   // ±8 g
        const uint8_t bw_bits  = 0b00;        // 400 Hz bandwidth selection
        if (!writeReg(REG_CTRL1_XL, (uint8_t)(odr_bits | fs_bits | bw_bits))) {
            return false;
        }
        // CTRL8_XL: default filters off (leave high bandwidth); could enable LPF2 if needed
        if (!writeReg(REG_CTRL8_XL, 0x00)) {
            return false;
        }
        return true;
    };

    if (applyCfg()) {
        return true;
    }
    // Retry once at slower I2C if the fast path failed.
    if (i2cClockHz != IMU_I2C_SLOW_HZ) {
        setI2CClock(IMU_I2C_SLOW_HZ);
        detectAddress();
        return applyCfg();
    }
    return false;
}

bool imuDataReady() {
    uint8_t status = 0;
    if (!readReg(REG_STATUS, status)) {
        return false;
    }
    return (status & 0x01) != 0; // XLDA
}

bool imuReadSample(AccelSample &sample) {
    uint8_t raw[6] = {0};
    if (!readMulti(REG_OUTX_L_XL, raw, sizeof(raw))) {
        return false;
    }
    sample.x = (int16_t)((raw[1] << 8) | raw[0]);
    sample.y = (int16_t)((raw[3] << 8) | raw[2]);
    sample.z = (int16_t)((raw[5] << 8) | raw[4]);
    return true;
}

uint8_t imuActiveAddress() {
    return activeAddr;
}

uint8_t imuWhoAmI() {
    uint8_t who = 0;
    if (!readReg(REG_WHO_AM_I, who)) {
        return 0x00;
    }
    return who;
}

/*
 * main.cpp
 * Minimal entry point that selects a mode and reproduces the old startup prints.
 */

#include <Arduino.h>
#include "config.h"
#include "imu_lsm6ds3.h"
#include "modes.h"

void setup() {
    Serial.begin(115200);

    imuWireBegin();

    uint8_t who = imuWhoAmI();
    Serial.print("LSM6DS3 WHO_AM_I = 0x");
    Serial.println(who, HEX);

    if (who != WHO_AM_I_EXPECTED) {
        Serial.println("WARNING: WHO_AM_I mismatch. Check wiring (SDA/SCL), address (0x6A), and power.");
    }

    imuConfigureFastAccel();

    Serial.println("Configured accel: ODR=6.66kHz, FS=±2g, high-performance enabled.");
    Serial.println("Printing summary stats (per print interval): mean / RMS / peak-to-peak on each axis.");
    Serial.println();
}

void loop() {
#if RUN_MODE == MODE_MONITOR
    monitorLoop();
#elif RUN_MODE == MODE_COLLECT
    collectLoop();
#elif RUN_MODE == MODE_INFER
    inferLoop();
#else
    monitorLoop();
#endif
}

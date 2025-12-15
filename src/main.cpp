/*
 * main.cpp
 * Entry point. Handles setup/loop and high-level mode selection.
 */

#include <Arduino.h>
#include "config.h"
#include "imu_lsm6ds3.h"
#include "features.h"
#include "model.h"
#include "modes.h"

void setup() {
    Serial.begin(115200);
    imuInit();

#if RUN_MODE == MODE_COLLECT
    Serial.println("Mode: DATA COLLECTION");
#elif RUN_MODE == MODE_INFER
    Serial.println("Mode: INFERENCE");
#endif
}

void loop() {
#if RUN_MODE == MODE_COLLECT
    collectLoop();
#elif RUN_MODE == MODE_INFER
    inferLoop();
#endif
}

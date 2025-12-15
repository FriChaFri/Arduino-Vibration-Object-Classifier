/*
 * modes.h
 * Behavior for data collection vs inference.
 */

#pragma once
#include <Arduino.h>
#include "imu_lsm6ds3.h"
#include "features.h"
#include "model.h"

inline void collectLoop() {
    float ax, ay, az;
    imuReadAccelMg(ax, ay, az);

    if (detectImpact(ax, ay, az)) {
        // TODO: capture window + print CSV row
        Serial.println("impact_detected");
        delay(500); // simple debounce
    }
}

inline void inferLoop() {
    float ax, ay, az;
    imuReadAccelMg(ax, ay, az);

    if (detectImpact(ax, ay, az)) {
        FeatureVector f = {};
        int cls = classify(f);
        Serial.print("class=");
        Serial.println(cls);
        delay(500);
    }
}

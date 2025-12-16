/*
 * protocol.cpp
 * Impact packet builder with CRC16 and COBS framing.
 */

#include "protocol.h"

#include <string.h>

#include "config.h"

namespace {
constexpr size_t MAX_PAYLOAD_BYTES = 18000;

inline bool pushByte(uint8_t *buf, size_t &idx, uint8_t value) {
    if (idx >= MAX_PAYLOAD_BYTES) return false;
    buf[idx++] = value;
    return true;
}

inline bool pushU16(uint8_t *buf, size_t &idx, uint16_t value) {
    return pushByte(buf, idx, (uint8_t)(value & 0xFF)) &&
           pushByte(buf, idx, (uint8_t)((value >> 8) & 0xFF));
}

inline bool pushU32(uint8_t *buf, size_t &idx, uint32_t value) {
    return pushU16(buf, idx, (uint16_t)(value & 0xFFFF)) &&
           pushU16(buf, idx, (uint16_t)((value >> 16) & 0xFFFF));
}

inline bool pushFloat(uint8_t *buf, size_t &idx, float value) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    uint8_t raw[4];
    memcpy(raw, &value, sizeof(float));
    for (size_t i = 0; i < sizeof(raw); ++i) {
        if (!pushByte(buf, idx, raw[i])) return false;
    }
    return true;
}
} // namespace

uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0; bit < 8; ++bit) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

size_t cobsEncode(const uint8_t *input, size_t length, uint8_t *output, size_t max_out) {
    if (length == 0 || max_out == 0) {
        return 0;
    }

    size_t read_index = 0;
    size_t write_index = 1;
    size_t code_index = 0;
    uint8_t code = 1;

    while (read_index < length) {
        if (input[read_index] == 0) {
            if (write_index >= max_out) return 0;
            output[code_index] = code;
            code_index = write_index++;
            code = 1;
            read_index++;
        } else {
            if (write_index >= max_out) return 0;
            output[write_index++] = input[read_index++];
            code++;
            if (code == 0xFF) {
                if (code_index >= max_out) return 0;
                output[code_index] = code;
                code_index = write_index++;
                code = 1;
            }
        }
    }

    if (code_index >= max_out) return 0;
    output[code_index] = code;
    return write_index;
}

bool encodeImpactPacket(const ImpactRecord &record, uint8_t *out, size_t max_out, size_t &encoded_len) {
    static uint8_t payload[MAX_PAYLOAD_BYTES];
    size_t idx = 0;

    if (!pushByte(payload, idx, PACKET_TYPE_IMPACT)) return false;
    if (!pushByte(payload, idx, PROTOCOL_VERSION)) return false;
    if (!pushU32(payload, idx, record.impact_id)) return false;
    if (!pushU32(payload, idx, record.trigger_time_us)) return false;
    if (!pushU16(payload, idx, IMU_ODR_HZ)) return false;
    if (!pushByte(payload, idx, IMU_FS_G)) return false;
    if (!pushByte(payload, idx, 0)) return false; // reserved
    if (!pushU16(payload, idx, (uint16_t)STAGE1_SAMPLES)) return false;
    if (!pushU16(payload, idx, (uint16_t)STAGE2_SAMPLES)) return false;
    if (!pushByte(payload, idx, STAGE2_DECIMATION)) return false;
    if (!pushU16(payload, idx, (uint16_t)PRETRIGGER_SAMPLES)) return false;
    if (!pushU16(payload, idx, record.pretrigger_recorded)) return false;
    if (!pushFloat(payload, idx, ACCEL_MG_PER_LSB)) return false;
    if (!pushFloat(payload, idx, record.baseline_mg)) return false;
    if (!pushByte(payload, idx, NUM_BAND_FEATURES)) return false;
    if (!pushByte(payload, idx, 0)) return false;       // reserved
    if (!pushU16(payload, idx, 0)) return false;        // reserved/alignment

    // Features
    if (!pushFloat(payload, idx, record.features.peak_mag_mg)) return false;
    if (!pushFloat(payload, idx, record.features.peak_dev_mg)) return false;
    if (!pushFloat(payload, idx, record.features.rms_dev_mg)) return false;
    if (!pushFloat(payload, idx, record.features.decay_ms)) return false;
    for (uint8_t b = 0; b < NUM_BAND_FEATURES; ++b) {
        if (!pushFloat(payload, idx, record.features.band_energy[b])) return false;
    }

    if (!pushU16(payload, idx, record.stage1_count)) return false;
    if (!pushU16(payload, idx, record.stage2_count)) return false;

    // Waveform: stage1 then stage2
    const size_t total_stage1 = record.stage1_count;
    const size_t total_stage2 = record.stage2_count;
    for (size_t i = 0; i < total_stage1; ++i) {
        if (!pushU16(payload, idx, (uint16_t)record.stage1[i].x)) return false;
        if (!pushU16(payload, idx, (uint16_t)record.stage1[i].y)) return false;
        if (!pushU16(payload, idx, (uint16_t)record.stage1[i].z)) return false;
    }
    for (size_t i = 0; i < total_stage2; ++i) {
        if (!pushU16(payload, idx, (uint16_t)record.stage2[i].x)) return false;
        if (!pushU16(payload, idx, (uint16_t)record.stage2[i].y)) return false;
        if (!pushU16(payload, idx, (uint16_t)record.stage2[i].z)) return false;
    }

    // Append CRC16 over payload so far
    if (idx + 2 > MAX_PAYLOAD_BYTES) {
        return false;
    }
    const uint16_t crc = crc16_ccitt(payload, idx);
    if (!pushU16(payload, idx, crc)) return false;

    const size_t encoded = cobsEncode(payload, idx, out, max_out);
    if (encoded == 0 || encoded + 1 > max_out) {
        return false;
    }
    out[encoded] = 0x00; // delimiter
    encoded_len = encoded + 1;
    return true;
}


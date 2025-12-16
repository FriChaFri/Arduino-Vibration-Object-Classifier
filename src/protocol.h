/*
 * protocol.h
 * Binary packet framing (COBS + CRC16) for impact transfers.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "impact_capture.h"

static const uint8_t PACKET_TYPE_IMPACT = 0x01;
static const uint8_t PROTOCOL_VERSION   = 0x01;

// Maximum encoded packet size (covers metadata + 2048 samples * 3 int16)
static const size_t MAX_ENCODED_PACKET_BYTES = 20000;

uint16_t crc16_ccitt(const uint8_t *data, size_t len);
size_t cobsEncode(const uint8_t *input, size_t length, uint8_t *output, size_t max_out);
bool encodeImpactPacket(const ImpactRecord &record, uint8_t *out, size_t max_out, size_t &encoded_len);


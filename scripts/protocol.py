#!/usr/bin/env python3
"""
protocol.py

Host-side helpers to decode binary IMPACT packets produced by the Teensy firmware.
Framing: COBS + trailing 0x00 delimiter, CRC16-CCITT for integrity.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

import numpy as np

PACKET_TYPE_IMPACT = 0x01
PROTOCOL_VERSION = 0x01


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def cobs_decode(data: bytes) -> bytes:
    if not data:
        return b""
    out = bytearray()
    idx = 0
    length = len(data)
    while idx < length:
        code = data[idx]
        idx += 1
        if code == 0 or idx + code - 1 > length and code != 1:
            raise ValueError("Invalid COBS frame")
        for _ in range(code - 1):
            out.append(data[idx])
            idx += 1
        if code < 0xFF and idx < length:
            out.append(0)
    return bytes(out)


def decode_frame(frame: bytes) -> bytes:
    """Frame includes the trailing 0x00 delimiter."""
    if not frame or frame[-1] != 0:
        raise ValueError("Frame missing delimiter")
    encoded = frame[:-1]
    return cobs_decode(encoded)


def parse_impact_payload(payload: bytes) -> Dict:
    """
    Parse a decoded (COBS-removed) impact payload. Returns a dict with metadata, features,
    and waveform numpy array.
    """
    if len(payload) < 4:
        raise ValueError("Payload too short")

    crc_rx = struct.unpack_from("<H", payload, len(payload) - 2)[0]
    body = payload[:-2]
    if crc16_ccitt(body) != crc_rx:
        raise ValueError("CRC mismatch")

    offset = 0
    pkt_type, version = struct.unpack_from("<BB", body, offset)
    offset += 2
    if pkt_type != PACKET_TYPE_IMPACT or version != PROTOCOL_VERSION:
        raise ValueError("Unsupported packet type/version")

    impact_id, timestamp_us = struct.unpack_from("<II", body, offset)
    offset += 8
    odr_hz, = struct.unpack_from("<H", body, offset)
    offset += 2
    fs_g, = struct.unpack_from("<B", body, offset)
    offset += 1
    offset += 1  # reserved
    cfg_stage1, cfg_stage2 = struct.unpack_from("<HH", body, offset)
    offset += 4
    stage2_decim, = struct.unpack_from("<B", body, offset)
    offset += 1
    pre_cfg, pre_recorded = struct.unpack_from("<HH", body, offset)
    offset += 4
    mg_per_lsb, baseline_mg = struct.unpack_from("<ff", body, offset)
    offset += 8
    num_bands, _reserved_b = struct.unpack_from("<BB", body, offset)
    offset += 2
    offset += 2  # reserved/alignment

    peak_mag, peak_dev, rms_dev, decay_ms = struct.unpack_from("<ffff", body, offset)
    offset += 16

    band_energy = list(struct.unpack_from(f"<{num_bands}f", body, offset))
    offset += 4 * num_bands

    stage1_count, stage2_count = struct.unpack_from("<HH", body, offset)
    offset += 4
    total_samples = stage1_count + stage2_count

    expected_bytes = total_samples * 3 * 2
    if offset + expected_bytes > len(body):
        raise ValueError("Waveform length mismatch")

    waveform = np.frombuffer(body, dtype="<i2", count=total_samples * 3, offset=offset).copy()
    waveform = waveform.reshape((total_samples, 3))

    return {
        "impact_id": impact_id,
        "timestamp_us": timestamp_us,
        "config": {
            "odr_hz": odr_hz,
            "fs_g": fs_g,
            "stage1_size": cfg_stage1,
            "stage2_size": cfg_stage2,
            "stage2_decimation": stage2_decim,
            "pretrigger_config": pre_cfg,
            "pretrigger_recorded": pre_recorded,
            "mg_per_lsb": mg_per_lsb,
            "baseline_mg": baseline_mg,
            "num_bands": num_bands,
        },
        "features": {
            "peak_mag_mg": peak_mag,
            "peak_dev_mg": peak_dev,
            "rms_dev_mg": rms_dev,
            "decay_ms": decay_ms,
            "band_energy": band_energy,
        },
        "stage1_count": stage1_count,
        "stage2_count": stage2_count,
        "waveform": waveform,
    }


def parse_impact_frame(frame: bytes) -> Dict:
    """Decode a full frame (COBS + delimiter) into a structured dict."""
    payload = decode_frame(frame)
    return parse_impact_payload(payload)


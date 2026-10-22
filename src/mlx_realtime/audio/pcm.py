"""Mono 16-bit little-endian PCM: rates and duration arithmetic."""

from __future__ import annotations

SAMPLE_WIDTH = 2
"""Bytes per sample (s16le, mono)."""

WIRE_RATE = 24_000
"""The rate of ``audio/pcm`` on the wire, both directions."""

CANONICAL_RATE = 16_000
"""The rate at which the gateway holds input audio (ARCHITECTURE §4.3)."""


def samples_to_ms(samples: int, rate: int) -> int:
    return samples * 1000 // rate


def ms_to_samples(ms: int, rate: int) -> int:
    return ms * rate // 1000


def bytes_to_ms(nbytes: int, rate: int) -> int:
    return samples_to_ms(nbytes // SAMPLE_WIDTH, rate)


def ms_to_bytes(ms: int, rate: int) -> int:
    return ms_to_samples(ms, rate) * SAMPLE_WIDTH

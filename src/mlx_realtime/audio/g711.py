"""G.711 μ-law and A-law (ITU-T G.711) with lookup tables (ARCHITECTURE §4.3, M7).

The tables follow the algorithm of CPython's ``Modules/audioop.c`` (``st_14linear2ulaw``,
``st_linear2alaw`` and their inverses, from the Sun reference code): 256 entries to decode, and
65 536 to encode, one per 16-bit sample. M0's S7 found that Sun's later ``g711.c`` disagrees
with ``audioop`` on 381 μ-law and 896 A-law negative inputs, so that one is not ported. The
tests check every code and every sample against ``audioop`` (still in Python 3.12, deprecated);
nothing here depends on it at run time.

G.711 samples are 8 bits at 8 kHz (D3): one byte is one sample, any byte count is whole.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

type Law = Literal["ulaw", "alaw"]

RATE = 8000
"""G.711's sample rate (D3)."""

_QUANT_MASK = 0x0F
_SEG_SHIFT = 4
_SEG_MASK = 0x70
_SIGN_BIT = 0x80
_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def _ulaw_decode_table() -> npt.NDArray[np.int16]:
    """``st_ulaw2linear16`` for every code."""
    codes = ~np.arange(256, dtype=np.int32) & 0xFF
    t = ((codes & _QUANT_MASK) << 3) + _ULAW_BIAS
    t <<= (codes & _SEG_MASK) >> _SEG_SHIFT
    return np.where(codes & _SIGN_BIT, _ULAW_BIAS - t, t - _ULAW_BIAS).astype(np.int16)


def _alaw_decode_table() -> npt.NDArray[np.int16]:
    """``st_alaw2linear16`` for every code."""
    codes = np.arange(256, dtype=np.int32) ^ 0x55
    t = (codes & _QUANT_MASK) << 4
    segment = (codes & _SEG_MASK) >> _SEG_SHIFT
    t = np.where(
        segment == 0, t + 8, np.where(segment == 1, t + 0x108, (t + 0x108) << (segment - 1))
    )
    return np.where(codes & _SIGN_BIT, t, -t).astype(np.int16)


def _every_sample() -> npt.NDArray[np.int32]:
    """Every 16-bit sample, at the index of its unsigned bit pattern."""
    return np.arange(65536, dtype=np.int32).astype(np.uint16).view(np.int16).astype(np.int32)


def _ulaw_encode_table() -> npt.NDArray[np.uint8]:
    """``st_14linear2ulaw(sample >> 2)`` for every 16-bit sample."""
    value = _every_sample() >> 2
    mask = np.where(value < 0, 0x7F, 0xFF)
    magnitude = np.minimum(np.abs(value), _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    segment = np.searchsorted(_ULAW_SEG_END, magnitude)  # audioop's search(): first end >= value
    code = (segment << _SEG_SHIFT) | ((magnitude >> (segment + 1)) & _QUANT_MASK)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ mask).astype(np.uint8)


def _alaw_encode_table() -> npt.NDArray[np.uint8]:
    """``st_linear2alaw(sample >> 3)`` for every 16-bit sample."""
    value = _every_sample() >> 3
    mask = np.where(value >= 0, 0xD5, 0x55)
    magnitude = np.where(value >= 0, value, -value - 1)
    segment = np.searchsorted(_ALAW_SEG_END, magnitude)
    shift = np.where(segment < 2, 1, segment)
    code = (segment << _SEG_SHIFT) | ((magnitude >> shift) & _QUANT_MASK)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ mask).astype(np.uint8)


DECODE: dict[Law, npt.NDArray[np.int16]] = {
    "ulaw": _ulaw_decode_table(),
    "alaw": _alaw_decode_table(),
}
"""Code → 16-bit sample (256 entries per law)."""

ENCODE: dict[Law, npt.NDArray[np.uint8]] = {
    "ulaw": _ulaw_encode_table(),
    "alaw": _alaw_encode_table(),
}
"""16-bit sample, indexed by its unsigned bit pattern → code (65 536 entries per law)."""

SILENCE: dict[Law, int] = {"ulaw": 0xFF, "alaw": 0xD5}
"""The code of a zero sample."""


def decode_samples(data: bytes, law: Law) -> npt.NDArray[np.int16]:
    """G.711 codes to 16-bit samples."""
    return DECODE[law][np.frombuffer(data, dtype=np.uint8)]


def encode_samples(samples: npt.NDArray[np.int16], law: Law) -> bytes:
    """16-bit samples to G.711 codes."""
    return ENCODE[law][samples.astype(np.int16, copy=False).view(np.uint16)].tobytes()


def decode(data: bytes, law: Law) -> bytes:
    """G.711 codes to mono s16le PCM at the same rate."""
    return decode_samples(data, law).astype("<i2", copy=False).tobytes()


def encode(pcm: bytes, law: Law) -> bytes:
    """Mono s16le PCM (whole samples) to G.711 codes."""
    return encode_samples(np.frombuffer(pcm, dtype="<i2").astype(np.int16), law)

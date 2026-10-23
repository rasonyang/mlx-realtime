"""The only module that changes sample rates (ARCHITECTURE §4.3)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import soxr  # pyright: ignore[reportMissingTypeStubs]  # typed inline, no py.typed marker

_SCALE = 32768.0

type Quality = Literal["QQ", "LQ", "MQ", "HQ", "VHQ"]
"""soxr's quality recipes. HQ is the default; LQ halves the filter delay (M0 S7: 8 → 16 kHz
lags 64 ms on average with HQ, 32 ms with LQ) at the cost of a passband that rolls off from
3 kHz at 8 kHz (-8 dB at 3.4 kHz)."""


class Resampler:
    """A stateful mono PCM16 resampler: one soxr stream (quality HQ unless asked) per stream and
    direction.

    soxr runs on float32 here. Its int16 path dithers on every call, so chunked output would
    differ from one-shot output by a few LSB; on float32 a chunked stream is bit-identical to
    resampling the whole signal at once.
    """

    def __init__(self, in_rate: int, out_rate: int, quality: Quality = "HQ") -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.quality: Quality = quality
        self._stream = self._new_stream()

    def process(self, pcm: bytes) -> bytes:
        """Resample the next chunk; output lags input by the filter delay until ``flush``."""
        if self._stream is None:
            return pcm
        if not pcm:
            return b""
        return _to_pcm(self._stream.resample_chunk(_to_float(pcm), last=False))

    def flush(self) -> bytes:
        """End the stream, return the delayed tail, and start a fresh stream for later input."""
        if self._stream is None:
            return b""
        tail = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        self._stream = self._new_stream()
        return _to_pcm(tail)

    def _new_stream(self) -> soxr.ResampleStream | None:
        if self.in_rate == self.out_rate:
            return None
        return soxr.ResampleStream(
            self.in_rate, self.out_rate, 1, dtype="float32", quality=self.quality
        )


def _to_float(pcm: bytes) -> npt.NDArray[np.float32]:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(_SCALE)


def _to_pcm(samples: npt.NDArray[np.floating]) -> bytes:
    return np.clip(np.rint(samples * _SCALE), -32768, 32767).astype("<i2").tobytes()

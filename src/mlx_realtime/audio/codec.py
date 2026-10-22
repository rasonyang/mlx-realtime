"""Wire audio formats to and from canonical PCM (ARCHITECTURE §4.3).

M1 serves ``audio/pcm`` at 24 kHz in both directions; G.711 (``audio/pcmu``, ``audio/pcma``)
arrives in M7.
"""

from __future__ import annotations

from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, WIRE_RATE, ms_to_bytes
from mlx_realtime.audio.resample import Resampler


class InputDecoder:
    """Client audio (``audio/pcm``, 24 kHz s16le) to canonical 16 kHz PCM16."""

    def __init__(self) -> None:
        self._resampler = Resampler(WIRE_RATE, CANONICAL_RATE)

    def decode(self, pcm: bytes) -> bytes:
        return self._resampler.process(pcm)

    def flush(self) -> bytes:
        """Return the resampler's delayed tail; the next ``decode`` starts a new stream."""
        return self._resampler.flush()


class OutputEncoder:
    """Synthesized PCM to ``audio/pcm`` deltas at 24 kHz, each at most ``max_delta_ms`` long.

    One encoder serves one synthesis request: ``flush`` at its end releases the resampler tail
    when the provider's rate differs from the wire rate.
    """

    def __init__(self, max_delta_ms: int) -> None:
        self._max_bytes = max(ms_to_bytes(max_delta_ms, WIRE_RATE), SAMPLE_WIDTH)
        self._resampler: Resampler | None = None

    def encode(self, pcm: bytes, sample_rate: int) -> list[bytes]:
        if self._resampler is None:
            self._resampler = Resampler(sample_rate, WIRE_RATE)
        elif self._resampler.in_rate != sample_rate:
            raise ValueError("the sample rate changed within one synthesis request")
        return self._split(self._resampler.process(pcm))

    def flush(self) -> list[bytes]:
        if self._resampler is None:
            return []
        return self._split(self._resampler.flush())

    def _split(self, pcm: bytes) -> list[bytes]:
        return [
            pcm[start : start + self._max_bytes] for start in range(0, len(pcm), self._max_bytes)
        ]

"""Wire audio formats to and from canonical PCM (ARCHITECTURE §4.3).

Input: ``audio/pcm`` (24 kHz s16le) is resampled to the canonical 16 kHz; ``audio/pcmu`` and
``audio/pcma`` (G.711, 8 kHz) are decoded through the lookup tables, then resampled 8 → 16 kHz.
Output: synthesized PCM is resampled to the format's rate, G.711-encoded when asked, and cut
into deltas of at most ``max_delta_ms``.
"""

from __future__ import annotations

from mlx_realtime.audio import g711
from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.audio.pcm import CANONICAL_RATE
from mlx_realtime.audio.resample import Quality, Resampler


class InputDecoder:
    """Client audio in ``audio_format`` to canonical 16 kHz PCM16. ``g711_quality`` is the
    soxr quality of the 8 → 16 kHz step (24 kHz PCM always resamples at HQ)."""

    def __init__(self, audio_format: AudioFormat = PCM, g711_quality: Quality = "HQ") -> None:
        self.audio_format = audio_format
        quality: Quality = "HQ" if audio_format.law is None else g711_quality
        self._resampler = Resampler(audio_format.rate, CANONICAL_RATE, quality)

    def decode(self, data: bytes) -> bytes:
        """Whole samples of the format in (any byte count for G.711, even for PCM)."""
        return self._resampler.process(to_linear(data, self.audio_format))

    def flush(self) -> bytes:
        """Return the resampler's delayed tail; the next ``decode`` starts a new stream."""
        return self._resampler.flush()


class OutputEncoder:
    """Synthesized PCM to deltas in ``audio_format``, each at most ``max_delta_ms`` long.

    One encoder serves one response: a stateful resampler from the provider's rate to the
    format's (``quality``), whose tail ``flush`` releases at the end of each synthesis request
    (a segment), so every segment's audio is complete when its last delta goes out. A new
    segment may come at a different rate (the fallback TTS speaks it).
    """

    def __init__(self, max_delta_ms: int, audio_format: AudioFormat = PCM, quality: Quality = "HQ"):
        self.audio_format = audio_format
        self._max_bytes = max(audio_format.ms_to_bytes(max_delta_ms), audio_format.sample_width)
        self._quality: Quality = quality
        self._resampler: Resampler | None = None

    def encode(self, pcm: bytes, sample_rate: int) -> list[bytes]:
        if self._resampler is None:
            self._resampler = Resampler(sample_rate, self.audio_format.rate, self._quality)
        elif self._resampler.in_rate != sample_rate:
            raise ValueError("the sample rate changed within one synthesis request")
        return self._split(self._resampler.process(pcm))

    def flush(self) -> list[bytes]:
        """End the synthesis request: its resampler tail, as deltas."""
        resampler, self._resampler = self._resampler, None
        if resampler is None:
            return []
        return self._split(resampler.flush())

    def _split(self, pcm: bytes) -> list[bytes]:
        data = from_linear(pcm, self.audio_format)
        step = self._max_bytes
        return [data[start : start + step] for start in range(0, len(data), step)]


def to_linear(data: bytes, audio_format: AudioFormat) -> bytes:
    """Wire audio to s16le PCM at the format's own rate."""
    law = audio_format.law
    return data if law is None else g711.decode(data, law)


def from_linear(pcm: bytes, audio_format: AudioFormat) -> bytes:
    """s16le PCM at the format's rate to wire audio."""
    law = audio_format.law
    return pcm if law is None else g711.encode(pcm, law)

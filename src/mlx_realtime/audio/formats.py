"""The wire audio formats (GA ``RealtimeAudioFormats``) and their duration arithmetic.

``audio/pcm`` is s16le mono at 24 kHz; ``audio/pcmu`` and ``audio/pcma`` are G.711 at 8 kHz
(D3), one byte per sample. Every byte count ↔ millisecond conversion of wire audio goes through
the format, so the input inbox, the output ledger, truncation, D5, pacing and the delta size
count G.711 right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_realtime.audio.g711 import RATE as G711_RATE
from mlx_realtime.audio.g711 import Law
from mlx_realtime.audio.pcm import SAMPLE_WIDTH, WIRE_RATE

type FormatType = Literal["audio/pcm", "audio/pcmu", "audio/pcma"]

FORMAT_TYPES: tuple[FormatType, ...] = ("audio/pcm", "audio/pcmu", "audio/pcma")


@dataclass(frozen=True, slots=True)
class AudioFormat:
    type: FormatType = "audio/pcm"

    @property
    def law(self) -> Law | None:
        """The G.711 law, or None for linear PCM."""
        match self.type:
            case "audio/pcmu":
                return "ulaw"
            case "audio/pcma":
                return "alaw"
            case "audio/pcm":
                return None

    @property
    def rate(self) -> int:
        return WIRE_RATE if self.law is None else G711_RATE

    @property
    def sample_width(self) -> int:
        """Bytes per sample: any multiple of it is whole samples."""
        return SAMPLE_WIDTH if self.law is None else 1

    @property
    def bytes_per_second(self) -> int:
        return self.rate * self.sample_width

    def bytes_to_ms(self, nbytes: int) -> int:
        """Whole milliseconds in ``nbytes`` (rounded down)."""
        return nbytes // self.sample_width * 1000 // self.rate

    def duration_ms(self, nbytes: int) -> float:
        return nbytes / self.sample_width * 1000 / self.rate

    def ms_to_bytes(self, ms: int) -> int:
        """The bytes of ``ms`` milliseconds, whole samples (rounded down)."""
        return ms * self.rate // 1000 * self.sample_width


PCM = AudioFormat("audio/pcm")
PCMU = AudioFormat("audio/pcmu")
PCMA = AudioFormat("audio/pcma")

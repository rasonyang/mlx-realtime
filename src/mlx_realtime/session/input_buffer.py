"""The input audio buffer, held at the canonical rate on the session's input timeline.

The timeline counts canonical (16 kHz) samples since the session's first append, so a
millisecond position is ``sample // 16``. GA reports ``audio_start_ms`` and ``audio_end_ms`` on
this timeline: milliseconds since the start of all audio written to the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_realtime.audio.codec import InputDecoder
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples, samples_to_ms
from mlx_realtime.session.errors import SessionFatal


@dataclass(frozen=True, slots=True)
class CommittedAudio:
    pcm: bytes  # canonical PCM16
    start_ms: int
    end_ms: int

    @property
    def audio_ms(self) -> int:
        return self.end_ms - self.start_ms


class InputBuffer:
    def __init__(self, max_ms: int) -> None:
        self._decoder = InputDecoder()
        self._max_samples = ms_to_samples(max_ms, CANONICAL_RATE)
        self._pcm = bytearray()
        self._start = 0  # timeline sample of the first buffered sample

    @property
    def start_ms(self) -> int:
        return samples_to_ms(self._start, CANONICAL_RATE)

    @property
    def end_ms(self) -> int:
        return samples_to_ms(self._end, CANONICAL_RATE)

    @property
    def _end(self) -> int:
        return self._start + len(self._pcm) // SAMPLE_WIDTH

    def append(self, pcm: bytes) -> tuple[int, bytes]:
        """Decode client audio; return the timeline sample where the new audio starts, and it."""
        start = self._end
        canonical = self._decoder.decode(pcm)
        self._pcm += canonical
        if len(self._pcm) // SAMPLE_WIDTH > self._max_samples:
            raise SessionFatal(
                "input_audio_buffer_overflow", "The input audio buffer exceeded its maximum length."
            )
        return start, canonical

    def commit(self) -> CommittedAudio | None:
        """Commit everything buffered, the resampler's tail included; ``None`` if empty."""
        self._pcm += self._decoder.flush()
        if not self._pcm:
            return None
        audio = CommittedAudio(bytes(self._pcm), self.start_ms, self.end_ms)
        self._drop_before(self._end)
        return audio

    def commit_span(self, start_ms: int, end_ms: int) -> CommittedAudio:
        """Commit ``[start_ms, end_ms)``; earlier audio is dropped, later audio stays buffered."""
        first = max(ms_to_samples(start_ms, CANONICAL_RATE), self._start)
        last = max(min(ms_to_samples(end_ms, CANONICAL_RATE), self._end), first)
        audio = CommittedAudio(
            self._slice(first, last),
            samples_to_ms(first, CANONICAL_RATE),
            samples_to_ms(last, CANONICAL_RATE),
        )
        self._drop_before(last)
        return audio

    def clear(self) -> None:
        """Drop everything uncommitted, including audio still inside the resampler."""
        tail = self._decoder.flush()
        self._start = self._end + len(tail) // SAMPLE_WIDTH
        self._pcm.clear()

    def trim_before(self, ms: int) -> None:
        """Drop audio before ``ms`` (idle trimming in VAD modes)."""
        self._drop_before(min(ms_to_samples(ms, CANONICAL_RATE), self._end))

    def slice(self, start_ms: int, end_ms: int) -> bytes:
        return self._slice(
            ms_to_samples(start_ms, CANONICAL_RATE), ms_to_samples(end_ms, CANONICAL_RATE)
        )

    def _slice(self, first: int, last: int) -> bytes:
        first = min(max(first, self._start), self._end)
        last = min(max(last, first), self._end)
        return bytes(
            self._pcm[(first - self._start) * SAMPLE_WIDTH : (last - self._start) * SAMPLE_WIDTH]
        )

    def _drop_before(self, sample: int) -> None:
        if sample <= self._start:
            return
        del self._pcm[: (sample - self._start) * SAMPLE_WIDTH]
        self._start = sample

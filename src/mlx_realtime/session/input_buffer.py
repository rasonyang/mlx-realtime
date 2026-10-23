"""The input audio buffer, held at the canonical rate on the session's input timeline.

The timeline counts canonical (16 kHz) samples since the session's first append, so a
millisecond position is ``sample // 16``. GA reports ``audio_start_ms`` and ``audio_end_ms`` on
this timeline: milliseconds since the start of all audio written to the buffer.

With the echo canceller on (D17) the buffer keeps a second track: the canceller's output,
written over the same positions as it arrives (it lags the input). Smart Turn reads that track;
commits give the ASR the track ``asr_input`` names. Where the canceller has not run yet the
processed track holds the input as received.

Client audio arrives in the session's input format (24 kHz PCM, or G.711 at 8 kHz) and is held
at 16 kHz either way, so positions and milliseconds never depend on the format. A format change
flushes the old decoder's tail onto the timeline first.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_realtime.audio.codec import InputDecoder
from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples, samples_to_ms
from mlx_realtime.audio.resample import Quality
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
    def __init__(
        self,
        max_ms: int,
        *,
        processed: bool = False,
        asr_processed: bool = True,
        input_format: AudioFormat = PCM,
        g711_quality: Quality = "HQ",
    ) -> None:
        """``processed``: keep the echo canceller's track; ``asr_processed``: commits take it.
        ``g711_quality``: the 8 → 16 kHz resampling of G.711 input."""
        self._g711_quality: Quality = g711_quality
        self._decoder = InputDecoder(input_format, g711_quality)
        self._max_samples = ms_to_samples(max_ms, CANONICAL_RATE)
        self._pcm = bytearray()
        self._clean: bytearray | None = bytearray() if processed else None
        self._asr_processed = processed and asr_processed
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

    @property
    def input_format(self) -> AudioFormat:
        return self._decoder.audio_format

    def set_format(self, input_format: AudioFormat) -> tuple[int, bytes]:
        """Decode later audio as ``input_format``. The old decoder's tail goes onto the timeline
        first; returns where it starts, and it (empty when the format is unchanged)."""
        if input_format == self._decoder.audio_format:
            return self._end, b""
        tail = self._decoder.flush()
        self._decoder = InputDecoder(input_format, self._g711_quality)
        return self._add(tail)

    def append(self, audio: bytes) -> tuple[int, bytes]:
        """Decode client audio; return the timeline sample where the new audio starts, and it."""
        return self._add(self._decoder.decode(audio))

    def _add(self, canonical: bytes) -> tuple[int, bytes]:
        start = self._end
        self._pcm += canonical
        if self._clean is not None:
            self._clean += canonical
        if len(self._pcm) // SAMPLE_WIDTH > self._max_samples:
            raise SessionFatal(
                "input_audio_buffer_overflow", "The input audio buffer exceeded its maximum length."
            )
        return start, canonical

    def commit(self) -> CommittedAudio | None:
        """Commit everything buffered, the resampler's tail included; ``None`` if empty."""
        tail = self._decoder.flush()
        self._pcm += tail
        if self._clean is not None:
            self._clean += tail
        if not self._pcm:
            return None
        track = self._clean if self._asr_processed and self._clean is not None else self._pcm
        audio = CommittedAudio(bytes(track), self.start_ms, self.end_ms)
        self._drop_before(self._end)
        return audio

    def apply_processed(self, start_sample: int, pcm: bytes) -> None:
        """The echo canceller's output for ``[start_sample, ...)``: written over the processed
        track where those positions are still buffered."""
        clean = self._clean
        if clean is None:
            return
        first = max(start_sample, self._start)
        last = min(start_sample + len(pcm) // SAMPLE_WIDTH, self._end)
        if last <= first:
            return
        offset = (first - start_sample) * SAMPLE_WIDTH
        clean[(first - self._start) * SAMPLE_WIDTH : (last - self._start) * SAMPLE_WIDTH] = pcm[
            offset : offset + (last - first) * SAMPLE_WIDTH
        ]

    def commit_span(self, start_ms: int, end_ms: int) -> CommittedAudio:
        """Commit ``[start_ms, end_ms)``; earlier audio is dropped, later audio stays buffered."""
        first = max(ms_to_samples(start_ms, CANONICAL_RATE), self._start)
        last = max(min(ms_to_samples(end_ms, CANONICAL_RATE), self._end), first)
        audio = CommittedAudio(
            self._slice(first, last, processed=self._asr_processed),
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
        if self._clean is not None:
            self._clean.clear()

    def trim_before(self, ms: int) -> None:
        """Drop audio before ``ms`` (idle trimming in VAD modes)."""
        self._drop_before(min(ms_to_samples(ms, CANONICAL_RATE), self._end))

    def slice(self, start_ms: int, end_ms: int) -> bytes:
        """Audio for the detectors (Smart Turn): the processed track when there is one."""
        return self._slice(
            ms_to_samples(start_ms, CANONICAL_RATE),
            ms_to_samples(end_ms, CANONICAL_RATE),
            processed=True,
        )

    def _slice(self, first: int, last: int, *, processed: bool = False) -> bytes:
        track = self._clean if processed and self._clean is not None else self._pcm
        first = min(max(first, self._start), self._end)
        last = min(max(last, first), self._end)
        return bytes(
            track[(first - self._start) * SAMPLE_WIDTH : (last - self._start) * SAMPLE_WIDTH]
        )

    def _drop_before(self, sample: int) -> None:
        if sample <= self._start:
            return
        del self._pcm[: (sample - self._start) * SAMPLE_WIDTH]
        if self._clean is not None:
            del self._clean[: (sample - self._start) * SAMPLE_WIDTH]
        self._start = sample

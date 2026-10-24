"""The input audio buffer, held at the canonical rate on the session's input timeline.

The timeline counts canonical (16 kHz) samples since the session's first append, so a
millisecond position is ``sample // 16``. GA reports ``audio_start_ms`` and ``audio_end_ms`` on
this timeline: milliseconds since the start of all audio written to the buffer.

With echo cancellation on (D17) the buffer keeps more tracks over the same positions, each
written as the cancellers' output arrives (it lags the input): AEC3's output, which Smart Turn
reads, and with ``asr_input: speex`` the ASR path's (G7 dual path). Commits give the ASR the
track ``asr_input`` names: ``speex``, ``aec3`` or ``raw`` (the input as received). Where a
canceller has not run yet its track holds the input as received.

Client audio arrives in the session's input format (24 kHz PCM, or G.711 at 8 kHz) and is held
at 16 kHz either way, so positions and milliseconds never depend on the format. A format change
flushes the old decoder's tail onto the timeline first.

A transcription session's buffer is rolling (``Rolling``): the client sends audio all the time,
silence included, and commits only after speech. Until a frame is voiced the buffer keeps just
the last ``keep_ms`` (a held call costs no memory, and the ASR never gets minutes of silence in
front of an utterance); a buffer over its maximum drops its oldest audio instead of ending the
session. A commit of a buffer with no voiced frame is reported as such (``voiced``), so the
session can answer it without asking the ASR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mlx_realtime.audio.codec import InputDecoder
from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples, samples_to_ms
from mlx_realtime.audio.resample import Quality
from mlx_realtime.config import AsrInput
from mlx_realtime.session.errors import SessionFatal

logger = logging.getLogger(__name__)

_FRAME_SAMPLES = ms_to_samples(20, CANONICAL_RATE)


@dataclass(frozen=True, slots=True)
class CommittedAudio:
    pcm: bytes  # canonical PCM16
    start_ms: int
    end_ms: int
    voiced: bool = True  # some 20 ms frame reached the voiced level (rolling buffers only)

    @property
    def audio_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class Rolling:
    """A rolling buffer (transcription sessions): ``keep_ms`` of audio kept while nothing is
    voiced; a 20 ms frame is voiced from ``voiced_rms`` (RMS of its 16-bit samples)."""

    keep_ms: int
    voiced_rms: int


class InputBuffer:
    def __init__(
        self,
        max_ms: int,
        *,
        processed: bool = False,
        asr_input: AsrInput = "raw",
        input_format: AudioFormat = PCM,
        g711_quality: Quality = "HQ",
        rolling: Rolling | None = None,
    ) -> None:
        """``processed``: the cancellers run, so keep AEC3's track (and the ASR path's when
        ``asr_input`` is ``speex``); ``asr_input``: the track commits take (``raw`` without
        cancellers). ``g711_quality``: the 8 → 16 kHz resampling of G.711 input. ``rolling``:
        see the module docstring; ``None`` makes a buffer over its maximum fatal."""
        self._g711_quality: Quality = g711_quality
        self._decoder = InputDecoder(input_format, g711_quality)
        self._max_samples = ms_to_samples(max_ms, CANONICAL_RATE)
        self._pcm = bytearray()
        self._clean: bytearray | None = bytearray() if processed else None
        self._asr: bytearray | None = bytearray() if processed and asr_input == "speex" else None
        self._asr_input: AsrInput = asr_input if processed else "raw"
        self._start = 0  # timeline sample of the first buffered sample
        self._rolling = rolling
        self._voiced = False  # a voiced frame is buffered (rolling buffers)
        self._dropped_ms = 0  # audio dropped over the maximum since the last commit or clear

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
        for track in self._tracks():
            track += canonical
        if self._rolling is not None:
            self._roll(canonical)
        elif len(self._pcm) // SAMPLE_WIDTH > self._max_samples:
            raise SessionFatal(
                "input_audio_buffer_overflow", "The input audio buffer exceeded its maximum length."
            )
        return start, canonical

    def _roll(self, canonical: bytes) -> None:
        rolling = self._rolling
        if rolling is None:
            return
        self._voiced = self._voiced or _is_voiced(canonical, rolling.voiced_rms)
        keep = self._max_samples
        if not self._voiced:
            keep = min(keep, ms_to_samples(rolling.keep_ms, CANONICAL_RATE))
        excess = len(self._pcm) // SAMPLE_WIDTH - keep
        if excess <= 0:
            return
        if self._voiced:  # speech that does not fit: the oldest audio goes (never the session)
            if not self._dropped_ms:
                logger.warning(
                    "the input audio buffer is over its maximum; its oldest audio is dropped",
                    extra={"max_ms": samples_to_ms(self._max_samples, CANONICAL_RATE)},
                )
            self._dropped_ms += samples_to_ms(excess, CANONICAL_RATE)
        self._drop_before(self._start + excess)

    def commit(self, *, empty_ok: bool = False) -> CommittedAudio | None:
        """Commit everything buffered, the resampler's tail included; ``None`` if empty (unless
        ``empty_ok``: then an empty commit)."""
        tail = self._decoder.flush()
        self._pcm += tail
        for track in self._tracks():
            track += tail
        if self._rolling is not None:
            self._roll(tail)
        if not self._pcm and not empty_ok:
            return None
        audio = CommittedAudio(
            bytes(self._asr_track()),
            self.start_ms,
            self.end_ms,
            voiced=self._rolling is None or self._voiced,
        )
        self._drop_before(self._end)
        self._reset_roll()
        return audio

    def apply_processed(self, start_sample: int, pcm: bytes) -> None:
        """AEC3's output for ``[start_sample, ...)``: written over its track where those
        positions are still buffered."""
        self._apply(self._clean, start_sample, pcm)

    def apply_asr(self, start_sample: int, pcm: bytes) -> None:
        """The ASR path's canceller output for ``[start_sample, ...)``, likewise."""
        self._apply(self._asr, start_sample, pcm)

    def _apply(self, track: bytearray | None, start_sample: int, pcm: bytes) -> None:
        if track is None:
            return
        first = max(start_sample, self._start)
        last = min(start_sample + len(pcm) // SAMPLE_WIDTH, self._end)
        if last <= first:
            return
        offset = (first - start_sample) * SAMPLE_WIDTH
        track[(first - self._start) * SAMPLE_WIDTH : (last - self._start) * SAMPLE_WIDTH] = pcm[
            offset : offset + (last - first) * SAMPLE_WIDTH
        ]

    def commit_span(self, start_ms: int, end_ms: int) -> CommittedAudio:
        """Commit ``[start_ms, end_ms)``; earlier audio is dropped, later audio stays buffered."""
        first = max(ms_to_samples(start_ms, CANONICAL_RATE), self._start)
        last = max(min(ms_to_samples(end_ms, CANONICAL_RATE), self._end), first)
        audio = CommittedAudio(
            self._slice_track(self._asr_track(), first, last),
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
        for track in self._tracks():
            track.clear()
        self._reset_roll()

    def _reset_roll(self) -> None:
        if self._dropped_ms:
            logger.warning(
                "input audio dropped over the buffer maximum",
                extra={"dropped_ms": self._dropped_ms},
            )
        self._voiced = False
        self._dropped_ms = 0

    def trim_before(self, ms: int) -> None:
        """Drop audio before ``ms`` (idle trimming in VAD modes)."""
        self._drop_before(min(ms_to_samples(ms, CANONICAL_RATE), self._end))

    def slice(self, start_ms: int, end_ms: int) -> bytes:
        """Audio for the detectors (Smart Turn): AEC3's track when there is one."""
        return self._slice_track(
            self._pcm if self._clean is None else self._clean,
            ms_to_samples(start_ms, CANONICAL_RATE),
            ms_to_samples(end_ms, CANONICAL_RATE),
        )

    def _slice_track(self, track: bytearray, first: int, last: int) -> bytes:
        first = min(max(first, self._start), self._end)
        last = min(max(last, first), self._end)
        return bytes(
            track[(first - self._start) * SAMPLE_WIDTH : (last - self._start) * SAMPLE_WIDTH]
        )

    def _tracks(self) -> list[bytearray]:
        """The cancellers' tracks, which follow the input's positions."""
        return [track for track in (self._clean, self._asr) if track is not None]

    def _asr_track(self) -> bytearray:
        match self._asr_input:
            case "speex":
                return self._pcm if self._asr is None else self._asr
            case "aec3":
                return self._pcm if self._clean is None else self._clean
            case "raw":
                return self._pcm

    def _drop_before(self, sample: int) -> None:
        if sample <= self._start:
            return
        del self._pcm[: (sample - self._start) * SAMPLE_WIDTH]
        for track in self._tracks():
            del track[: (sample - self._start) * SAMPLE_WIDTH]
        self._start = sample


def _is_voiced(pcm: bytes, rms: int) -> bool:
    """Whether any 20 ms frame of ``pcm`` (a shorter tail counts as one) has an RMS of ``rms``."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    whole = len(samples) // _FRAME_SAMPLES * _FRAME_SAMPLES
    if whole:
        frames = samples[:whole].reshape(-1, _FRAME_SAMPLES)
        if bool((np.sqrt(np.mean(frames * frames, axis=1)) >= rms).any()):
            return True
    tail = samples[whole:]
    return bool(tail.size) and float(np.sqrt(np.mean(tail * tail))) >= rms

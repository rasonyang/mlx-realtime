"""The VAD feeder: one task per session that runs speech detection (ARCHITECTURE §4.2, §6.2).

The actor hands it canonical audio as the client appends it. With the echo canceller on
(ARCHITECTURE §4.3, D17), the audio first goes through it (``EchoStage``: 10 ms frames in lock
step with their reference, on the canceller executor), and everything after sees the
canceller's output. The feeder then cuts 512-sample frames, asks the speech detector for each
frame's probability (Silero, on its executor), runs the speech gate (with the client's modelled
playback state for each frame: the barge-in rules), and posts one ``VadUpdate`` per processed
chunk, carrying the canceller's output for the input buffer. Frames go to each model one at a
time, in order, so per-session model state needs no locking.

Audio and reconfiguration share one FIFO queue, so a reset is ordered with the audio around
it. Every update carries the epoch its audio was queued under; the actor ignores updates from
before its latest commit, clear or settings change. Audio the canceller still holds back when
a reset comes is processed first, under the old epoch.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples, samples_to_ms
from mlx_realtime.session.clock import Clock
from mlx_realtime.session.echo import EchoReference, EchoStage
from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.signals import VadUpdate
from mlx_realtime.session.turn import Framer
from mlx_realtime.vad.base import FRAME_MS, FRAME_SAMPLES, SpeechDetector
from mlx_realtime.vad.gate import GateEvent, GateParams, SpeechGate

_MIN_WAIT_S = 0.001  # never spin when held audio is due


@dataclass(frozen=True, slots=True)
class _Audio:
    start_sample: int  # on the session's input timeline
    pcm: bytes  # canonical PCM16


@dataclass(frozen=True, slots=True)
class _Configure:
    epoch: int
    params: GateParams | None  # None: detection off


class VadFeeder:
    def __init__(
        self,
        detector: SpeechDetector,
        post: Callable[[VadUpdate], None],
        *,
        max_queued_ms: int,
        reference: EchoReference | None = None,
        echo: EchoStage | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._detector = detector
        self._post = post
        self._max_queued = ms_to_samples(max_queued_ms, CANONICAL_RATE)
        self._queue: deque[_Audio | _Configure] = deque()
        self._queued = 0  # samples waiting in the queue
        self._ready = asyncio.Event()
        self._epoch = 0
        self._gate: SpeechGate | None = None
        self._framer = Framer()
        self._reference = reference
        self._echo = echo
        self._clock = clock
        if reference is not None:
            reference.on_change = self._ready.set  # held audio may have become final

    def configure(self, epoch: int, params: GateParams | None) -> None:
        """From the next audio on: a new epoch and a fresh gate (``None`` stops detection)."""
        self._put(_Configure(epoch, params))

    def push(self, start_sample: int, pcm: bytes) -> None:
        samples = len(pcm) // SAMPLE_WIDTH
        if self._queued + samples > self._max_queued:
            # Never skip audio: turns would be cut on a timeline the detector did not see.
            raise SessionFatal("input_queue_overflow", "Speech detection fell behind the audio.")
        self._queued += samples
        self._put(_Audio(start_sample, pcm))

    async def run(self) -> None:
        while True:
            self._ready.clear()  # a wakeup while this pass runs is kept for the next one
            while self._queue:
                match self._queue.popleft():
                    case _Configure(epoch=epoch, params=params):
                        if self._echo is not None:  # held audio belongs to the old epoch
                            await self._detect(await self._echo.release(force=True))
                        self._epoch = epoch
                        self._gate = None if params is None else SpeechGate(params)
                        self._framer = Framer()
                    case _Audio() as audio:
                        self._queued -= len(audio.pcm) // SAMPLE_WIDTH
                        if self._echo is None:
                            await self._detect([(audio.start_sample, audio.pcm)], processed=False)
                        else:
                            await self._detect(await self._echo.feed(audio.start_sample, audio.pcm))
            if self._echo is not None:
                await self._detect(await self._echo.release())
            if not self._queue:
                await self._idle()

    async def _idle(self) -> None:
        """Wait for audio, a reconfiguration, a change of the echo reference, or the time when
        held audio becomes final."""
        timeout: float | None = None
        echo, clock = self._echo, self._clock
        if echo is not None and clock is not None and (due := echo.wake_at()) is not None:
            timeout = max(due - clock.now(), _MIN_WAIT_S)
        try:
            async with asyncio.timeout(timeout):
                await self._ready.wait()
        except TimeoutError:
            pass

    async def _detect(self, chunks: list[tuple[int, bytes]], *, processed: bool = True) -> None:
        for start, pcm in chunks:
            await self._detect_chunk(start, pcm, processed=processed)

    async def _detect_chunk(self, start: int, pcm: bytes, *, processed: bool) -> None:
        gate = self._gate
        if gate is None:
            return
        frames = self._framer.push(start, pcm)
        if not frames and not processed:
            return
        events: list[GateEvent] = []
        for frame_start, frame in frames:
            start_ms = samples_to_ms(frame_start, CANONICAL_RATE)
            probability = await self._detector.speech_probability(
                frame, start_ms=start_ms, in_speech=gate.in_speech
            )
            playing, guard_end_ms = False, None
            if self._reference is not None:
                playback = self._reference.playback(frame_start + FRAME_SAMPLES)
                playing = playback.playing
                if playback.guard_end is not None:
                    guard_end_ms = samples_to_ms(playback.guard_end, CANONICAL_RATE)
            events += gate.on_frame(
                start_ms,
                start_ms + FRAME_MS,
                probability,
                playing=playing,
                guard_end_ms=guard_end_ms,
            )
        self._post(
            VadUpdate(
                self._epoch,
                tuple(events),
                gate.keep_from_ms(),
                processed_start=start if processed else 0,
                processed=pcm if processed else b"",
            )
        )

    def _put(self, item: _Audio | _Configure) -> None:
        self._queue.append(item)
        self._ready.set()

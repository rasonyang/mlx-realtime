"""The VAD feeder: one task per session that runs speech detection (ARCHITECTURE §4.2, §6.2).

The actor hands it canonical audio as the client appends it. The feeder cuts 512-sample frames,
asks the speech detector for each frame's probability (Silero, on its executor), runs the
speech gate, and posts one ``VadUpdate`` per processed chunk. Frames go to the detector one at a
time, in order, so per-session model state needs no locking.

Audio and reconfiguration share one FIFO queue, so a reset is ordered with the audio around
it. Every update carries the epoch its audio was queued under; the actor ignores updates from
before its latest commit, clear or settings change.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, ms_to_samples, samples_to_ms
from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.signals import VadUpdate
from mlx_realtime.session.turn import Framer
from mlx_realtime.vad.base import FRAME_MS, SpeechDetector
from mlx_realtime.vad.gate import GateEvent, GateParams, SpeechGate


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
            while not self._queue:
                self._ready.clear()
                await self._ready.wait()
            match self._queue.popleft():
                case _Configure(epoch=epoch, params=params):
                    self._epoch = epoch
                    self._gate = None if params is None else SpeechGate(params)
                    self._framer = Framer()
                case _Audio() as audio:
                    self._queued -= len(audio.pcm) // SAMPLE_WIDTH
                    await self._detect(audio)

    async def _detect(self, audio: _Audio) -> None:
        gate = self._gate
        if gate is None:
            return
        frames = self._framer.push(audio.start_sample, audio.pcm)
        if not frames:
            return
        events: list[GateEvent] = []
        for frame_start, frame in frames:
            start_ms = samples_to_ms(frame_start, CANONICAL_RATE)
            probability = await self._detector.speech_probability(
                frame, start_ms=start_ms, in_speech=gate.in_speech
            )
            events += gate.on_frame(start_ms, start_ms + FRAME_MS, probability)
        self._post(VadUpdate(self._epoch, tuple(events), gate.keep_from_ms()))

    def _put(self, item: _Audio | _Configure) -> None:
        self._queue.append(item)
        self._ready.set()

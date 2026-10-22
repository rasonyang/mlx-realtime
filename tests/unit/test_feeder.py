"""The VAD feeder: framing across chunk boundaries, one detector call at a time with the gate's
state, epochs, reconfiguration and the backlog bound."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.feeder import VadFeeder
from mlx_realtime.session.signals import VadUpdate
from mlx_realtime.vad.base import FRAME_SAMPLES
from mlx_realtime.vad.gate import GateParams, SpeechEnd, SpeechStart
from mlx_realtime.vad.scripted import ScriptedSpeech

PARAMS = GateParams(threshold=0.5, min_speech_ms=96, silence_duration_ms=200, max_silence_ms=None)


class Recorder:
    """A scripted detector that records every call it gets."""

    def __init__(self, speech: Sequence[tuple[int, int]] = ()) -> None:
        self.calls: list[tuple[int, int, bool]] = []
        self.busy = 0
        self._script = ScriptedSpeech(speech)

    async def speech_probability(self, frame: bytes, *, start_ms: int, in_speech: bool) -> float:
        self.busy += 1
        assert self.busy == 1, "one call at a time"
        await asyncio.sleep(0)
        self.calls.append((start_ms, len(frame), in_speech))
        self.busy -= 1
        return self._script.probability_at(start_ms)


@asynccontextmanager
async def feeding(
    detector: Recorder, *, max_queued_ms: int = 10_000
) -> AsyncIterator[tuple[VadFeeder, list[VadUpdate]]]:
    updates: list[VadUpdate] = []
    feeder = VadFeeder(detector, updates.append, max_queued_ms=max_queued_ms)
    feeder.configure(0, PARAMS)
    task = asyncio.create_task(feeder.run())
    try:
        yield feeder, updates
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def settle(detector: Recorder, frames: int) -> None:
    """Wait until the detector has seen ``frames`` frames, and a little longer."""
    async with asyncio.timeout(2):
        while len(detector.calls) < frames:  # noqa: ASYNC110 (a counter)
            await asyncio.sleep(0)
    for _ in range(20):
        await asyncio.sleep(0)


def audio(ms: int) -> bytes:
    return bytes(32 * ms)


async def test_frames_do_not_depend_on_how_the_audio_was_chunked() -> None:
    results: list[list[tuple[int, int, bool]]] = []
    for sizes in ([2000], [7, 993, 1000], [333] * 6 + [2]):
        detector = Recorder([(320, 1280)])
        async with feeding(detector) as (feeder, _):
            position = 0
            for size in sizes:
                feeder.push(position * 16, audio(size))
                position += size
            await settle(detector, 2000 * 16 // FRAME_SAMPLES)
        results.append(detector.calls)
    assert results[0] == results[1] == results[2]
    assert [start for start, _, _ in results[0]][:3] == [0, 32, 64]
    assert all(length == 2 * FRAME_SAMPLES for _, length, _ in results[0])


async def test_updates_carry_events_and_what_to_keep() -> None:
    detector = Recorder([(320, 1280)])
    async with feeding(detector) as (feeder, updates):
        feeder.push(0, audio(10))  # no whole frame yet: no update
        await asyncio.sleep(0.01)
        assert updates == []
        feeder.push(160, audio(1990))
        await settle(detector, 2000 * 16 // FRAME_SAMPLES)
    events = [event for update in updates for event in update.events]
    assert events == [SpeechStart(320), SpeechEnd(1280, 1504)]
    assert updates[-1].keep_from_ms == 1984 - 32
    # The gate's state before each frame reaches the detector: in speech from the candidate on.
    in_speech = {start: flag for start, _, flag in detector.calls}
    assert (in_speech[320], in_speech[352], in_speech[1504]) == (False, True, False)


async def test_a_new_epoch_restarts_detection_from_the_next_audio() -> None:
    detector = Recorder([(0, 2000)])
    async with feeding(detector) as (feeder, updates):
        feeder.push(0, audio(200))
        feeder.configure(7, PARAMS)
        feeder.push(3200, audio(200))  # after a commit: framing restarts at 200 ms
        await settle(detector, 2 * (200 * 16 // FRAME_SAMPLES))
    assert [update.epoch for update in updates] == [0, 7]
    assert [event for update in updates for event in update.events] == [
        SpeechStart(0),
        SpeechStart(200),  # a fresh gate: the onset is confirmed again, from 200 ms
    ]


async def test_detection_off_ignores_audio() -> None:
    detector = Recorder([(0, 2000)])
    async with feeding(detector) as (feeder, updates):
        feeder.configure(1, None)
        feeder.push(0, audio(500))
        await asyncio.sleep(0.01)
    assert (updates, detector.calls) == ([], [])


async def test_a_backlog_beyond_the_bound_is_fatal() -> None:
    detector = Recorder()
    async with feeding(detector, max_queued_ms=100) as (feeder, _):
        feeder.push(0, audio(60))
        with pytest.raises(SessionFatal) as raised:
            feeder.push(960, audio(60))
    assert raised.value.code == "input_queue_overflow"

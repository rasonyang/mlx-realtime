"""Multi-session TTS fairness in the pipeline (M6): the audio reserve keeps deltas flowing while
the next segment waits in the TTS queue, and every TTS request carries its session's deadline."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncGenerator

import pytest

from mlx_realtime.config import AudioReserveConfig, SegmenterConfig, SilenceTrimConfig
from mlx_realtime.providers.base import Lang, LLMDone, PCMChunk, SpeechTiming, TextDelta
from mlx_realtime.providers.mock import EchoLLM, SineTTS
from mlx_realtime.session.observer import NullObserver
from mlx_realtime.session.pipeline import PipelineContext, PipelineRequest, run_pipeline
from mlx_realtime.session.signals import (
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    SpeechFinished,
)
from mlx_realtime.text.normalize import NormalizerOptions
from tests.support import ManualClock

SEGMENTER = SegmenterConfig(first_min_chars=2, first_max_wait_ms=0, min_chars=2, max_chars=80)
FIRST = "一二三四五六七八九十" * 2 + "一二三四五六七八九。"  # 30 characters: 3 s
SECOND = "好的。"


class Recorder:
    """Posted signals with the virtual time each one was posted at."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.posted: list[tuple[float, PipelineSignal]] = []

    def post(self, signal: PipelineSignal) -> None:
        self.posted.append((self.clock.now(), signal))

    def audio(self) -> list[tuple[float, SegmentAudio]]:
        return [(at, s) for at, s in self.posted if isinstance(s, SegmentAudio)]


class Observed(NullObserver):
    def __init__(self) -> None:
        self.fallbacks = 0
        self.errors: list[tuple[str, str]] = []

    def tts_fallback(self) -> None:
        self.fallbacks += 1

    def provider_error(self, backend: str, kind: str) -> None:
        self.errors.append((backend, kind))


def make_context(
    llm: EchoLLM,
    tts: SineTTS,
    recorder: Recorder,
    *,
    reserve: AudioReserveConfig | None,
    grace_s: float = 0.0,
    observer: NullObserver | None = None,
) -> PipelineContext:
    async def room() -> None:
        await asyncio.sleep(0)

    return PipelineContext(
        llm=llm,
        tts=tts,
        segmenter=SEGMENTER,
        normalizer=NormalizerOptions(enabled=False),
        max_delta_ms=100,
        max_lead_ms=10_000,
        max_audio_ms=120_000,
        post=recorder.post,
        wait_for_room=room,
        clock=recorder.clock,
        trim_silence=SilenceTrimConfig(enabled=False),
        reserve=reserve,
        first_segment_grace_s=grace_s,
        observer=observer or NullObserver(),
    )


REQUEST = PipelineRequest(
    generation=1,
    messages=(),
    tools=(),
    tool_choice="auto",
    max_output_tokens=None,
    modality="audio",
    voice="vivian",
)


async def settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


async def test_the_reserve_holds_back_audio_and_keeps_deltas_flowing_while_the_tts_is_queued() -> (
    None
):
    clock = ManualClock()
    recorder = Recorder(clock)
    llm, tts = EchoLLM(), SineTTS(ms_per_char=100, chunk_ms=100)
    llm.script(TextDelta(FIRST), TextDelta(SECOND), LLMDone("stop", None))
    tts.queue.hold(after=1)  # the second segment waits in the TTS queue
    reserve = AudioReserveConfig(hold_ms=1000, keepalive_ms=500, min_lead_ms=400)
    task = asyncio.create_task(
        run_pipeline(REQUEST, make_context(llm, tts, recorder, reserve=reserve))
    )
    await settle()
    start = clock.now()
    assert len(recorder.audio()) == 20  # 3 s synthesized: all but the last 1 s went out
    for _ in range(60):  # 3 s of virtual time
        clock.advance(0.05)
        await settle()
    sent = recorder.audio()
    assert len(sent) == 30  # the reserve ran out only as the client's buffer did
    times = [at - start for at, _ in sent[19:]]
    step = 0.05 + 1e-6  # the virtual clock moves in steps of 50 ms
    assert max(b - a for a, b in itertools.pairwise(times)) <= 0.5 + step
    assert 0.5 - 1e-6 <= times[1] <= 0.5 + step  # a keepalive delta: the buffer still held 1.9 s
    # Played from the first delta, the client never ran dry before the reserve did.
    for index, (at, _) in enumerate(sent):
        assert (index + 1) * 0.1 - (at - start) >= -step
    tts.queue.release()
    await settle()
    clock.advance(0.05)
    await asyncio.wait_for(task, 1)
    kinds = [(type(s).__name__, getattr(s, "index", None)) for _, s in recorder.posted]
    second = kinds.index(("SegmentStarted", 1))
    assert all(kind != "SegmentAudio" or index == 1 for kind, index in kinds[second:-1])
    assert kinds[-1] == ("SpeechFinished", None)


async def test_without_more_segments_to_come_nothing_is_held() -> None:
    clock = ManualClock()
    recorder = Recorder(clock)
    llm, tts = EchoLLM(), SineTTS(ms_per_char=100, chunk_ms=100)
    llm.script(TextDelta(FIRST), LLMDone("stop", None))
    reserve = AudioReserveConfig(hold_ms=1000, keepalive_ms=500, min_lead_ms=400)
    await asyncio.wait_for(
        run_pipeline(REQUEST, make_context(llm, tts, recorder, reserve=reserve)), 1
    )
    assert len(recorder.audio()) == 30
    assert {at for at, _ in recorder.posted} == {clock.now()}  # all of it at once
    assert isinstance(recorder.posted[-1][1], SpeechFinished)


async def test_a_segment_without_audio_keeps_its_transcript_in_order_with_the_reserve() -> None:
    clock = ManualClock()
    recorder = Recorder(clock)
    llm, tts = EchoLLM(), SineTTS(ms_per_char=100, chunk_ms=100)
    llm.script(TextDelta("你好。"), TextDelta("\n……\n"), LLMDone("stop", None))
    reserve = AudioReserveConfig(hold_ms=5000, keepalive_ms=500, min_lead_ms=0)
    await asyncio.wait_for(
        run_pipeline(REQUEST, make_context(llm, tts, recorder, reserve=reserve)), 1
    )
    order = [
        (type(s).__name__, s.index)
        for _, s in recorder.posted
        if isinstance(s, SegmentStarted | SegmentAudio)
    ]
    assert order[0] == ("SegmentStarted", 0)
    assert order[-1] == ("SegmentStarted", 1)  # after every delta of the segment before it
    assert len(order) == 2 + 3
    assert isinstance(recorder.posted[-1][1], SpeechFinished)


class DeadlineTTS(SineTTS):
    """Records the deadline and slack each request carried."""

    def __init__(self) -> None:
        super().__init__(ms_per_char=100, chunk_ms=100)
        self.timings: list[tuple[float | None, float]] = []

    def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        assert timing is not None
        self.timings.append((timing.deadline, timing.slack_s))
        if text == SECOND:
            timing.fallback = True
            timing.primary_error = "timeout"
        return super().stream(text, voice, lang, speed=speed, timing=timing)


async def test_each_request_carries_when_its_session_runs_out_of_audio() -> None:
    clock = ManualClock()
    recorder = Recorder(clock)
    llm, tts = EchoLLM(), DeadlineTTS()
    llm.script(TextDelta(FIRST), TextDelta(SECOND), LLMDone("stop", None))
    observer = Observed()
    start = clock.now()
    await asyncio.wait_for(
        run_pipeline(
            REQUEST,
            make_context(llm, tts, recorder, reserve=None, grace_s=0.25, observer=observer),
        ),
        1,
    )
    (first_deadline, first_slack), (second_deadline, second_slack) = tts.timings
    assert first_deadline == pytest.approx(start + 0.25)  # no audio yet: the start, plus grace
    assert first_slack == pytest.approx(0.25)
    assert second_deadline == pytest.approx(start + 3.0)  # the 3 s already synthesized
    assert second_slack == pytest.approx(3.0)
    assert observer.fallbacks == 1
    assert observer.errors == [("tts", "timeout")]

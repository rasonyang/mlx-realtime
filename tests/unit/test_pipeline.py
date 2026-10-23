"""One response's pipeline, driven directly with mock providers and a virtual clock: the LLM or a
verbatim line (D16), segments to the TTS, the normalizer, the TTS stage draining every stream
while the sender paces (D6), the timeline marks, the audio cap and failures."""

from __future__ import annotations

import asyncio

import pytest

from mlx_realtime.config import SegmenterConfig, SilenceTrimConfig
from mlx_realtime.providers.base import LLMDone, ProviderError, TextDelta, Usage
from mlx_realtime.providers.mock import EchoLLM, SineTTS
from mlx_realtime.session.pipeline import PipelineContext, PipelineRequest, run_pipeline
from mlx_realtime.session.signals import (
    LlmFinished,
    LlmText,
    PipelineFailed,
    PipelineMark,
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    SpeechFinished,
)
from mlx_realtime.text.normalize import NormalizerOptions

SEGMENTER = SegmenterConfig(first_min_chars=4, first_max_wait_ms=400, min_chars=8, max_chars=80)


class StepClock:
    """Virtual time. While held, a sleeper waits for ``run_free``; after it, sleeps pass at once."""

    def __init__(self, *, held: bool = False) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []
        self.sleeping = asyncio.Event()
        self._free = asyncio.Event()
        if not held:
            self._free.set()

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.sleeping.set()
        await self._free.wait()
        self.t += seconds

    def run_free(self) -> None:
        self._free.set()


def context(
    llm: EchoLLM,
    tts: SineTTS,
    posted: list[PipelineSignal],
    clock: StepClock,
    *,
    max_lead_ms: int = 10_000,
    max_audio_ms: int = 120_000,
) -> PipelineContext:
    async def room() -> None:
        await asyncio.sleep(0)

    return PipelineContext(
        llm=llm,
        tts=tts,
        segmenter=SEGMENTER,
        normalizer=NormalizerOptions(),
        max_delta_ms=100,
        max_lead_ms=max_lead_ms,
        max_audio_ms=max_audio_ms,
        post=posted.append,
        wait_for_room=room,
        clock=clock,
        trim_silence=SilenceTrimConfig(),
    )


def request(*, verbatim: str | None = None, modality: str = "audio") -> PipelineRequest:
    return PipelineRequest(
        generation=7,
        messages=(),
        tools=(),
        tool_choice="auto",
        max_output_tokens=None,
        modality="audio" if modality == "audio" else "text",
        voice="vivian",
        verbatim=verbatim,
    )


def kinds(posted: list[PipelineSignal]) -> list[str]:
    return [type(signal).__name__ for signal in posted if not isinstance(signal, PipelineMark)]


def audio_of(posted: list[PipelineSignal], index: int | None = None) -> int:
    return sum(
        len(signal.pcm)
        for signal in posted
        if isinstance(signal, SegmentAudio) and index in (None, signal.index)
    )


async def test_each_segments_transcript_precedes_its_audio() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    llm.script(
        TextDelta("你好。"), TextDelta("今天天气很好，我们出去走走吧。"), LLMDone("stop", None)
    )
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    order = [
        (type(signal).__name__, getattr(signal, "index", None))
        for signal in posted
        if isinstance(signal, SegmentStarted | SegmentAudio | SpeechFinished)
    ]
    assert order[0] == ("SegmentStarted", 0)
    first_of_second = order.index(("SegmentStarted", 1))
    assert all(index == 0 for _, index in order[1:first_of_second])
    assert all(
        kind == "SegmentAudio" and index == 1 for kind, index in order[first_of_second + 1 : -1]
    )
    assert order[-1] == ("SpeechFinished", None)
    assert [r.text for r in tts.requests] == ["你好。", "今天天气很好，我们出去走走吧。"]
    assert audio_of(posted, 0) == tts.samples_for("你好。") * 2
    assert {r.lang for r in tts.requests} == {"zh"}


async def test_the_tts_drains_every_stream_while_the_sender_waits_at_the_lead_cap() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=200, chunk_ms=100)
    llm.script(
        TextDelta("你好。"), TextDelta("今天天气很好，我们出去走走吧。"), LLMDone("stop", None)
    )
    posted: list[PipelineSignal] = []
    clock = StepClock(held=True)
    task = asyncio.create_task(
        run_pipeline(request(), context(llm, tts, posted, clock, max_lead_ms=1000))
    )
    await clock.sleeping.wait()  # the sender is at the cap: 1 s of audio sent at once
    async with asyncio.timeout(2):
        await tts.streams.idle()
        while len(tts.requests) < 2:  # noqa: ASYNC110 (the TTS stage moves on by itself)
            await asyncio.sleep(0.001)
        await tts.streams.idle()
    assert audio_of(posted) == 1000 * 48  # 1 s at 24 kHz s16le: the lead cap
    assert tts.streams.opened == tts.streams.closed == 2  # both requests generated and closed
    clock.run_free()
    await task
    total = sum(tts.samples_for(request.text) for request in tts.requests) * 2
    assert total == (3 + 15) * 200 * 48  # 18 characters, punctuation included, 200 ms each
    assert audio_of(posted) == total
    assert sum(clock.sleeps) == pytest.approx((total / 48 - 1000) / 1000)  # real time past the cap
    assert kinds(posted)[-1] == "SpeechFinished"


async def test_a_verbatim_line_skips_the_llm() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    posted: list[PipelineSignal] = []
    line = "Thanks for calling NovaNet. Have a good day."
    await run_pipeline(request(verbatim=line), context(llm, tts, posted, StepClock()))
    assert llm.requests == []
    texts = [signal for signal in posted if isinstance(signal, LlmText)]
    assert [signal.text for signal in texts] == [line]  # the whole line is the answer
    (finished,) = (signal for signal in posted if isinstance(signal, LlmFinished))
    assert finished.done == LLMDone("stop", None)  # no usage: no tokens were generated
    segments = [signal for signal in posted if isinstance(signal, SegmentStarted)]
    assert "".join(segment.text for segment in segments) == line
    assert [r.text for r in tts.requests] == [segment.text.strip() for segment in segments]
    assert {r.lang for r in tts.requests} == {"en"}
    assert kinds(posted)[-1] == "SpeechFinished"


async def test_a_verbatim_line_in_text_modality_is_one_text_delta() -> None:
    llm, tts = EchoLLM(), SineTTS()
    posted: list[PipelineSignal] = []
    await run_pipeline(
        request(verbatim="请稍等。", modality="text"), context(llm, tts, posted, StepClock())
    )
    assert kinds(posted) == ["LlmText", "LlmFinished"]
    assert (llm.requests, tts.requests) == ([], [])


async def test_an_empty_line_says_nothing() -> None:
    llm, tts = EchoLLM(), SineTTS()
    posted: list[PipelineSignal] = []
    await run_pipeline(request(verbatim=""), context(llm, tts, posted, StepClock()))
    assert kinds(posted) == ["LlmFinished", "SpeechFinished"]


async def test_the_tts_hears_normalized_text_and_the_transcript_keeps_the_original() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    text = "请拨打 138-0013-8000 联系我们。"
    llm.script(TextDelta(text), LLMDone("stop", Usage(5, 5)))
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    (segment,) = (signal for signal in posted if isinstance(signal, SegmentStarted))
    assert segment.text == text
    assert "138" not in segment.spoken
    assert "幺三八" in segment.spoken  # the phone number is read digit by digit (default on)
    assert tts.requests[0].text == segment.spoken.strip()


async def test_segments_with_nothing_to_say_are_not_synthesized() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    llm.script(TextDelta("好的。"), TextDelta("\n……\n"), LLMDone("stop", None))
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    segments = [signal for signal in posted if isinstance(signal, SegmentStarted)]
    # The answer's trailing newline is dropped (M5); the rest of its transcript is kept.
    assert "".join(segment.text for segment in segments) == "好的。\n……"
    assert [r.text for r in tts.requests] == ["好的。"]  # punctuation alone is never sent


async def test_whitespace_around_the_answer_is_dropped() -> None:
    """Leading whitespace never opens a message; trailing whitespace waits for more text and is
    dropped at the end; whitespace inside the answer is kept."""
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    deltas = ["\n", " Sure.", "\n\n", "Done. ", "\n"]
    llm.script(*(TextDelta(text) for text in deltas), LLMDone("stop", None))
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    texts = [signal.text for signal in posted if isinstance(signal, LlmText)]
    assert texts == ["Sure.", "\n\nDone."]


async def test_timeline_marks() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    llm.script(TextDelta("你好。"), TextDelta("再见。"), LLMDone("stop", None))
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    marks = [signal for signal in posted if isinstance(signal, PipelineMark)]
    assert [mark.name for mark in marks] == ["first_sentence_ready", "tts_first_chunk"]
    values = dict(marks[1].values)
    assert values == {"tts_queue_ms": 0, "tts_model": "sine"}
    assert all(mark.generation == 7 for mark in marks)


async def test_synthesis_stops_once_the_response_passes_its_audio_limit() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=100, chunk_ms=40)
    llm.script(
        TextDelta("这是第一句很长很长的话。"),
        TextDelta("这是第二句很长很长的话。"),
        TextDelta("这是第三句很长很长的话。"),
        LLMDone("stop", None),
    )
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock(), max_audio_ms=500))
    assert 500 * 48 < audio_of(posted) <= (500 + 40) * 48  # past the limit by under one chunk
    assert len(tts.requests) == 1  # no request after the limit
    assert tts.streams.active == 0
    assert kinds(posted)[-1] == "SpeechFinished"


async def test_a_tts_failure_ends_the_pipeline_with_the_stage() -> None:
    llm, tts = EchoLLM(), SineTTS(ms_per_char=20, chunk_ms=40)
    llm.script(TextDelta("你好。"), TextDelta("再见。"), LLMDone("stop", None))
    tts.next_chunk.fail(ProviderError("bad_response", "tts: the stream was cut"))
    posted: list[PipelineSignal] = []
    await run_pipeline(request(), context(llm, tts, posted, StepClock()))
    (failed,) = (signal for signal in posted if isinstance(signal, PipelineFailed))
    assert (failed.stage, failed.error.kind) == ("tts", "bad_response")
    assert "SpeechFinished" not in kinds(posted)
    assert tts.streams.active == 0

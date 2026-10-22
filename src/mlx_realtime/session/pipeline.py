"""One response's pipeline: LLM → segmenter → normalizer → TTS → encoder → sender
(ARCHITECTURE §7).

The pipeline runs in the response scope, a TaskGroup that the actor cancels to stop it. Its
tasks report to the actor only through signals; the actor turns them into events and emits
every terminal event itself.

    LLM reader ─text─► segmenter ─segments─► TTS (normalize, synthesize, encode) ─audio─► sender
        └── text, tool calls, done ───────────────────────────────────────────► actor ◄──┘

- A verbatim line (D16) takes the LLM reader's place: its text is the whole answer.
- The TTS stage makes one request per segment, in order, the next as soon as the previous one
  has finished generating. It drains each stream as fast as the backend produces it, so a slow
  client or the lead cap never holds a backend that every session shares; the audio waits in
  this response's queue, bounded by ``max_audio_ms``.
- The sender posts that audio paced (D6): at once, but at most ``max_lead_ms`` ahead of real
  time. A segment's transcript goes out just before its first audio delta.
- Text modality runs the LLM reader (or the line) alone and never opens TTS.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal

from mlx_realtime.audio.codec import OutputEncoder
from mlx_realtime.audio.pcm import SAMPLE_WIDTH, WIRE_RATE, ms_to_bytes
from mlx_realtime.config import SegmenterConfig
from mlx_realtime.providers.base import (
    LLMDone,
    LLMProvider,
    Message,
    ProviderError,
    SpeechTiming,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolDef,
    TTSProvider,
)
from mlx_realtime.session.clock import Clock
from mlx_realtime.session.errors import first_leaf
from mlx_realtime.session.pacing import Pacer
from mlx_realtime.session.settings import Modality
from mlx_realtime.session.signals import (
    LlmFinished,
    LlmText,
    LlmToolCall,
    PipelineFailed,
    PipelineMark,
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    SpeechFinished,
)
from mlx_realtime.text.lang import detect_lang
from mlx_realtime.text.normalize import NormalizerOptions, normalize
from mlx_realtime.text.segmenter import Segmenter

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    generation: int
    messages: Sequence[Message]
    tools: Sequence[ToolDef]
    tool_choice: ToolChoice
    max_output_tokens: int | None
    modality: Modality
    voice: str
    verbatim: str | None = None  # D16: say this, as written, instead of asking the LLM


@dataclass(frozen=True, slots=True)
class PipelineContext:
    llm: LLMProvider
    tts: TTSProvider
    segmenter: SegmenterConfig
    normalizer: NormalizerOptions
    max_delta_ms: int
    max_lead_ms: int
    max_audio_ms: int  # synthesis stops past this; the actor ends the response incomplete
    post: Callable[[PipelineSignal], None]
    wait_for_room: Callable[[], Awaitable[None]]  # outbound backpressure, before each delta event
    clock: Clock


@dataclass(frozen=True, slots=True)
class _Delta:
    segment: SegmentStarted
    pcm: bytes  # at most max_delta_ms


@dataclass(frozen=True, slots=True)
class _SegmentEnd:
    segment: SegmentStarted


type _Audio = _Delta | _SegmentEnd | None  # None: no more segments


class _StageFailed(Exception):
    def __init__(self, stage: Literal["llm", "tts"], error: ProviderError) -> None:
        super().__init__(str(error))
        self.stage: Literal["llm", "tts"] = stage
        self.error = error


async def run_pipeline(request: PipelineRequest, context: PipelineContext) -> None:
    """The response scope. Failures become a ``PipelineFailed`` signal; cancellation propagates."""
    try:
        async with asyncio.TaskGroup() as scope:
            if request.modality == "text":
                scope.create_task(_stage("llm", _source(request, context, None)))
            else:
                texts: asyncio.Queue[str | None] = asyncio.Queue()
                segments: asyncio.Queue[str | None] = asyncio.Queue()
                audio: asyncio.Queue[_Audio] = asyncio.Queue()
                scope.create_task(_stage("llm", _source(request, context, texts)))
                scope.create_task(_segment(request.generation, texts, segments, context))
                scope.create_task(_stage("tts", _synthesize(request, segments, audio, context)))
                scope.create_task(_send(request.generation, audio, context))
    except* _StageFailed as group:
        failure = first_leaf(group)
        context.post(PipelineFailed(request.generation, failure.stage, failure.error))
    except* Exception as group:
        logger.error("response pipeline failed", exc_info=group.exceptions[0])
        error = ProviderError("internal", "internal error")
        context.post(PipelineFailed(request.generation, "pipeline", error))


async def _stage(stage: Literal["llm", "tts"], work: Coroutine[object, object, None]) -> None:
    """Tag a provider failure with the stage it came from."""
    try:
        await work
    except ProviderError as error:
        raise _StageFailed(stage, error) from error


async def _source(
    request: PipelineRequest, context: PipelineContext, texts: asyncio.Queue[str | None] | None
) -> None:
    if request.verbatim is None:
        await _read_llm(request, context, texts)
    else:
        await _read_line(request.generation, request.verbatim, context, texts)


async def _read_line(
    generation: int, line: str, context: PipelineContext, texts: asyncio.Queue[str | None] | None
) -> None:
    """A verbatim line, delivered as an LLM that answered it in one delta (no usage)."""
    if line:
        if texts is None:
            await context.wait_for_room()  # text modality: the line is an output_text delta
        context.post(LlmText(generation, line))
        if texts is not None:
            texts.put_nowait(line)
    context.post(LlmFinished(generation, LLMDone("stop", None)))
    if texts is not None:
        texts.put_nowait(None)


async def _read_llm(
    request: PipelineRequest, context: PipelineContext, texts: asyncio.Queue[str | None] | None
) -> None:
    generation = request.generation
    stream = context.llm.stream(
        request.messages,
        request.tools,
        tool_choice=request.tool_choice,
        max_output_tokens=request.max_output_tokens,
    )
    # aclosing: a cancel that lands outside the provider (here, waiting for room) still
    # closes its stream, and with it the backend request, before this task ends.
    async with aclosing(stream):
        async for delta in stream:
            match delta:
                case TextDelta(text=text):
                    if texts is None:
                        await context.wait_for_room()  # text modality: each delta is an event
                        context.post(LlmText(generation, text))
                    else:
                        context.post(LlmText(generation, text))
                        texts.put_nowait(text)
                case ToolCall():
                    context.post(LlmToolCall(generation, delta))
                case LLMDone():
                    context.post(LlmFinished(generation, delta))
    if texts is not None:
        texts.put_nowait(None)


async def _segment(
    generation: int,
    texts: asyncio.Queue[str | None],
    segments: asyncio.Queue[str | None],
    context: PipelineContext,
) -> None:
    config = context.segmenter
    segmenter = Segmenter(
        first_min_chars=config.first_min_chars,
        first_max_wait_ms=config.first_max_wait_ms,
        min_chars=config.min_chars,
        max_chars=config.max_chars,
    )
    first = True

    def now_ms() -> float:
        return context.clock.now() * 1000

    def put(completed: list[str]) -> None:
        nonlocal first
        for segment in completed:
            if first:
                first = False
                context.post(PipelineMark(generation, "first_sentence_ready", context.clock.now()))
            segments.put_nowait(segment)

    while True:
        deadline = segmenter.deadline_ms()
        delay = None if deadline is None else max(deadline - now_ms(), 0.0) / 1000
        try:
            async with asyncio.timeout(delay):
                text = await texts.get()
        except TimeoutError:
            completed = segmenter.poll(now_ms())
            if completed:
                put(completed)
                continue
            # The first segment's wait expired without a boundary; the next text decides.
            text = await texts.get()
        if text is None:
            put(segmenter.flush())
            segments.put_nowait(None)
            return
        put(segmenter.push(text, now_ms()))


async def _synthesize(
    request: PipelineRequest,
    segments: asyncio.Queue[str | None],
    audio: asyncio.Queue[_Audio],
    context: PipelineContext,
) -> None:
    """One TTS request per segment, in order; every stream is read as fast as it arrives."""
    limit_bytes = ms_to_bytes(context.max_audio_ms, WIRE_RATE)
    produced = 0
    first_chunk = True
    index = 0
    while (text := await segments.get()) is not None:
        lang = detect_lang(text)  # mixed text is zh
        spoken = normalize(text, context.normalizer, lang=lang)
        segment = SegmentStarted(request.generation, index, text, spoken)
        index += 1
        if _speakable(spoken):
            timing = SpeechTiming()
            encoder = OutputEncoder(context.max_delta_ms)
            stream = context.tts.stream(spoken.strip(), request.voice, lang, timing=timing)
            async with aclosing(stream):
                async for chunk in stream:
                    if first_chunk:
                        first_chunk = False
                        context.post(_first_chunk_mark(request.generation, timing, context))
                    for pcm in encoder.encode(chunk.pcm, chunk.sample_rate):
                        audio.put_nowait(_Delta(segment, pcm))
                        produced += len(pcm)
                    if produced > limit_bytes:
                        break
            for pcm in encoder.flush():
                audio.put_nowait(_Delta(segment, pcm))
                produced += len(pcm)
        audio.put_nowait(_SegmentEnd(segment))
        if produced > limit_bytes:
            break  # the actor ends the response incomplete once this much audio is sent
    audio.put_nowait(None)


def _first_chunk_mark(
    generation: int, timing: SpeechTiming, context: PipelineContext
) -> PipelineMark:
    values: tuple[tuple[str, object], ...] = (
        ("tts_queue_ms", round(timing.queued_s * 1000)),
        ("tts_model", timing.model),
    )
    if timing.fallback:
        values += (("tts_fallback", True),)
    return PipelineMark(generation, "tts_first_chunk", context.clock.now(), values)


async def _send(generation: int, audio: asyncio.Queue[_Audio], context: PipelineContext) -> None:
    """Post the synthesized audio, paced (D6); each segment's transcript just before its audio."""
    pacer = Pacer(context.max_lead_ms, context.clock)
    announced: set[int] = set()
    while (item := await audio.get()) is not None:
        segment = item.segment
        if isinstance(item, _Delta):
            await pacer.admit(len(item.pcm) / SAMPLE_WIDTH * 1000 / WIRE_RATE)
            await context.wait_for_room()
            if segment.index not in announced:
                announced.add(segment.index)
                context.post(segment)
            context.post(SegmentAudio(generation, segment.index, item.pcm))
        elif segment.index not in announced:
            announced.add(segment.index)
            context.post(segment)  # a segment that produced no audio still has its transcript
    context.post(SpeechFinished(generation))


def _speakable(text: str) -> bool:
    """Whether the text has anything to say: a letter, a digit or a CJK character."""
    return any(char.isalnum() for char in text)

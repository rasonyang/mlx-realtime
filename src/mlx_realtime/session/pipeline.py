"""One response's pipeline: LLM → segmenter → normalizer → TTS → encoder (ARCHITECTURE §7).

The pipeline runs in the response scope, a TaskGroup that the actor cancels to stop it. Its
tasks report to the actor only through signals; the actor turns them into events and emits
every terminal event itself.

    LLM reader ──text──► segmenter + normalizer ──segments──► TTS + encoder
        └── text, tool calls, done ─────────────────────────► actor ◄── audio

Text modality runs the LLM reader alone and never opens TTS.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal

from mlx_realtime.audio.codec import OutputEncoder
from mlx_realtime.config import SegmenterConfig
from mlx_realtime.providers.base import (
    LLMDone,
    LLMProvider,
    Message,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolDef,
    TTSProvider,
)
from mlx_realtime.session.errors import first_leaf
from mlx_realtime.session.settings import Modality
from mlx_realtime.session.signals import (
    LlmFinished,
    LlmText,
    LlmToolCall,
    PipelineFailed,
    PipelineSignal,
    SegmentAudio,
    SegmentStarted,
    SpeechFinished,
)
from mlx_realtime.text.lang import detect_lang
from mlx_realtime.text.normalize import normalize
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


@dataclass(frozen=True, slots=True)
class PipelineContext:
    llm: LLMProvider
    tts: TTSProvider
    segmenter: SegmenterConfig
    max_delta_ms: int
    post: Callable[[PipelineSignal], None]
    wait_for_room: Callable[[], Awaitable[None]]  # outbound backpressure, before each delta event
    clock_ms: Callable[[], float]


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
                scope.create_task(_stage("llm", _read_llm(request, context, None)))
            else:
                texts: asyncio.Queue[str | None] = asyncio.Queue()
                segments: asyncio.Queue[str | None] = asyncio.Queue()
                scope.create_task(_stage("llm", _read_llm(request, context, texts)))
                scope.create_task(_segment(texts, segments, context))
                scope.create_task(_stage("tts", _speak(request, segments, context)))
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
    texts: asyncio.Queue[str | None], segments: asyncio.Queue[str | None], context: PipelineContext
) -> None:
    config = context.segmenter
    segmenter = Segmenter(
        first_min_chars=config.first_min_chars,
        first_max_wait_ms=config.first_max_wait_ms,
        min_chars=config.min_chars,
        max_chars=config.max_chars,
    )

    def put(completed: list[str]) -> None:
        for segment in completed:
            segments.put_nowait(segment)

    while True:
        deadline = segmenter.deadline_ms()
        delay = None if deadline is None else max(deadline - context.clock_ms(), 0.0) / 1000
        try:
            async with asyncio.timeout(delay):
                text = await texts.get()
        except TimeoutError:
            completed = segmenter.poll(context.clock_ms())
            if completed:
                put(completed)
                continue
            # The first segment's wait expired without a boundary; the next text decides.
            text = await texts.get()
        if text is None:
            put(segmenter.flush())
            segments.put_nowait(None)
            return
        put(segmenter.push(text, context.clock_ms()))


async def _speak(
    request: PipelineRequest, segments: asyncio.Queue[str | None], context: PipelineContext
) -> None:
    index = 0
    while (text := await segments.get()) is not None:
        await _speak_segment(request, index, text, context)
        index += 1
    context.post(SpeechFinished(request.generation))


async def _speak_segment(
    request: PipelineRequest, index: int, text: str, context: PipelineContext
) -> None:
    spoken = normalize(text)
    started = SegmentStarted(request.generation, index, text, spoken)
    announced = False
    if spoken.strip():
        encoder = OutputEncoder(context.max_delta_ms)
        stream = context.tts.stream(spoken.strip(), request.voice, detect_lang(spoken))
        async with aclosing(stream):
            async for chunk in stream:
                deltas = encoder.encode(chunk.pcm, chunk.sample_rate)
                announced = await _deliver(deltas, started, announced, context)
        announced = await _deliver(encoder.flush(), started, announced, context)
    if not announced:
        context.post(started)


async def _deliver(
    deltas: list[bytes], started: SegmentStarted, announced: bool, context: PipelineContext
) -> bool:
    """Post audio deltas; the segment's transcript goes out just before its first audio."""
    for delta in deltas:
        await context.wait_for_room()
        if not announced:
            context.post(started)
            announced = True
        context.post(SegmentAudio(started.generation, started.index, delta))
    return announced

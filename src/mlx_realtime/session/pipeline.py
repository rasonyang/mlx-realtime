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
- Each segment's edge silence is trimmed as it streams (``audio/trim.py``, M0 S3): at most
  60 ms before its first voiced frame, 100 ms after its last. Trimmed audio never enters the
  queue, so everything downstream counts the audio actually sent.
- The sender posts that audio paced (D6): at once, but at most ``max_lead_ms`` ahead of real
  time. A segment's transcript goes out just before its first audio delta.
- Multi-session TTS fairness (M6). Each TTS request carries a deadline: when this response's
  audio runs out without it (the playout horizon of the audio synthesized so far), or, for the
  first segment, when the pipeline started (plus ``first_segment_grace_s``); a shared TTS
  queue serves the earliest first. With
  the audio reserve on, the sender holds back up to ``hold_ms`` of audio while segments are
  still to come and releases it a delta at a time, at least every ``keepalive_ms`` and whenever
  the client's modelled buffer runs low, so deltas keep flowing while the next segment waits
  in the queue.
- Text modality runs the LLM reader (or the line) alone and never opens TTS.
- Tool calls go to the actor as they complete, whatever the TTS is doing, so a tool-only
  response is done as soon as the LLM is. A call the response did not offer (``tool_choice``
  ``none``, a function not in the session's tools) is dropped. Whitespace at either end of the
  answer is dropped, so the newline a model writes around a call opens no empty message and
  never ends a transcript.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal

from mlx_realtime.audio.codec import OutputEncoder
from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.audio.resample import Quality
from mlx_realtime.audio.trim import SilenceTrimmer
from mlx_realtime.config import AudioReserveConfig, SegmenterConfig, SilenceTrimConfig
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
from mlx_realtime.session.observer import NULL_OBSERVER, SessionObserver
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
from mlx_realtime.text.normalize import NormalizerOptions, normalize_with_map
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
    output_format: AudioFormat = PCM  # the response's wire audio


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
    trim_silence: SilenceTrimConfig  # each segment's edge silence (M0 S3)
    reserve: AudioReserveConfig | None = None  # the audio reserve (M6); None: off
    first_segment_grace_s: float = 0.0  # a first segment's TTS deadline after the start (M6)
    observer: SessionObserver = NULL_OBSERVER  # TTS fallback uses and absorbed primary errors
    g711_quality: Quality = "HQ"  # the TTS rate → 8 kHz resampling of G.711 output (M7)


@dataclass(frozen=True, slots=True)
class _Delta:
    segment: SegmentStarted
    pcm: bytes  # in the response's output format, at most max_delta_ms
    ms: float  # its duration


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
                synthesize = _synthesize(request, segments, audio, context, context.clock.now())
                scope.create_task(_stage("tts", synthesize))
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
    callable_tools = (
        {tool.name for tool in request.tools} if request.tool_choice == "auto" else set[str]()
    )
    # Whitespace at either end of the answer is dropped: before its first visible text, and
    # held back after its latest until more text follows (else dropped at the end).
    started = False
    held = ""
    first = True  # the first token's mark is taken here, when it arrives (M6)

    def mark_first() -> None:
        nonlocal first
        if first:
            first = False
            context.post(PipelineMark(generation, "llm_first_token", context.clock.now()))

    # aclosing: a cancel that lands outside the provider (here, waiting for room) still
    # closes its stream, and with it the backend request, before this task ends.
    async with aclosing(stream):
        async for delta in stream:
            match delta:
                case TextDelta(text=text):
                    text = held + text if started else text.lstrip()
                    body = text.rstrip()
                    held = text[len(body) :]
                    if not body:
                        continue  # e.g. the newline a model writes around a tool call
                    started = True
                    text = body
                    mark_first()
                    if texts is None:
                        await context.wait_for_room()  # text modality: each delta is an event
                        context.post(LlmText(generation, text))
                    else:
                        context.post(LlmText(generation, text))
                        texts.put_nowait(text)
                case ToolCall(name=name) if name in callable_tools:
                    mark_first()
                    context.post(LlmToolCall(generation, delta))
                case ToolCall(name=name):
                    # The server parses call markup even when no tools were sent; a call the
                    # response may not make (tool_choice none, an undeclared function) is
                    # dropped, never delivered.
                    logger.warning(
                        "tool call dropped: not offered to this response",
                        extra={"tool": name, "tool_choice": request.tool_choice},
                    )
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
    started_at: float,
) -> None:
    """One TTS request per segment, in order; every stream is read as fast as it arrives.

    One encoder serves the response: the resampler to the output format's rate (24 → 8 kHz for
    G.711) is flushed at each segment's end, so a segment's audio is whole when it is sent."""
    audio_format = request.output_format
    limit_bytes = audio_format.ms_to_bytes(context.max_audio_ms)
    produced = 0
    first_chunk = True
    index = 0
    horizon = _Horizon(context.clock)
    quality: Quality = "HQ" if audio_format.law is None else context.g711_quality
    encoder = OutputEncoder(context.max_delta_ms, audio_format, quality)

    def delta(segment: SegmentStarted, pcm: bytes) -> _Delta:
        nonlocal produced
        produced += len(pcm)
        ms = audio_format.duration_ms(len(pcm))
        horizon.add(ms)
        return _Delta(segment, pcm, ms)

    while (text := await segments.get()) is not None:
        lang = detect_lang(text)  # mixed text is zh
        spoken, source = normalize_with_map(text, context.normalizer, lang=lang)
        segment = SegmentStarted(request.generation, index, text, spoken, source)
        index += 1
        if _speakable(spoken):
            deadline = (
                started_at + context.first_segment_grace_s
                if horizon.dry_at is None
                else horizon.dry_at
            )
            slack_s = max(deadline - context.clock.now(), 0.0)
            timing = SpeechTiming(deadline=deadline, slack_s=slack_s)
            trimmer: SilenceTrimmer | None = None
            rate = 0
            stream = context.tts.stream(spoken.strip(), request.voice, lang, timing=timing)
            async with aclosing(stream):
                async for chunk in stream:
                    if first_chunk:
                        first_chunk = False
                        context.post(_first_chunk_mark(request.generation, timing, context))
                    if not rate:
                        rate, trimmer = chunk.sample_rate, _trimmer(chunk.sample_rate, context)
                    pcm = chunk.pcm if trimmer is None else trimmer.push(chunk.pcm)
                    for data in encoder.encode(pcm, chunk.sample_rate):
                        audio.put_nowait(delta(segment, data))
                    if produced > limit_bytes:
                        break
            _report_fallback(timing, context)
            # Only audio that passed the trimmer exists from here on: the segment map,
            # truncation, D5 and the echo reference all count what is sent.
            tail = b"" if trimmer is None else trimmer.flush()
            deltas = encoder.encode(tail, rate) if tail else []
            for data in [*deltas, *encoder.flush()]:
                audio.put_nowait(delta(segment, data))
        audio.put_nowait(_SegmentEnd(segment))
        if produced > limit_bytes:
            break  # the actor ends the response incomplete once this much audio is sent
    audio.put_nowait(None)


class _Horizon:
    """When the audio synthesized so far runs out at the client, if it plays from when each
    piece was made (the response's deadline for its next segment)."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.dry_at: float | None = None

    def add(self, ms: float) -> None:
        now = self._clock.now()
        start = now if self.dry_at is None else max(self.dry_at, now)
        self.dry_at = start + ms / 1000


def _report_fallback(timing: SpeechTiming, context: PipelineContext) -> None:
    if timing.fallback:
        context.observer.tts_fallback()
    if timing.primary_error is not None:
        context.observer.provider_error("tts", timing.primary_error)


def _trimmer(sample_rate: int, context: PipelineContext) -> SilenceTrimmer | None:
    """The segment's silence trimmer, or None when trimming is off."""
    config = context.trim_silence
    if not config.enabled:
        return None
    return SilenceTrimmer(
        sample_rate,
        threshold_dbfs=config.threshold_dbfs,
        frame_ms=config.frame_ms,
        max_lead_ms=config.max_lead_ms,
        max_tail_ms=config.max_tail_ms,
    )


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
    reserve = context.reserve
    if reserve is not None and reserve.enabled and reserve.hold_ms > 0:
        await _ReserveSender(generation, audio, context, reserve).run()
        return
    pacer = Pacer(context.max_lead_ms, context.clock)
    announced: set[int] = set()
    while (item := await audio.get()) is not None:
        segment = item.segment
        if isinstance(item, _Delta):
            await pacer.admit(item.ms)
            await context.wait_for_room()
            if segment.index not in announced:
                announced.add(segment.index)
                context.post(segment)
            context.post(SegmentAudio(generation, segment.index, item.pcm))
        elif segment.index not in announced:
            announced.add(segment.index)
            context.post(segment)  # a segment that produced no audio still has its transcript
    context.post(SpeechFinished(generation))


class _ReserveSender:
    """The sender with the audio reserve (M6): while more audio is to come, the last
    ``hold_ms`` of what is synthesized stays here and goes out one delta at a time, when the
    wire has been quiet for ``keepalive_ms`` or the client's modelled buffer (the pacer's lead)
    drops below ``min_lead_ms``. Everything else goes out as the plain sender sends it."""

    def __init__(
        self,
        generation: int,
        audio: asyncio.Queue[_Audio],
        context: PipelineContext,
        reserve: AudioReserveConfig,
    ) -> None:
        self._generation = generation
        self._audio = audio
        self._context = context
        self._clock = context.clock
        self._hold_ms = reserve.hold_ms
        self._keepalive_s = reserve.keepalive_ms / 1000
        self._min_lead_ms = reserve.min_lead_ms
        self._pacer = Pacer(context.max_lead_ms, context.clock)
        self._pending: deque[_Delta | _SegmentEnd] = deque()
        self._pending_ms = 0.0
        self._finished = False
        self._announced: set[int] = set()
        self._last_sent: float | None = None
        self._getter: asyncio.Task[_Audio] | None = None

    async def run(self) -> None:
        try:
            while True:
                self._take_available()
                if self._pending:
                    head = self._pending[0]
                    if isinstance(head, _SegmentEnd):
                        self._pending.popleft()
                        self._announce(head.segment)
                        continue
                    wait_s = self._hold_s(head)
                    if wait_s <= 0:
                        await self._send(head)
                        continue
                elif self._finished:
                    break
                else:
                    wait_s = None
                await self._wait(wait_s)
        finally:
            if self._getter is not None:
                self._getter.cancel()
        self._context.post(SpeechFinished(self._generation))

    def _take_available(self) -> None:
        while not self._finished and not self._audio.empty():
            self._accept(self._audio.get_nowait())

    def _accept(self, item: _Audio) -> None:
        if item is None:
            self._finished = True
            return
        self._pending.append(item)
        if isinstance(item, _Delta):
            self._pending_ms += item.ms

    def _hold_s(self, head: _Delta) -> float:
        """How long the head delta may still wait; 0 or less: send it now."""
        if self._finished or self._pending_ms - head.ms >= self._hold_ms:
            return 0.0
        if self._last_sent is None:
            return 0.0  # the response's first audio never waits
        now = self._clock.now()
        keepalive_s = self._last_sent + self._keepalive_s - now
        lead_s = (self._pacer.lead_ms() - self._min_lead_ms) / 1000
        return min(keepalive_s, lead_s)

    async def _send(self, item: _Delta) -> None:
        self._pending.popleft()
        self._pending_ms -= item.ms
        await self._pacer.admit(item.ms)
        await self._context.wait_for_room()
        self._announce(item.segment)
        self._context.post(SegmentAudio(self._generation, item.segment.index, item.pcm))
        self._last_sent = self._clock.now()

    def _announce(self, segment: SegmentStarted) -> None:
        if segment.index not in self._announced:
            self._announced.add(segment.index)
            self._context.post(segment)

    async def _wait(self, wait_s: float | None) -> None:
        """Until more audio arrives, or ``wait_s`` has passed (None: no limit)."""
        if self._getter is None:
            self._getter = asyncio.ensure_future(self._audio.get())
        if wait_s is None:
            await asyncio.wait({self._getter})
        else:
            sleeper = asyncio.ensure_future(self._clock.sleep(wait_s))
            try:
                await asyncio.wait({self._getter, sleeper}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                sleeper.cancel()
        if self._getter.done():
            getter, self._getter = self._getter, None
            self._accept(getter.result())


def _speakable(text: str) -> bool:
    """Whether the text has anything to say: a letter, a digit or a CJK character."""
    return any(char.isalnum() for char in text)

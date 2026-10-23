"""``FallbackTTS`` (ARCHITECTURE §5.2, D13): a segment moves to the fallback only if the primary
fails before its first chunk; the circuit breaker; cancellation; the Backend side."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field

import pytest

from mlx_realtime.providers.base import (
    Lang,
    PCMChunk,
    ProviderError,
    ProviderTimeout,
    Readiness,
    SpeechTiming,
)
from mlx_realtime.providers.fallback import FallbackTTS

FAIL = ProviderError("bad_response", "tts: the stream was cut")


@dataclass
class Plan:
    """One stream: ``chunks`` chunks, then ``error`` (if any); ``hold`` stops before chunk N."""

    chunks: int = 2
    error: ProviderError | None = None
    hold: int | None = None


@dataclass
class ScriptedTTS:
    name: str
    plans: deque[Plan] = field(default_factory=deque[Plan])
    calls: list[tuple[str, str, Lang]] = field(default_factory=list[tuple[str, str, Lang]])
    closed: int = 0
    warmed: int = 0
    readiness: Readiness = field(default_factory=lambda: Readiness(ready=True, detail="ok"))
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    holding: asyncio.Event = field(default_factory=asyncio.Event)

    async def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        self.calls.append((text, voice, lang))
        plan = self.plans.popleft() if self.plans else Plan()
        if timing is not None:
            timing.model = self.name
            timing.queued_s += 0.01
        try:
            for index in range(plan.chunks):
                if plan.hold == index:
                    self.holding.set()
                    await self.gate.wait()
                yield PCMChunk(self.name.encode() + bytes([index]), 24000)
            if plan.error is not None:
                raise plan.error
        finally:
            self.closed += 1

    async def warmup(self) -> None:
        self.warmed += 1

    async def ready(self) -> Readiness:
        return self.readiness

    async def aclose(self) -> None:
        return None


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def fallback_tts(
    primary: ScriptedTTS, fallback: ScriptedTTS, clock: Clock | None = None
) -> FallbackTTS:
    return FallbackTTS(primary, fallback, failures=3, cooldown_s=30.0, now=clock or Clock())


async def speak(tts: FallbackTTS, timing: SpeechTiming | None = None) -> list[bytes]:
    async with aclosing(tts.stream("你好。", "vivian", "zh", timing=timing)) as stream:
        return [chunk.pcm async for chunk in stream]


async def test_the_primary_speaks_when_it_works() -> None:
    primary, fallback = ScriptedTTS("p"), ScriptedTTS("f")
    timing = SpeechTiming()
    assert await speak(fallback_tts(primary, fallback), timing) == [b"p\x00", b"p\x01"]
    assert fallback.calls == []
    assert (timing.model, timing.fallback) == ("p", False)


@pytest.mark.parametrize(
    "error",
    [FAIL, ProviderTimeout("first_item"), ProviderError("unavailable", "down")],
    ids=["bad-response", "first-chunk-timeout", "unavailable"],
)
async def test_a_primary_failure_before_audio_moves_the_segment_to_the_fallback(
    error: ProviderError,
) -> None:
    primary, fallback = ScriptedTTS("p", deque([Plan(chunks=0, error=error)])), ScriptedTTS("f")
    timing = SpeechTiming()
    assert await speak(fallback_tts(primary, fallback), timing) == [b"f\x00", b"f\x01"]
    assert fallback.calls == [("你好。", "vivian", "zh")]  # the same voice, so the same speaker
    assert timing.model == "f"
    assert timing.queued_s == pytest.approx(0.02)  # both waits count
    assert primary.closed == 1


async def test_a_primary_failure_after_audio_fails_the_segment() -> None:
    primary, fallback = ScriptedTTS("p", deque([Plan(chunks=1, error=FAIL)])), ScriptedTTS("f")
    received: list[bytes] = []
    tts = fallback_tts(primary, fallback)

    async def consume() -> None:
        async with aclosing(tts.stream("你好。", "vivian", "zh")) as stream:
            async for chunk in stream:
                received.append(chunk.pcm)  # noqa: PERF401 (the chunks before the failure count)

    with pytest.raises(ProviderError) as failed:
        await consume()
    assert failed.value is FAIL
    assert received == [b"p\x00"]  # the voice never switches mid-sentence
    assert fallback.calls == []


async def test_a_full_backend_queue_is_not_retried_on_the_same_backend() -> None:
    busy = ProviderError("overloaded", "too many requests in flight", retryable=True)
    primary, fallback = ScriptedTTS("p", deque([Plan(chunks=0, error=busy)] * 5)), ScriptedTTS("f")
    tts = fallback_tts(primary, fallback)
    for _ in range(5):
        with pytest.raises(ProviderError) as failed:
            await speak(tts)
        assert failed.value.kind == "overloaded"
    assert fallback.calls == []
    assert not tts.breaker.open  # a busy backend is not the primary model failing


async def test_the_breaker_opens_after_consecutive_failures_for_the_cooldown() -> None:
    clock = Clock()
    plans = deque([Plan(chunks=0, error=FAIL)] * 3)
    primary, fallback = ScriptedTTS("p", plans), ScriptedTTS("f")
    tts = fallback_tts(primary, fallback, clock)
    for _ in range(3):
        assert await speak(tts) == [b"f\x00", b"f\x01"]
    assert tts.breaker.open
    assert len(primary.calls) == 3
    clock.now += 29.0
    await speak(tts)
    assert len(primary.calls) == 3  # open: straight to the fallback
    timing = SpeechTiming()
    await speak(tts, timing)
    assert timing.model == "f"
    clock.now += 2.0  # the cooldown is over: the primary is tried again
    primary.plans.append(Plan(chunks=0, error=FAIL))
    await speak(tts)
    assert len(primary.calls) == 4
    assert tts.breaker.open  # one more failure reopens the circuit at once
    clock.now += 31.0
    assert await speak(tts) == [b"p\x00", b"p\x01"]
    assert not tts.breaker.open


async def test_a_success_resets_the_count() -> None:
    plans = deque([Plan(chunks=0, error=FAIL), Plan(chunks=0, error=FAIL), Plan(), Plan(0, FAIL)])
    primary, fallback = ScriptedTTS("p", plans), ScriptedTTS("f")
    tts = fallback_tts(primary, fallback)
    for _ in range(4):
        await speak(tts)
    assert not tts.breaker.open


@pytest.mark.parametrize("on", ["primary", "fallback"])
async def test_cancellation_closes_the_running_stream(on: str) -> None:
    plans = deque([Plan(chunks=3, hold=1)] if on == "primary" else [Plan(chunks=0, error=FAIL)])
    primary = ScriptedTTS("p", plans)
    fallback = ScriptedTTS("f", deque([Plan(chunks=3, hold=1)]))
    tts = fallback_tts(primary, fallback)
    task = asyncio.create_task(speak(tts))
    running = primary if on == "primary" else fallback
    await running.holding.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert running.closed == 1
    assert not tts.breaker.open


async def test_warmup_readiness_and_close_cover_both_models() -> None:
    primary, fallback = ScriptedTTS("p"), ScriptedTTS("f")
    fallback.readiness = Readiness(ready=False, detail="fallback model not loaded")
    tts = fallback_tts(primary, fallback)
    await tts.warmup()
    assert (primary.warmed, fallback.warmed) == (1, 1)
    state = await tts.ready()
    assert not state.ready
    assert state.detail == "primary: ok; fallback: fallback model not loaded"
    await tts.aclose()


async def collect_text(tts: FallbackTTS, text: str, lang: Lang = "zh") -> list[bytes]:
    async with aclosing(tts.stream(text, "vivian", lang)) as stream:
        return [chunk.pcm async for chunk in stream]


@pytest.mark.parametrize("text", ["请登录 App 查看。", "Hello.", "订单 A B 1 2。", "ＯＫ。"])
async def test_a_segment_with_latin_letters_falls_back_like_any_other(text: str) -> None:
    """D13 final: the fallback is a Qwen3-TTS build too, which reads English inside Chinese
    (Kokoro did not, M0 S3), so Latin letters change nothing."""
    primary, fallback = ScriptedTTS("p", deque([Plan(chunks=0, error=FAIL)])), ScriptedTTS("f")
    tts = fallback_tts(primary, fallback)
    assert await collect_text(tts, text) == [b"f\x00", b"f\x01"]
    assert fallback.calls == [(text, "vivian", "zh")]


async def test_every_segment_goes_to_the_fallback_while_the_circuit_is_open() -> None:
    primary = ScriptedTTS("p", deque([Plan(chunks=0, error=FAIL)] * 3))
    fallback = ScriptedTTS("f")
    tts = fallback_tts(primary, fallback)
    for _ in range(3):
        await collect_text(tts, "你好。")
    assert tts.breaker.open
    assert await collect_text(tts, "你好。") == [b"f\x00", b"f\x01"]
    assert await collect_text(tts, "Hello.", "en") == [b"f\x00", b"f\x01"]  # Latin letters too
    assert len(primary.calls) == 3

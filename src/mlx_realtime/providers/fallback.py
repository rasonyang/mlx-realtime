"""``FallbackTTS``: a primary TTS provider with a fallback, per segment (ARCHITECTURE §5.2, D13).

- The primary fails before its first chunk of a segment: the segment is retried on the
  fallback, which maps the session voice to its own voice for the segment's language.
- The primary fails after its audio started: the segment fails. The voice never switches in
  the middle of a sentence.
- ``overloaded`` is raised as is: both models run on the same backend, whose queue is full.
- After ``failures`` consecutive primary failures the circuit opens: for ``cooldown_s`` every
  segment goes straight to the fallback. The first segment after the cooldown tries the
  primary again; one more failure reopens the circuit.

Cancellation and ``aclose`` pass through to whichever stream is running.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing

from mlx_realtime.providers.base import (
    Lang,
    PCMChunk,
    ProviderError,
    Readiness,
    SpeechTiming,
    TTSBackend,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(
        self, failures: int, cooldown_s: float, *, now: Callable[[], float] = time.monotonic
    ) -> None:
        self._threshold = failures
        self._cooldown_s = cooldown_s
        self._now = now
        self._failures = 0
        self._open_until: float | None = None

    @property
    def open(self) -> bool:
        return self._open_until is not None and self._now() < self._open_until

    def failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and not self.open:
            self._open_until = self._now() + self._cooldown_s
            logger.warning(
                "tts circuit open: the fallback takes every segment",
                extra={"failures": self._failures, "cooldown_s": self._cooldown_s},
            )

    def success(self) -> None:
        if self._failures >= self._threshold:
            logger.info("tts circuit closed: the primary is back")
        self._failures = 0
        self._open_until = None


class FallbackTTS:
    def __init__(
        self,
        primary: TTSBackend,
        fallback: TTSBackend,
        *,
        failures: int,
        cooldown_s: float,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.breaker = CircuitBreaker(failures, cooldown_s, now=now)

    async def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        timing = timing or SpeechTiming()
        if not self.breaker.open:
            started = False
            try:
                primary = self.primary.stream(text, voice, lang, speed=speed, timing=timing)
                async with aclosing(primary):
                    async for chunk in primary:
                        started = True
                        yield chunk
            except ProviderError as error:
                if error.kind == "overloaded":
                    raise
                self.breaker.failure()
                if started:
                    raise
                logger.warning(
                    "primary tts failed before its first chunk; the fallback speaks the segment",
                    extra={"error": str(error), "kind": error.kind},
                )
            else:
                self.breaker.success()
                return
        fallback = self.fallback.stream(text, voice, lang, speed=speed, timing=timing)
        async with aclosing(fallback):
            async for chunk in fallback:
                yield chunk

    async def warmup(self) -> None:
        await self.primary.warmup()
        await self.fallback.warmup()

    async def ready(self) -> Readiness:
        primary, fallback = await self.primary.ready(), await self.fallback.ready()
        detail = f"primary: {primary.detail}; fallback: {fallback.detail}"
        if self.breaker.open:
            detail += "; circuit open"
        return Readiness(ready=primary.ready and fallback.ready, detail=detail)

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.fallback.aclose()

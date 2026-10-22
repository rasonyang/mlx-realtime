"""Mock providers: an echoing LLM, a sine-wave TTS and a scripted ASR (ARCHITECTURE §5.4, M1).

Every await point of a mock is an ``AwaitPoint``: a delay, a gate a test can hold, and an
error a test can inject. A test can therefore stop a pipeline at an exact phase, act, and
observe through ``StreamLog`` that the cancellation reached the provider.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

import numpy as np

from mlx_realtime.audio.pcm import CANONICAL_RATE, bytes_to_ms
from mlx_realtime.providers.base import (
    Delta,
    Lang,
    LLMDone,
    Message,
    PCMChunk,
    ProviderError,
    Readiness,
    TextDelta,
    ToolChoice,
    ToolDef,
    Transcript,
    Usage,
)
from mlx_realtime.text.lang import is_cjk


class AwaitPoint:
    """One await point of a mock stream."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.hits = 0
        self.waiting = 0
        self._gate = asyncio.Event()
        self._gate.set()
        self._arrived = asyncio.Event()
        self._error: ProviderError | None = None
        self._free_passes = 0

    def hold(self, *, after: int = 0) -> None:
        """Stop streams here until ``release``, once ``after`` more passes have gone through."""
        self._free_passes = after
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    def fail(self, error: ProviderError) -> None:
        """Make the next stream that passes this point raise ``error``."""
        self._error = error

    async def held(self) -> None:
        """Return once a stream is waiting at this (held) point."""
        await self._arrived.wait()

    async def __call__(self) -> None:
        self.hits += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self._free_passes > 0:
            self._free_passes -= 1
        else:
            await self._wait_at_gate()
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    async def _wait_at_gate(self) -> None:
        self.waiting += 1
        self._arrived.set()
        try:
            await self._gate.wait()
        finally:
            self.waiting -= 1
            if self.waiting == 0:
                self._arrived.clear()


class StreamLog:
    """Counts the streams (or requests) a mock opened and closed."""

    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return self.opened - self.closed

    def open(self) -> None:
        self.opened += 1
        self._idle.clear()

    def close(self) -> None:
        self.closed += 1
        if self.active == 0:
            self._idle.set()

    async def idle(self) -> None:
        """Return once every opened stream has closed."""
        await self._idle.wait()


class _MockBackend:
    """The ``Backend`` side of every mock: warmup can be held, readiness is immediate."""

    def __init__(self) -> None:
        self.warming = AwaitPoint()

    async def warmup(self) -> None:
        await self.warming()

    async def ready(self) -> Readiness:
        return Readiness(ready=True, detail="mock")

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDef, ...]
    tool_choice: ToolChoice
    max_output_tokens: int | None


class EchoLLM(_MockBackend):
    """Streams back the text of the last user message, one CJK character or word per delta.

    With no user message it streams ``greeting``. ``script`` replaces the next reply with
    given deltas, which is how tests obtain tool calls.
    """

    def __init__(
        self,
        *,
        first_token_delay_s: float = 0.0,
        token_delay_s: float = 0.0,
        greeting: str = "Hello.",
    ) -> None:
        super().__init__()
        self.first_token = AwaitPoint(first_token_delay_s)
        self.next_token = AwaitPoint(token_delay_s)
        self.streams = StreamLog()
        self.requests: list[LLMRequest] = []
        self._greeting = greeting
        self._scripted: deque[tuple[Delta, ...]] = deque()

    def script(self, *deltas: Delta) -> None:
        """Make the next stream yield exactly ``deltas``."""
        self._scripted.append(deltas)

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef],
        *,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
    ) -> AsyncGenerator[Delta]:
        self.requests.append(
            LLMRequest(tuple(messages), tuple(tools), tool_choice, max_output_tokens)
        )
        deltas = (
            self._scripted.popleft() if self._scripted else self._echo(messages, max_output_tokens)
        )
        self.streams.open()
        try:
            for index, delta in enumerate(deltas):
                await (self.first_token if index == 0 else self.next_token)()
                yield delta
        finally:
            self.streams.close()

    def _echo(self, messages: Sequence[Message], max_output_tokens: int | None) -> list[Delta]:
        users = [message.content for message in messages if message.role == "user"]
        tokens = split_tokens(users[-1] if users else self._greeting)
        finish = "stop"
        if max_output_tokens is not None and len(tokens) > max_output_tokens:
            tokens, finish = tokens[:max_output_tokens], "length"
        prompt_tokens = sum(len(split_tokens(message.content)) for message in messages)
        usage = Usage(input_tokens=prompt_tokens, output_tokens=len(tokens))
        return [*(TextDelta(token) for token in tokens), LLMDone(finish, usage)]


def split_tokens(text: str) -> list[str]:
    """One token per CJK character or Latin word; whitespace goes with the word after it."""
    tokens: list[str] = []
    start = 0
    for index in range(1, len(text)):
        char, previous = text[index], text[index - 1]
        if char.isspace():
            boundary = not previous.isspace()
        else:
            boundary = not previous.isspace() and (is_cjk(char) or is_cjk(previous))
        if boundary:
            tokens.append(text[start:index])
            start = index
    if text:
        tokens.append(text[start:])
    return tokens


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    text: str
    voice: str
    lang: Lang


class SineTTS(_MockBackend):
    """A sine tone whose length is ``ms_per_char`` per non-space character, in chunks."""

    def __init__(
        self,
        *,
        ms_per_char: int = 60,
        chunk_ms: int = 80,
        first_chunk_delay_s: float = 0.0,
        chunk_delay_s: float = 0.0,
        frequency_hz: float = 440.0,
        sample_rate: int = 24_000,
    ) -> None:
        super().__init__()
        self.first_chunk = AwaitPoint(first_chunk_delay_s)
        self.next_chunk = AwaitPoint(chunk_delay_s)
        self.streams = StreamLog()
        self.requests: list[SpeechRequest] = []
        self.sample_rate = sample_rate
        self._ms_per_char = ms_per_char
        self._chunk_samples = max(chunk_ms * sample_rate // 1000, 1)
        self._frequency = frequency_hz

    def samples_for(self, text: str) -> int:
        characters = sum(1 for char in text if not char.isspace())
        return characters * self._ms_per_char * self.sample_rate // 1000

    async def stream(
        self, text: str, voice: str, lang: Lang, *, speed: float = 1.0
    ) -> AsyncGenerator[PCMChunk]:
        self.requests.append(SpeechRequest(text, voice, lang))
        total = self.samples_for(text)
        self.streams.open()
        try:
            produced = 0
            while produced < total:
                await (self.first_chunk if produced == 0 else self.next_chunk)()
                count = min(self._chunk_samples, total - produced)
                yield PCMChunk(self._tone(produced, count), self.sample_rate)
                produced += count
        finally:
            self.streams.close()

    def _tone(self, start: int, count: int) -> bytes:
        times = np.arange(start, start + count, dtype=np.float64) / self.sample_rate
        wave = 0.2 * 32767 * np.sin(2 * math.pi * self._frequency * times)
        return wave.astype("<i2").tobytes()


class ScriptedASR(_MockBackend):
    """Returns the scripted transcripts in order, cycling; an empty script yields ``""``."""

    def __init__(self, transcripts: Sequence[str] = (), *, delay_s: float = 0.0) -> None:
        super().__init__()
        self.result = AwaitPoint(delay_s)
        self.calls = StreamLog()
        self.requests: list[int] = []
        self._script = list(transcripts)
        self._next = 0

    async def transcribe(self, pcm16_16k: bytes, *, language: str | None = None) -> Transcript:
        audio_ms = bytes_to_ms(len(pcm16_16k), CANONICAL_RATE)
        self.requests.append(audio_ms)
        self.calls.open()
        try:
            await self.result()
        finally:
            self.calls.close()
        text = self._script[self._next % len(self._script)] if self._script else ""
        self._next += 1
        return Transcript(text=text, language=language, audio_ms=audio_ms)

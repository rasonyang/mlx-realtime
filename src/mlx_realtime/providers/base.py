"""Provider interfaces, value types and errors (ARCHITECTURE §5.1, §5.2).

Contracts every provider keeps:

- Cancellation: the only mechanism is cancelling the consuming task or closing the iterator.
  A provider lets ``CancelledError`` propagate, aborts its backend request in ``finally``
  without awaiting anything that can block, and yields nothing after cancellation.
- Deadlines: every request has connect, first-item, idle and total deadlines; expiry raises
  ``ProviderTimeout``. Nothing waits without a deadline.
- Errors: providers raise ``ProviderError(kind, retryable)``. They never emit protocol events
  or touch session state; the session maps errors to the protocol (ARCHITECTURE §8).
- Backend quirks stay inside the provider that has them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Lang = Literal["zh", "en"]
ToolChoice = Literal["auto", "none"]
"""``required`` and forced functions stay unsupported until M5 decides (ARCHITECTURE §5.2)."""

FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
ProviderErrorKind = Literal["unavailable", "timeout", "overloaded", "bad_response", "internal"]
TimeoutStage = Literal["connect", "first_item", "idle", "total"]


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None  # if the model reports it
    audio_ms: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str  # never contains tool-call markup or reasoning


@dataclass(frozen=True, slots=True)
class ToolCall:  # always complete; fragments are assembled inside the provider
    call_id: str  # provider-side id; the protocol layer mints the wire call_id
    name: str
    arguments: str  # raw JSON text; the gateway never parses it


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMDone:
    finish_reason: FinishReason
    usage: Usage | None


type Delta = TextDelta | ToolCall | LLMDone


@dataclass(frozen=True, slots=True)
class PCMChunk:
    pcm: bytes  # s16le mono
    sample_rate: int


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message; providers adapt the list to what their backend accepts."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages that called tools
    tool_call_id: str | None = None  # tool messages: the provider-side id of the call answered


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str | None
    parameters: Mapping[str, object] | None  # JSON Schema, passed through verbatim


@dataclass(frozen=True, slots=True)
class Readiness:
    ready: bool
    detail: str = ""


class ProviderError(Exception):
    def __init__(self, kind: ProviderErrorKind, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.kind: ProviderErrorKind = kind
        self.retryable = retryable


class ProviderTimeout(ProviderError):
    def __init__(self, stage: TimeoutStage, message: str | None = None) -> None:
        super().__init__("timeout", message or f"{stage} deadline expired", retryable=True)
        self.stage: TimeoutStage = stage


class ASRProvider(Protocol):
    async def transcribe(self, pcm16_16k: bytes, *, language: str | None = None) -> Transcript: ...


class LLMProvider(Protocol):
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef],
        *,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[Delta]: ...


class TTSProvider(Protocol):
    def stream(
        self, text: str, voice: str, lang: Lang, *, speed: float = 1.0
    ) -> AsyncIterator[PCMChunk]: ...


class Backend(Protocol):  # every provider also implements this
    async def warmup(self) -> None: ...
    async def ready(self) -> Readiness: ...
    async def aclose(self) -> None: ...

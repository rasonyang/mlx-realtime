"""Provider interfaces, value types and errors (ARCHITECTURE §5.1, §5.2).

Contracts every provider keeps:

- Cancellation: the only mechanism is cancelling the consuming task or closing the generator
  (``aclose``). A provider lets ``CancelledError`` propagate, aborts its backend request in
  ``finally`` without awaiting anything that can block, and yields nothing after cancellation.
- Deadlines: every request has connect, first-item, idle and total deadlines; expiry raises
  ``ProviderTimeout``. Nothing waits without a deadline.
- Errors: providers raise ``ProviderError(kind, retryable)``. They never emit protocol events
  or touch session state; the session maps errors to the protocol (ARCHITECTURE §8).
- Backend quirks stay inside the provider that has them.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Lang = Literal["zh", "en"]
ToolChoice = Literal["auto", "none"]
"""``none`` sends no tools. ``required`` and forced functions are rejected by the protocol layer:
mlx_lm.server ignores ``tool_choice``, so they cannot be enforced (M5)."""

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


@dataclass(slots=True)
class SpeechTiming:
    """What a TTS provider reports about one stream while it runs (the turn timeline).

    The caller passes a fresh instance to ``TTSProvider.stream``; the provider fills it in.
    """

    queued_s: float = 0.0  # time spent waiting for a backend slot, over every attempt
    model: str = ""  # the model that produced the audio
    fallback: bool = False  # the stream was served by the fallback model
    primary_error: ProviderErrorKind | None = None  # why the primary failed, if the fallback spoke
    # Set by the caller (M6). ``deadline``: when, on the session clock, the session runs out of
    # audio without this segment; a shared queue serves the earliest first. ``slack_s``: how
    # far ahead of now that deadline lies, so a queue can let the request wait that much
    # longer before giving up.
    deadline: float | None = None
    slack_s: float = 0.0


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
    ) -> AsyncGenerator[Delta]: ...


class TTSProvider(Protocol):
    """``voice`` is a session voice name (``audio.output.voice``, D19); the provider maps it to
    what each model takes. ``lang`` is the segment's language tag."""

    def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]: ...


class Backend(Protocol):  # every provider also implements this
    async def warmup(self) -> None: ...
    async def ready(self) -> Readiness: ...
    async def aclose(self) -> None: ...


class LLMBackend(LLMProvider, Backend, Protocol): ...


class ASRBackend(ASRProvider, Backend, Protocol): ...


class TTSBackend(TTSProvider, Backend, Protocol): ...


@runtime_checkable
class Queued(Protocol):
    """A provider that queues requests for its backend, and can say how many wait now."""

    @property
    def queue_depth(self) -> int: ...


def queue_depth(provider: object) -> int:
    """The requests waiting in ``provider``'s queue now (0 for a provider without one)."""
    return provider.queue_depth if isinstance(provider, Queued) else 0

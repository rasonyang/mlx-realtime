"""Signals: results that session-owned tasks (the VAD feeder, end-of-turn evaluations, ASR
requests, response pipelines) post to the actor.

ARCHITECTURE §4.2 calls these facts too; the code keeps "facts" for what the actor emits to the
protocol layer, so the two directions read apart. Pipeline signals carry their response's
generation and turn-detection signals their VAD epoch; the actor drops stale ones at ingress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_realtime.providers.base import LLMDone, ProviderError, ToolCall, Transcript
from mlx_realtime.vad.gate import GateEvent


@dataclass(frozen=True, slots=True)
class VadUpdate:
    """What the speech gate found in the audio the feeder processed since its last update."""

    epoch: int
    events: tuple[GateEvent, ...]
    keep_from_ms: int  # earliest input position a turn not yet ended can start at
    # The echo canceller's output for this stretch of the input timeline (D17), if it ran.
    processed_start: int = 0  # input timeline sample
    processed: bytes = b""


@dataclass(frozen=True, slots=True)
class TurnVerdict:
    epoch: int
    evaluation: int
    probability: float  # that the turn is complete


@dataclass(frozen=True, slots=True)
class TranscriptReady:
    item_id: str
    transcript: Transcript


@dataclass(frozen=True, slots=True)
class TranscriptFailed:
    item_id: str
    error: ProviderError


@dataclass(frozen=True, slots=True)
class LlmText:
    generation: int
    text: str


@dataclass(frozen=True, slots=True)
class LlmToolCall:
    generation: int
    call: ToolCall


@dataclass(frozen=True, slots=True)
class LlmFinished:
    generation: int
    done: LLMDone


@dataclass(frozen=True, slots=True)
class SegmentStarted:
    """Synthesis of a segment produced its first audio (or finished without any)."""

    generation: int
    index: int
    text: str
    spoken: str
    source: tuple[int, ...] | None = None  # spoken character -> text index (normalize_with_map)


@dataclass(frozen=True, slots=True)
class SegmentAudio:
    generation: int
    index: int
    pcm: bytes  # in the response's output format, at most max_delta_ms


@dataclass(frozen=True, slots=True)
class SpeechFinished:
    """Every segment of the response has been synthesized."""

    generation: int


@dataclass(frozen=True, slots=True)
class PipelineFailed:
    generation: int
    stage: Literal["llm", "tts", "pipeline"]  # "pipeline": an internal error, not a backend's
    error: ProviderError


@dataclass(frozen=True, slots=True)
class PipelineMark:
    """A turn-timeline mark the pipeline took (``first_sentence_ready``, ``tts_first_chunk``),
    with values logged next to the marks (``tts_queue_ms``, ``tts_model``)."""

    generation: int
    name: str
    at_s: float
    values: tuple[tuple[str, object], ...] = ()


type PipelineSignal = (
    LlmText
    | LlmToolCall
    | LlmFinished
    | SegmentStarted
    | SegmentAudio
    | SpeechFinished
    | PipelineFailed
    | PipelineMark
)
type Signal = VadUpdate | TurnVerdict | TranscriptReady | TranscriptFailed | PipelineSignal

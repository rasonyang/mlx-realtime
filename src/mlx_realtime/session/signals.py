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


@dataclass(frozen=True, slots=True)
class SegmentAudio:
    generation: int
    index: int
    pcm: bytes  # audio/pcm, 24 kHz, at most max_delta_ms


@dataclass(frozen=True, slots=True)
class SpeechFinished:
    """Every segment of the response has been synthesized."""

    generation: int


@dataclass(frozen=True, slots=True)
class PipelineFailed:
    generation: int
    stage: Literal["llm", "tts", "pipeline"]  # "pipeline": an internal error, not a backend's
    error: ProviderError


type PipelineSignal = (
    LlmText
    | LlmToolCall
    | LlmFinished
    | SegmentStarted
    | SegmentAudio
    | SpeechFinished
    | PipelineFailed
)
type Signal = VadUpdate | TurnVerdict | TranscriptReady | TranscriptFailed | PipelineSignal

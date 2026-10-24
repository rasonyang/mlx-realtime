"""The effective session configuration and the merge rules of ``session.update``.

A ``session.update`` is validated as a whole by the protocol layer, which hands the actor a
``SessionPatch``. Only the fields present in the update change; nested objects merge field by
field; ``null`` clears a nullable field. A nested object that was ``null`` is rebuilt from the
server's session defaults, so its absent fields take the configured values.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Literal

from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.providers.base import ToolChoice, ToolDef

type Modality = Literal["audio", "text"]
type SessionType = Literal["realtime", "transcription"]


class Unset(enum.Enum):
    UNSET = enum.auto()


UNSET = Unset.UNSET
type Maybe[T] = T | Literal[Unset.UNSET]


@dataclass(frozen=True, slots=True)
class ServerVad:
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200
    create_response: bool = True
    interrupt_response: bool = True


@dataclass(frozen=True, slots=True)
class Transcription:
    model: str | None = None
    language: str | None = None
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class SessionSettings:
    type: SessionType = "realtime"
    model: str = "mlx-realtime"
    instructions: str = ""
    output_modality: Modality = "audio"
    input_format: AudioFormat = PCM
    output_format: AudioFormat = PCM
    transcription: Transcription | None = Transcription()
    turn_detection: ServerVad | None = ServerVad()
    voice: str = "default"
    tools: tuple[ToolDef, ...] = ()
    tool_choice: ToolChoice = "auto"
    max_output_tokens: int | None = None  # None is "inf"

    def apply(self, patch: SessionPatch, defaults: SessionSettings) -> SessionSettings:
        """The settings after ``patch``; ``defaults`` seeds nested objects that were null."""
        return SessionSettings(
            type=_pick(patch.type, self.type),
            model=_pick(patch.model, self.model),
            instructions=_pick(patch.instructions, self.instructions),
            output_modality=_pick(patch.output_modality, self.output_modality),
            input_format=_pick(patch.input_format, self.input_format),
            output_format=_pick(patch.output_format, self.output_format),
            transcription=_merge_transcription(
                patch.transcription, self.transcription, defaults.transcription
            ),
            turn_detection=_merge_vad(
                patch.turn_detection, self.turn_detection, defaults.turn_detection
            ),
            voice=_pick(patch.voice, self.voice),
            tools=_pick(patch.tools, self.tools),
            tool_choice=_pick(patch.tool_choice, self.tool_choice),
            max_output_tokens=_pick(patch.max_output_tokens, self.max_output_tokens),
        )


def settings_for(
    defaults: SessionSettings, session_type: SessionType, model: str
) -> SessionSettings:
    """A session's settings as its type starts: the configured defaults, except that a
    transcription session has no turn detection (the client commits each utterance). It runs no
    model, so ``model`` is only kept for a switch to ``realtime``."""
    if session_type == "realtime":
        return replace(defaults, type="realtime", model=model)
    return replace(defaults, type="transcription", model=model, turn_detection=None)


@dataclass(frozen=True, slots=True)
class ServerVadPatch:
    threshold: Maybe[float] = UNSET
    prefix_padding_ms: Maybe[int] = UNSET
    silence_duration_ms: Maybe[int] = UNSET
    create_response: Maybe[bool] = UNSET
    interrupt_response: Maybe[bool] = UNSET


@dataclass(frozen=True, slots=True)
class TranscriptionPatch:
    model: Maybe[str] = UNSET
    language: Maybe[str] = UNSET
    prompt: Maybe[str] = UNSET


@dataclass(frozen=True, slots=True)
class SessionPatch:
    type: Maybe[SessionType] = UNSET
    model: Maybe[str] = UNSET
    instructions: Maybe[str] = UNSET
    output_modality: Maybe[Modality] = UNSET
    input_format: Maybe[AudioFormat] = UNSET
    output_format: Maybe[AudioFormat] = UNSET
    transcription: Maybe[TranscriptionPatch | None] = UNSET
    turn_detection: Maybe[ServerVadPatch | None] = UNSET
    voice: Maybe[str] = UNSET
    tools: Maybe[tuple[ToolDef, ...]] = UNSET
    tool_choice: Maybe[ToolChoice] = UNSET
    max_output_tokens: Maybe[int | None] = UNSET


def _pick[T](value: Maybe[T], current: T) -> T:
    return current if value is UNSET else value


def _merge_vad(
    patch: Maybe[ServerVadPatch | None], current: ServerVad | None, default: ServerVad | None
) -> ServerVad | None:
    if patch is UNSET:
        return current
    if patch is None:
        return None
    base = current or default or ServerVad()
    return replace(
        base,
        threshold=_pick(patch.threshold, base.threshold),
        prefix_padding_ms=_pick(patch.prefix_padding_ms, base.prefix_padding_ms),
        silence_duration_ms=_pick(patch.silence_duration_ms, base.silence_duration_ms),
        create_response=_pick(patch.create_response, base.create_response),
        interrupt_response=_pick(patch.interrupt_response, base.interrupt_response),
    )


def _merge_transcription(
    patch: Maybe[TranscriptionPatch | None],
    current: Transcription | None,
    default: Transcription | None,
) -> Transcription | None:
    if patch is UNSET:
        return current
    if patch is None:
        return None
    base = current or default or Transcription()
    return Transcription(
        model=_pick(patch.model, base.model),
        language=_pick(patch.language, base.language),
        prompt=_pick(patch.prompt, base.prompt),
    )

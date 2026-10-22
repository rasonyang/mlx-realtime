"""Commands: validated client events, as the protocol layer hands them to the actor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from mlx_realtime.providers.base import ToolChoice
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.settings import UNSET, Maybe, Modality, SessionPatch


@dataclass(frozen=True, slots=True)
class AppendAudio:
    pcm: bytes  # audio/pcm, 24 kHz s16le, whole samples


@dataclass(frozen=True, slots=True)
class CommitAudio:
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ClearAudio:
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class UpdateSession:
    patch: SessionPatch
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class NewMessage:
    role: Literal["system", "user", "assistant"]
    texts: tuple[str, ...]
    id: str | None = None


@dataclass(frozen=True, slots=True)
class NewFunctionCallOutput:
    call_id: str
    output: str
    id: str | None = None


@dataclass(frozen=True, slots=True)
class CreateItem:
    item: NewMessage | NewFunctionCallOutput
    previous_item_id: str | None = None  # None appends; ROOT inserts first
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class TruncateItem:
    item_id: str
    content_index: int
    audio_end_ms: int
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class DeleteItem:
    item_id: str
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseParams:
    """Per-response overrides of ``response.create``; unset fields take the session's."""

    instructions: str | None = None
    output_modality: Modality | None = None
    max_output_tokens: Maybe[int | None] = UNSET
    tool_choice: ToolChoice | None = None
    voice: str | None = None
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class CreateResponse:
    params: ResponseParams = ResponseParams()
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class CancelResponse:
    response_id: str | None = None
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class RejectEvent:
    """A client event that failed validation; the actor reports it in order."""

    error: ErrorInfo


type Command = (
    AppendAudio
    | CommitAudio
    | ClearAudio
    | UpdateSession
    | CreateItem
    | TruncateItem
    | DeleteItem
    | CreateResponse
    | CancelResponse
    | RejectEvent
)

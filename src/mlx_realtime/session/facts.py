"""Facts: what the actor reports, for the protocol layer to encode as GA server events.

A fact may expand to several events (for instance ``MessageOpened`` to ``output_item.added``,
``conversation.item.added`` and ``content_part.added``); their order is the protocol layer's
knowledge. Facts may reference live session objects: they are encoded synchronously, inside
the actor step that emits them.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_realtime.session.conversation import AssistantMessage, AudioMessage, FunctionCall, Item
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.output import Delivery
from mlx_realtime.session.response import Response
from mlx_realtime.session.settings import SessionSettings


@dataclass(frozen=True, slots=True)
class SessionStarted:
    session_id: str
    conversation_id: str
    settings: SessionSettings


@dataclass(frozen=True, slots=True)
class SessionUpdated:
    session_id: str
    settings: SessionSettings


@dataclass(frozen=True, slots=True)
class ErrorRaised:
    error: ErrorInfo


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    item_id: str  # the item the commit will create
    audio_start_ms: int


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    item_id: str
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class AudioCommitted:
    item: AudioMessage
    previous_item_id: str | None


@dataclass(frozen=True, slots=True)
class AudioCleared:
    pass


@dataclass(frozen=True, slots=True)
class ItemCreated:
    """A client-created item, complete on arrival."""

    item: Item
    previous_item_id: str | None


@dataclass(frozen=True, slots=True)
class TranscriptCompleted:
    item: AudioMessage
    previous_item_id: str | None


@dataclass(frozen=True, slots=True)
class TranscriptionFailed:
    item: AudioMessage
    previous_item_id: str | None
    error: ErrorInfo


@dataclass(frozen=True, slots=True)
class ItemDeleted:
    item_id: str


@dataclass(frozen=True, slots=True)
class ItemTruncated:
    item_id: str
    audio_end_ms: int


@dataclass(frozen=True, slots=True)
class ResponseStarted:
    response: Response


@dataclass(frozen=True, slots=True)
class MessageOpened:
    response: Response
    message: AssistantMessage
    previous_item_id: str | None


@dataclass(frozen=True, slots=True)
class OutputTextDelta:
    response: Response
    message: AssistantMessage
    text: str


@dataclass(frozen=True, slots=True)
class OutputTranscriptDelta:
    response: Response
    message: AssistantMessage
    text: str
    delivery: Delivery


@dataclass(frozen=True, slots=True)
class OutputAudioDelta:
    response: Response
    message: AssistantMessage
    pcm: bytes
    delivery: Delivery


@dataclass(frozen=True, slots=True)
class MessageClosed:
    response: Response
    message: AssistantMessage
    previous_item_id: str | None
    in_conversation: bool  # False once the client deleted the item


@dataclass(frozen=True, slots=True)
class FunctionCallDelivered:
    response: Response
    call: FunctionCall
    previous_item_id: str | None


@dataclass(frozen=True, slots=True)
class ResponseFinished:
    response: Response


type Fact = (
    SessionStarted
    | SessionUpdated
    | ErrorRaised
    | SpeechStarted
    | SpeechStopped
    | AudioCommitted
    | AudioCleared
    | ItemCreated
    | TranscriptCompleted
    | TranscriptionFailed
    | ItemDeleted
    | ItemTruncated
    | ResponseStarted
    | MessageOpened
    | OutputTextDelta
    | OutputTranscriptDelta
    | OutputAudioDelta
    | MessageClosed
    | FunctionCallDelivered
    | ResponseFinished
)

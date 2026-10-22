"""Facts to GA server events, in the orders documented in docs/PROTOCOL.md.

``conversation.item.created`` is never emitted (D1): items are answered with
``conversation.item.added`` and ``conversation.item.done``.
"""

from __future__ import annotations

import base64

from mlx_realtime.protocol.session import WIRE_FORMAT, render_max_output_tokens, render_session
from mlx_realtime.providers.base import Usage
from mlx_realtime.session.conversation import (
    AssistantMessage,
    AudioMessage,
    FunctionCall,
    FunctionCallOutput,
    Item,
    TextMessage,
)
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.facts import (
    AudioCleared,
    AudioCommitted,
    ErrorRaised,
    Fact,
    FunctionCallDelivered,
    ItemCreated,
    ItemDeleted,
    ItemTruncated,
    MessageClosed,
    MessageOpened,
    OutputAudioDelta,
    OutputTextDelta,
    OutputTranscriptDelta,
    ResponseFinished,
    ResponseStarted,
    SessionStarted,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    TranscriptCompleted,
    TranscriptionFailed,
)
from mlx_realtime.session.response import Response, StatusDetails

type Event = dict[str, object]


class EventEncoder:
    """Encodes one session's facts; event ids are ``event_<n>``, increasing within the session."""

    def __init__(self) -> None:
        self._count = 0

    def encode(self, fact: Fact) -> list[Event]:
        return [self._stamp(event) for event in _events(fact)]

    def error(self, error: ErrorInfo) -> Event:
        """An ``error`` event outside any session (admission) or ending one (fatal errors)."""
        return self._stamp(_error_event(error))

    def _stamp(self, event: Event) -> Event:
        self._count += 1
        return {"type": event.pop("type"), "event_id": f"event_{self._count}", **event}


def _events(fact: Fact) -> list[Event]:
    match fact:
        case SessionStarted(
            session_id=session_id, conversation_id=conversation_id, settings=settings
        ):
            conversation = {"id": conversation_id, "object": "realtime.conversation"}
            return [
                {"type": "session.created", "session": render_session(session_id, settings)},
                {"type": "conversation.created", "conversation": conversation},
            ]
        case SessionUpdated(session_id=session_id, settings=settings):
            return [{"type": "session.updated", "session": render_session(session_id, settings)}]
        case ErrorRaised(error=error):
            return [_error_event(error)]
        case SpeechStarted(item_id=item_id, audio_start_ms=audio_start_ms):
            return [
                {
                    "type": "input_audio_buffer.speech_started",
                    "audio_start_ms": audio_start_ms,
                    "item_id": item_id,
                }
            ]
        case SpeechStopped(item_id=item_id, audio_end_ms=audio_end_ms):
            return [
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "audio_end_ms": audio_end_ms,
                    "item_id": item_id,
                }
            ]
        case AudioCommitted(item=item, previous_item_id=previous):
            return [
                {
                    "type": "input_audio_buffer.committed",
                    "previous_item_id": previous,
                    "item_id": item.id,
                },
                _item_event("conversation.item.added", item, previous),
            ]
        case AudioCleared():
            return [{"type": "input_audio_buffer.cleared"}]
        case ItemCreated(item=item, previous_item_id=previous):
            return [
                _item_event("conversation.item.added", item, previous),
                _item_event("conversation.item.done", item, previous),
            ]
        case TranscriptCompleted(item=item, previous_item_id=previous):
            events: list[Event] = []
            if item.show_transcript:
                events.append(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item.id,
                        "content_index": 0,
                        "transcript": item.transcript or "",
                        "usage": {"type": "duration", "seconds": item.audio_ms / 1000},
                    }
                )
            return [*events, _item_event("conversation.item.done", item, previous)]
        case TranscriptionFailed(item=item, previous_item_id=previous, error=error):
            events = []
            if item.show_transcript:
                events.append(
                    {
                        "type": "conversation.item.input_audio_transcription.failed",
                        "item_id": item.id,
                        "content_index": 0,
                        "error": {
                            "type": error.type,
                            "code": error.code,
                            "message": error.message,
                            "param": error.param,
                        },
                    }
                )
            return [*events, _item_event("conversation.item.done", item, previous)]
        case ItemDeleted(item_id=item_id):
            return [{"type": "conversation.item.deleted", "item_id": item_id}]
        case ItemTruncated(item_id=item_id, audio_end_ms=audio_end_ms):
            return [
                {
                    "type": "conversation.item.truncated",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": audio_end_ms,
                }
            ]
        case ResponseStarted(response=response):
            return [{"type": "response.created", "response": _render_response(response)}]
        case MessageOpened(response=response, message=message, previous_item_id=previous):
            part: Event = (
                {"type": "audio", "transcript": ""}
                if message.modality == "audio"
                else {"type": "text", "text": ""}
            )
            return [
                _output_item("response.output_item.added", response, message, opened=True),
                _item_event("conversation.item.added", message, previous, opened=True),
                {
                    "type": "response.content_part.added",
                    **_part_ref(response, message),
                    "part": part,
                },
            ]
        case OutputTextDelta(response=response, message=message, text=text):
            return [
                {
                    "type": "response.output_text.delta",
                    **_part_ref(response, message),
                    "delta": text,
                }
            ]
        case OutputTranscriptDelta(response=response, message=message, text=text):
            return [
                {
                    "type": "response.output_audio_transcript.delta",
                    **_part_ref(response, message),
                    "delta": text,
                }
            ]
        case OutputAudioDelta(response=response, message=message, pcm=pcm):
            delta = base64.b64encode(pcm).decode("ascii")
            return [
                {
                    "type": "response.output_audio.delta",
                    **_part_ref(response, message),
                    "delta": delta,
                }
            ]
        case MessageClosed():
            return _message_closed(fact)
        case FunctionCallDelivered():
            return _function_call(fact)
        case ResponseFinished(response=response):
            return [{"type": "response.done", "response": _render_response(response)}]


def _message_closed(fact: MessageClosed) -> list[Event]:
    response, message = fact.response, fact.message
    ref = _part_ref(response, message)
    if message.modality == "audio":
        events: list[Event] = [
            {"type": "response.output_audio.done", **ref},
            {"type": "response.output_audio_transcript.done", **ref, "transcript": message.text},
        ]
        part: Event = {"type": "audio", "transcript": message.text}
    else:
        events = [{"type": "response.output_text.done", **ref, "text": message.text}]
        part = {"type": "text", "text": message.text}
    events.append({"type": "response.content_part.done", **ref, "part": part})
    events.append(_output_item("response.output_item.done", response, message))
    if fact.in_conversation:
        events.append(_item_event("conversation.item.done", message, fact.previous_item_id))
    return events


def _function_call(fact: FunctionCallDelivered) -> list[Event]:
    """A complete call, delivered at once: its arguments arrive in a single delta (§6.7)."""
    response, call, previous = fact.response, fact.call, fact.previous_item_id
    ref: Event = {
        "response_id": response.id,
        "item_id": call.id,
        "output_index": response.output_index(call),
        "call_id": call.call_id,
    }
    return [
        _output_item("response.output_item.added", response, call, opened=True),
        _item_event("conversation.item.added", call, previous, opened=True),
        {"type": "response.function_call_arguments.delta", **ref, "delta": call.arguments},
        {
            "type": "response.function_call_arguments.done",
            **ref,
            "name": call.name,
            "arguments": call.arguments,
        },
        _output_item("response.output_item.done", response, call),
        _item_event("conversation.item.done", call, previous),
    ]


def render_item(item: Item, *, opened: bool = False) -> Event:
    """The GA item; ``opened`` renders a response item as ``*.added`` shows it: no content yet."""
    base: Event = {"id": item.id, "object": "realtime.item"}
    match item:
        case TextMessage(role=role, texts=texts, status=status):
            part_type = "output_text" if role == "assistant" else "input_text"
            content = [{"type": part_type, "text": text} for text in texts]
            return {**base, "type": "message", "status": status, "role": role, "content": content}
        case AudioMessage(status=status):
            transcript = item.transcript if item.show_transcript else None
            content = [{"type": "input_audio", "transcript": transcript}]
            return {**base, "type": "message", "status": status, "role": "user", "content": content}
        case AssistantMessage(modality=modality, text=text, status=status):
            if opened:
                return {
                    **base,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
            part: Event = (
                {"type": "output_audio", "transcript": text}
                if modality == "audio"
                else {"type": "output_text", "text": text}
            )
            return {
                **base,
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [part],
            }
        case FunctionCall(call_id=call_id, name=name, arguments=arguments, status=status):
            return {
                **base,
                "type": "function_call",
                "status": "in_progress" if opened else status,
                "name": name,
                "call_id": call_id,
                "arguments": "" if opened else arguments,
            }
        case FunctionCallOutput(call_id=call_id, output=output, status=status):
            return {
                **base,
                "type": "function_call_output",
                "status": status,
                "call_id": call_id,
                "output": output,
            }


def _output_item(
    event_type: str,
    response: Response,
    item: AssistantMessage | FunctionCall,
    *,
    opened: bool = False,
) -> Event:
    return {
        "type": event_type,
        "response_id": response.id,
        "output_index": response.output_index(item),
        "item": render_item(item, opened=opened),
    }


def _item_event(
    event_type: str, item: Item, previous: str | None, *, opened: bool = False
) -> Event:
    return {
        "type": event_type,
        "previous_item_id": previous,
        "item": render_item(item, opened=opened),
    }


def _part_ref(response: Response, message: AssistantMessage) -> Event:
    return {
        "response_id": response.id,
        "item_id": message.id,
        "output_index": response.output_index(message),
        "content_index": 0,
    }


def _error_event(error: ErrorInfo) -> Event:
    return {
        "type": "error",
        "error": {
            "type": error.type,
            "code": error.code,
            "message": error.message,
            "param": error.param,
            "event_id": error.event_id,
        },
    }


def _render_response(response: Response) -> Event:
    return {
        "object": "realtime.response",
        "id": response.id,
        "status": response.status,
        "status_details": _render_status(response.status_details),
        "output": [render_item(item) for item in response.output],
        "conversation_id": response.conversation_id,
        "output_modalities": [response.modality],
        "max_output_tokens": render_max_output_tokens(response.max_output_tokens),
        "audio": {"output": {"format": dict(WIRE_FORMAT), "voice": response.voice}},
        "usage": _render_usage(response.usage),
        "metadata": None if response.metadata is None else dict(response.metadata),
    }


def _render_status(details: StatusDetails | None) -> Event | None:
    if details is None:
        return None
    rendered: Event = {"type": details.type}
    if details.reason is not None:
        rendered["reason"] = details.reason
    if details.error is not None:
        error = details.error
        rendered["error"] = {"type": error.type, "code": error.code, "message": error.message}
    return rendered


def _render_usage(usage: Usage | None) -> Event | None:
    if usage is None:
        return None
    return {
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "input_token_details": {
            "text_tokens": usage.input_tokens,
            "audio_tokens": 0,
            "cached_tokens": usage.cached_tokens,
        },
        "output_token_details": {"text_tokens": usage.output_tokens, "audio_tokens": 0},
    }

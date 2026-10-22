"""Every fact encodes to events the installed SDK parses with no missing or unknown fields."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mlx_realtime.protocol.encode import EventEncoder
from mlx_realtime.providers.base import ToolDef, Usage
from mlx_realtime.session.conversation import (
    AssistantMessage,
    AudioMessage,
    FunctionCall,
    FunctionCallOutput,
    SpokenSegment,
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
from mlx_realtime.session.output import Delivery
from mlx_realtime.session.response import Response, StatusDetails
from mlx_realtime.session.settings import ServerVad, SessionSettings, Transcription
from tests.support import Event, sdk_parse

SETTINGS = SessionSettings(
    model="m",
    instructions="Be brief.",
    transcription=Transcription(model="qwen3-asr", language="zh"),
    turn_detection=ServerVad(),
    voice="vivian",
    tools=(ToolDef("lookup", "Look up.", {"type": "object"}), ToolDef("bare", None, None)),
    max_output_tokens=100,
)


def response(modality: str = "audio") -> Response:
    return Response(
        id="resp_1",
        generation=1,
        conversation_id="conv_1",
        modality="audio" if modality == "audio" else "text",
        voice="vivian",
        instructions="",
        max_output_tokens=None,
        tool_choice="auto",
        metadata={"k": "v"},
        turn_item_id=None,
    )


def facts() -> Iterator[Fact]:
    error = ErrorInfo("invalid_request_error", "invalid_value", "bad", "session.foo", "c1")
    yield SessionStarted("sess_1", "conv_1", SETTINGS)
    yield SessionUpdated("sess_1", SessionSettings(transcription=None, turn_detection=None))
    yield ErrorRaised(error)
    yield ErrorRaised(ErrorInfo("server_error", "provider_error", "llm failed"))
    yield SpeechStarted("item_1", 120)
    yield SpeechStopped("item_1", 900)
    audio_item = AudioMessage("item_1", 780, show_transcript=True)
    yield AudioCommitted(audio_item, None)
    yield AudioCleared()
    yield ItemCreated(TextMessage("item_2", "user", ["hi", "there"]), "item_1")
    yield ItemCreated(TextMessage("item_3", "assistant", ["ok"]), "item_2")
    yield ItemCreated(TextMessage("item_4", "system", ["note"]), None)
    yield ItemCreated(FunctionCallOutput("item_5", "call_1", '{"a":1}'), "item_4")
    audio_item.transcript, audio_item.status = "你好", "completed"
    yield TranscriptCompleted(audio_item, None)
    yield TranscriptCompleted(
        AudioMessage("item_6", 100, show_transcript=False, transcript=""), None
    )
    failed = AudioMessage("item_7", 100, show_transcript=True, status="completed")
    yield TranscriptionFailed(failed, "item_6", ErrorInfo("server_error", "backend_timeout", "asr"))
    yield ItemDeleted("item_7")
    yield ItemTruncated("item_8", 1500)

    spoken = response("audio")
    yield ResponseStarted(spoken)
    message = AssistantMessage("item_8", "audio")
    spoken.output.append(message)
    spoken.message = message
    yield MessageOpened(spoken, message, "item_1")
    message.segments.append(SpokenSegment("你好。", "你好。", 0, 4800))
    message.text = "你好。"
    yield OutputTranscriptDelta(spoken, message, "你好。", Delivery(1, "item_8", 0, "transcript"))
    yield OutputAudioDelta(
        spoken, message, b"\x00\x01" * 2400, Delivery(1, "item_8", 0, "audio", 4800)
    )
    call = FunctionCall("item_9", "call_1", "provider-1", "lookup", '{"id":"42"}')
    spoken.output.append(call)
    yield FunctionCallDelivered(spoken, call, "item_8")
    message.status = "completed"
    yield MessageClosed(spoken, message, "item_1", in_conversation=True)
    spoken.status = "completed"
    spoken.usage = Usage(10, 3, 4)
    yield ResponseFinished(spoken)

    written = response("text")
    yield ResponseStarted(written)
    text_message = AssistantMessage("item_10", "text")
    written.output.append(text_message)
    written.message = text_message
    yield MessageOpened(written, text_message, None)
    text_message.text = "hi"
    yield OutputTextDelta(written, text_message, "hi")
    text_message.status = "incomplete"
    yield MessageClosed(written, text_message, None, in_conversation=False)
    for details in (
        StatusDetails("cancelled", "turn_detected"),
        StatusDetails("incomplete", "max_output_tokens"),
        StatusDetails("failed", error=ErrorInfo("server_error", "backend_busy", "tts busy")),
    ):
        written.status, written.status_details = details.type, details
        yield ResponseFinished(written)


def encoded() -> list[Event]:
    encoder = EventEncoder()
    return [event for fact in facts() for event in encoder.encode(fact)]


@pytest.mark.parametrize("event", encoded(), ids=lambda event: str(event["type"]))
def test_events_validate_against_the_sdk_models(event: Event) -> None:
    sdk_parse(event)


def test_event_ids_count_up_and_items_are_never_created() -> None:
    events = encoded()
    assert [event["event_id"] for event in events] == [
        f"event_{n}" for n in range(1, len(events) + 1)
    ]
    assert "conversation.item.created" not in {event["type"] for event in events}  # D1


def test_session_echo_is_the_full_effective_session() -> None:
    event = EventEncoder().encode(SessionStarted("sess_1", "conv_1", SETTINGS))[0]
    session = event["session"]
    assert isinstance(session, dict)
    assert (session["id"], session["object"], session["type"]) == (
        "sess_1",
        "realtime.session",
        "realtime",
    )
    assert session["audio"] == {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "qwen3-asr", "language": "zh"},
            "noise_reduction": None,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 200,
                "idle_timeout_ms": None,
                "create_response": True,
                "interrupt_response": True,
            },
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "vivian", "speed": 1.0},
    }
    assert session["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up.",
            "parameters": {"type": "object"},
        },
        {"type": "function", "name": "bare"},
    ]
    assert (session["max_output_tokens"], session["truncation"], session["tracing"]) == (
        100,
        "auto",
        None,
    )


def test_error_event_echoes_the_client_event_id() -> None:
    error = ErrorInfo("invalid_request_error", "invalid_value", "bad", "session.foo", "c1")
    event = EventEncoder().encode(ErrorRaised(error))[0]
    assert event == {
        "type": "error",
        "event_id": "event_1",
        "error": {
            "type": "invalid_request_error",
            "code": "invalid_value",
            "message": "bad",
            "param": "session.foo",
            "event_id": "c1",
        },
    }


def test_function_call_lifecycle_order() -> None:
    spoken = response()
    call = FunctionCall("item_9", "call_1", "p", "lookup", "{}")
    spoken.output.append(call)
    events = EventEncoder().encode(FunctionCallDelivered(spoken, call, None))
    assert [event["type"] for event in events] == [
        "response.output_item.added",
        "conversation.item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "conversation.item.done",
    ]
    assert events[2]["delta"] == "{}"
    assert events[2]["call_id"] == "call_1"


def test_deleted_message_closes_without_conversation_item_done() -> None:
    written = response("text")
    message = AssistantMessage("item_1", "text", text="x", status="completed")
    written.output.append(message)
    events = EventEncoder().encode(MessageClosed(written, message, None, in_conversation=False))
    assert [event["type"] for event in events] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]


def test_the_sdk_check_catches_missing_and_invented_fields() -> None:
    with pytest.raises(ValueError, match="event_id"):
        sdk_parse({"type": "conversation.item.deleted", "item_id": "item_1"})
    with pytest.raises(AssertionError, match=r"conversation\.item\.deleted\.extra"):
        sdk_parse(
            {"type": "conversation.item.deleted", "event_id": "e", "item_id": "i", "extra": 1}
        )
    item = {"type": "function_call_output", "call_id": "c", "output": "", "color": "red"}
    with pytest.raises(AssertionError, match=r"response\.done\.response\.output\[0\]\.color"):
        sdk_parse({"type": "response.done", "event_id": "e", "response": {"output": [item]}})

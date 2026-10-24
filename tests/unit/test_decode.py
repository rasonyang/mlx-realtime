from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from mlx_realtime.audio.formats import PCM as PCM_FORMAT
from mlx_realtime.audio.formats import PCMA, PCMU, AudioFormat
from mlx_realtime.protocol.decode import decode_text_frame, reject_binary_frame
from mlx_realtime.protocol.fields import ProtocolError
from mlx_realtime.protocol.session import decode_session, decode_session_defaults
from mlx_realtime.providers.base import ToolDef
from mlx_realtime.session.commands import (
    AppendAudio,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateItem,
    CreateResponse,
    DeleteItem,
    NewFunctionCallOutput,
    NewMessage,
    RejectEvent,
    ResponseParams,
    TruncateItem,
    UpdateSession,
)
from mlx_realtime.session.settings import (
    UNSET,
    ServerVadPatch,
    SessionPatch,
    SessionSettings,
    TranscriptionPatch,
)

TRACES = Path(__file__).parents[2] / "spikes" / "aicc" / "traces"


def decode(event: object) -> object:
    return decode_text_frame(json.dumps(event))


def session_update(session: dict[str, object]) -> dict[str, object]:
    return {"type": "session.update", "event_id": "c1", "session": {"type": "realtime", **session}}


def transcription_update(session: dict[str, object]) -> dict[str, object]:
    return {
        "type": "session.update",
        "event_id": "c1",
        "session": {"type": "transcription", **session},
    }


def item_create(item: dict[str, object], **extra: object) -> dict[str, object]:
    return {"type": "conversation.item.create", "event_id": "c1", "item": item, **extra}


def message(role: str, *parts: dict[str, object]) -> dict[str, object]:
    return {"type": "message", "role": role, "content": list(parts)}


def response_create(response: object) -> dict[str, object]:
    return {"type": "response.create", "event_id": "c1", "response": response}


PCM = base64.b64encode(b"\x01\x00\x02\x00").decode()


# Every rejection: (event, code, param). Each case is one field wrong in an otherwise valid event.
REJECTIONS: list[tuple[str, object, str, str | None]] = [
    ("missing type", {"event_id": "c1"}, "invalid_event", None),
    ("type not a string", {"type": 5}, "invalid_value", "type"),
    ("unknown type", {"type": "scooby.dooby.doo"}, "invalid_value", "type"),
    ("retrieve", {"type": "conversation.item.retrieve", "item_id": "x"}, "invalid_event", "type"),
    ("webrtc clear", {"type": "output_audio_buffer.clear"}, "invalid_event", "type"),
    (
        "beta transcription update",
        {"type": "transcription_session.update"},
        "invalid_event",
        "type",
    ),
    (
        "event_id not a string",
        {"type": "input_audio_buffer.commit", "event_id": 1},
        "invalid_value",
        "event_id",
    ),
    # input_audio_buffer.*
    ("append without audio", {"type": "input_audio_buffer.append"}, "invalid_value", "audio"),
    (
        "append audio not a string",
        {"type": "input_audio_buffer.append", "audio": 1},
        "invalid_value",
        "audio",
    ),
    (
        "append bad base64",
        {"type": "input_audio_buffer.append", "audio": "!!"},
        "invalid_value",
        "audio",
    ),
    (
        "append unknown field",
        {"type": "input_audio_buffer.append", "audio": PCM, "x": 1},
        "invalid_value",
        "x",
    ),
    ("commit unknown field", {"type": "input_audio_buffer.commit", "x": 1}, "invalid_value", "x"),
    ("clear unknown field", {"type": "input_audio_buffer.clear", "x": 1}, "invalid_value", "x"),
    # session.update
    ("no session", {"type": "session.update"}, "invalid_value", "session"),
    (
        "session not an object",
        {"type": "session.update", "session": []},
        "invalid_value",
        "session",
    ),
    (
        "session without type",
        {"type": "session.update", "session": {}},
        "invalid_value",
        "session.type",
    ),
    # transcription sessions
    (
        "transcription server_vad",
        transcription_update({"audio": {"input": {"turn_detection": {"type": "server_vad"}}}}),
        "invalid_value",
        "session.audio.input.turn_detection",
    ),
    (
        "transcription semantic_vad",
        transcription_update({"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}),
        "invalid_value",
        "session.audio.input.turn_detection",
    ),
    (
        "transcription pcm rate",
        transcription_update(
            {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        ),
        "invalid_value",
        "session.audio.input.format.rate",
    ),
    (
        "transcription output audio",
        transcription_update({"audio": {"output": {"voice": "vivian"}}}),
        "invalid_value",
        "session.audio.output",
    ),
    (
        "transcription instructions",
        transcription_update({"instructions": "x"}),
        "invalid_value",
        "session.instructions",
    ),
    ("transcription model", transcription_update({"model": "x"}), "invalid_value", "session.model"),
    (
        "transcription prompt too long",
        transcription_update({"audio": {"input": {"transcription": {"prompt": "术" * 1025}}}}),
        "invalid_value",
        "session.audio.input.transcription.prompt",
    ),
    (
        "other session type",
        {"type": "session.update", "session": {"type": "chat"}},
        "invalid_value",
        "session.type",
    ),
    ("unknown session field", session_update({"foo": 1}), "invalid_value", "session.foo"),
    (
        "beta modalities",
        session_update({"modalities": ["text"]}),
        "invalid_value",
        "session.modalities",
    ),
    ("beta voice", session_update({"voice": "alloy"}), "invalid_value", "session.voice"),
    (
        "beta turn_detection",
        session_update({"turn_detection": None}),
        "invalid_value",
        "session.turn_detection",
    ),
    (
        "beta input format",
        session_update({"input_audio_format": "pcm16"}),
        "invalid_value",
        "session.input_audio_format",
    ),
    (
        "beta temperature",
        session_update({"temperature": 0.8}),
        "invalid_value",
        "session.temperature",
    ),
    (
        "parallel tool calls",
        session_update({"parallel_tool_calls": True}),
        "invalid_value",
        "session.parallel_tool_calls",
    ),
    (
        "reasoning",
        session_update({"reasoning": {"effort": "low"}}),
        "invalid_value",
        "session.reasoning",
    ),
    (
        "instructions not a string",
        session_update({"instructions": 1}),
        "invalid_value",
        "session.instructions",
    ),
    ("model not a string", session_update({"model": 1}), "invalid_value", "session.model"),
    (
        "both modalities",
        session_update({"output_modalities": ["audio", "text"]}),
        "invalid_value",
        "session.output_modalities",
    ),
    (
        "modalities not a list",
        session_update({"output_modalities": "audio"}),
        "invalid_value",
        "session.output_modalities",
    ),
    (
        "modality not a string",
        session_update({"output_modalities": [1]}),
        "invalid_value",
        "session.output_modalities[0]",
    ),
    (
        "unknown audio field",
        session_update({"audio": {"foo": {}}}),
        "invalid_value",
        "session.audio.foo",
    ),
    (
        "unknown input field",
        session_update({"audio": {"input": {"foo": 1}}}),
        "invalid_value",
        "session.audio.input.foo",
    ),
    (
        "pcmu input at 16 kHz",
        session_update({"audio": {"input": {"format": {"type": "audio/pcmu", "rate": 16000}}}}),
        "invalid_value",
        "session.audio.input.format.rate",
    ),
    (
        "pcma output at 24 kHz",
        session_update({"audio": {"output": {"format": {"type": "audio/pcma", "rate": 24000}}}}),
        "invalid_value",
        "session.audio.output.format.rate",
    ),
    (
        "pcm at 8 kHz",
        session_update({"audio": {"input": {"format": {"type": "audio/pcm", "rate": 8000}}}}),
        "invalid_value",
        "session.audio.input.format.rate",
    ),
    (
        "rate as a string",
        session_update({"audio": {"input": {"format": {"type": "audio/pcmu", "rate": "8000"}}}}),
        "invalid_value",
        "session.audio.input.format.rate",
    ),
    (
        "beta format name",
        session_update({"audio": {"input": {"format": {"type": "g711_ulaw"}}}}),
        "invalid_value",
        "session.audio.input.format.type",
    ),
    (
        "format unknown key",
        session_update({"audio": {"input": {"format": {"type": "audio/pcmu", "law": "mu"}}}}),
        "invalid_value",
        "session.audio.input.format.law",
    ),
    (
        "other format",
        session_update({"audio": {"input": {"format": {"type": "audio/wav"}}}}),
        "invalid_value",
        "session.audio.input.format.type",
    ),
    (
        "16 kHz",
        session_update({"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}),
        "invalid_value",
        "session.audio.input.format.rate",
    ),
    (
        "format field",
        session_update({"audio": {"input": {"format": {"type": "audio/pcm", "channels": 1}}}}),
        "invalid_value",
        "session.audio.input.format.channels",
    ),
    (
        "noise reduction",
        session_update({"audio": {"input": {"noise_reduction": {"type": "near_field"}}}}),
        "invalid_value",
        "session.audio.input.noise_reduction",
    ),
    (
        "transcription delay",
        session_update({"audio": {"input": {"transcription": {"delay": "low"}}}}),
        "invalid_value",
        "session.audio.input.transcription.delay",
    ),
    (
        "transcription keywords",
        session_update({"audio": {"input": {"transcription": {"keywords": ["a"]}}}}),
        "invalid_value",
        "session.audio.input.transcription.keywords",
    ),
    (
        "transcription prompt type",
        session_update({"audio": {"input": {"transcription": {"prompt": 5}}}}),
        "invalid_value",
        "session.audio.input.transcription.prompt",
    ),
    (
        "transcription prompt length",
        session_update({"audio": {"input": {"transcription": {"prompt": "术" * 1025}}}}),
        "invalid_value",
        "session.audio.input.transcription.prompt",
    ),
    (
        "transcription language",
        session_update({"audio": {"input": {"transcription": {"language": 5}}}}),
        "invalid_value",
        "session.audio.input.transcription.language",
    ),
    (
        "semantic vad",
        session_update({"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}),
        "invalid_value",
        "session.audio.input.turn_detection.type",
    ),
    (
        "vad without type",
        session_update({"audio": {"input": {"turn_detection": {"threshold": 0.5}}}}),
        "invalid_value",
        "session.audio.input.turn_detection.type",
    ),
    (
        "threshold range",
        session_update(
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 1.5}}}}
        ),
        "invalid_value",
        "session.audio.input.turn_detection.threshold",
    ),
    (
        "eagerness on server_vad",
        session_update(
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "eagerness": "low"}}}}
        ),
        "invalid_value",
        "session.audio.input.turn_detection.eagerness",
    ),
    (
        "idle timeout",
        session_update(
            {
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000}}
                }
            }
        ),
        "invalid_value",
        "session.audio.input.turn_detection.idle_timeout_ms",
    ),
    (
        "negative padding",
        session_update(
            {
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "prefix_padding_ms": -1}}
                }
            }
        ),
        "invalid_value",
        "session.audio.input.turn_detection.prefix_padding_ms",
    ),
    (
        "silence as float",
        session_update(
            {
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 1.5}}
                }
            }
        ),
        "invalid_value",
        "session.audio.input.turn_detection.silence_duration_ms",
    ),
    (
        "create_response type",
        session_update(
            {
                "audio": {
                    "input": {"turn_detection": {"type": "server_vad", "create_response": "yes"}}
                }
            }
        ),
        "invalid_value",
        "session.audio.input.turn_detection.create_response",
    ),
    (
        "custom voice",
        session_update({"audio": {"output": {"voice": {"id": "voice_1"}}}}),
        "invalid_value",
        "session.audio.output.voice",
    ),
    (
        "speed",
        session_update({"audio": {"output": {"speed": 1.5}}}),
        "invalid_value",
        "session.audio.output.speed",
    ),
    (
        "unknown output field",
        session_update({"audio": {"output": {"volume": 1}}}),
        "invalid_value",
        "session.audio.output.volume",
    ),
    ("tools not a list", session_update({"tools": {}}), "invalid_value", "session.tools"),
    (
        "mcp tool",
        session_update({"tools": [{"type": "mcp", "server_label": "x"}]}),
        "invalid_value",
        "session.tools[0].type",
    ),
    (
        "tool without type",
        session_update({"tools": [{"name": "f"}]}),
        "invalid_value",
        "session.tools[0].type",
    ),
    (
        "tool without name",
        session_update({"tools": [{"type": "function"}]}),
        "invalid_value",
        "session.tools[0].name",
    ),
    (
        "duplicate tool",
        session_update(
            {"tools": [{"type": "function", "name": "f"}, {"type": "function", "name": "f"}]}
        ),
        "invalid_value",
        "session.tools[1].name",
    ),
    (
        "tool parameters",
        session_update({"tools": [{"type": "function", "name": "f", "parameters": "{}"}]}),
        "invalid_value",
        "session.tools[0].parameters",
    ),
    (
        "tool field",
        session_update({"tools": [{"type": "function", "name": "f", "strict": True}]}),
        "invalid_value",
        "session.tools[0].strict",
    ),
    (
        "tool_choice required",
        session_update({"tool_choice": "required"}),
        "invalid_value",
        "session.tool_choice",
    ),
    (
        "forced function",
        session_update({"tool_choice": {"type": "function", "name": "f"}}),
        "invalid_value",
        "session.tool_choice",
    ),
    (
        "tool_choice value",
        session_update({"tool_choice": "maybe"}),
        "invalid_value",
        "session.tool_choice",
    ),
    (
        "zero tokens",
        session_update({"max_output_tokens": 0}),
        "invalid_value",
        "session.max_output_tokens",
    ),
    (
        "too many tokens",
        session_update({"max_output_tokens": 4097}),
        "invalid_value",
        "session.max_output_tokens",
    ),
    (
        "tokens string",
        session_update({"max_output_tokens": "lots"}),
        "invalid_value",
        "session.max_output_tokens",
    ),
    (
        "truncation",
        session_update({"truncation": "disabled"}),
        "invalid_value",
        "session.truncation",
    ),
    ("tracing", session_update({"tracing": "auto"}), "invalid_value", "session.tracing"),
    ("prompt", session_update({"prompt": {"id": "pmpt_1"}}), "invalid_value", "session.prompt"),
    (
        "include",
        session_update({"include": ["item.input_audio_transcription.logprobs"]}),
        "invalid_value",
        "session.include",
    ),
    # conversation.item.create
    ("no item", {"type": "conversation.item.create"}, "invalid_value", "item"),
    (
        "function_call item",
        item_create({"type": "function_call", "name": "f", "arguments": "{}"}),
        "invalid_value",
        "item.type",
    ),
    ("mcp item", item_create({"type": "mcp_call"}), "invalid_value", "item.type"),
    (
        "unknown item",
        item_create({"type": "item_reference", "id": "x"}),
        "invalid_value",
        "item.type",
    ),
    ("no role", item_create({"type": "message", "content": []}), "invalid_value", "item.role"),
    (
        "tool role",
        item_create(message("tool", {"type": "input_text", "text": "x"})),
        "invalid_value",
        "item.role",
    ),
    (
        "no content",
        item_create({"type": "message", "role": "user"}),
        "invalid_value",
        "item.content",
    ),
    ("empty content", item_create(message("user")), "invalid_value", "item.content"),
    (
        "user audio",
        item_create(message("user", {"type": "input_audio", "audio": PCM})),
        "invalid_value",
        "item.content[0].type",
    ),
    (
        "user image",
        item_create(message("user", {"type": "input_image", "image_url": "x"})),
        "invalid_value",
        "item.content[0].type",
    ),
    (
        "assistant input_text",
        item_create(message("assistant", {"type": "input_text", "text": "x"})),
        "invalid_value",
        "item.content[0].type",
    ),
    (
        "assistant audio",
        item_create(message("assistant", {"type": "output_audio", "transcript": "x"})),
        "invalid_value",
        "item.content[0].type",
    ),
    (
        "system output_text",
        item_create(message("system", {"type": "output_text", "text": "x"})),
        "invalid_value",
        "item.content[0].type",
    ),
    (
        "part without text",
        item_create(message("user", {"type": "input_text"})),
        "invalid_value",
        "item.content[0].text",
    ),
    (
        "second part",
        item_create(
            message("user", {"type": "input_text", "text": "a"}, {"type": "input_text", "text": 2})
        ),
        "invalid_value",
        "item.content[1].text",
    ),
    (
        "part field",
        item_create(message("user", {"type": "input_text", "text": "a", "audio": PCM})),
        "invalid_value",
        "item.content[0].audio",
    ),
    (
        "empty id",
        item_create({**message("user", {"type": "input_text", "text": "a"}), "id": ""}),
        "invalid_value",
        "item.id",
    ),
    (
        "object",
        item_create({**message("user", {"type": "input_text", "text": "a"}), "object": "item"}),
        "invalid_value",
        "item.object",
    ),
    (
        "status",
        item_create({**message("user", {"type": "input_text", "text": "a"}), "status": "done"}),
        "invalid_value",
        "item.status",
    ),
    (
        "message field",
        item_create({**message("user", {"type": "input_text", "text": "a"}), "name": "x"}),
        "invalid_value",
        "item.name",
    ),
    (
        "output without call_id",
        item_create({"type": "function_call_output", "output": "{}"}),
        "invalid_value",
        "item.call_id",
    ),
    (
        "output without output",
        item_create({"type": "function_call_output", "call_id": "call_1"}),
        "invalid_value",
        "item.output",
    ),
    (
        "previous id",
        item_create(message("user", {"type": "input_text", "text": "a"}), previous_item_id=5),
        "invalid_value",
        "previous_item_id",
    ),
    # truncate, delete
    (
        "truncate without item",
        {"type": "conversation.item.truncate", "content_index": 0, "audio_end_ms": 0},
        "invalid_value",
        "item_id",
    ),
    (
        "negative index",
        {
            "type": "conversation.item.truncate",
            "item_id": "i",
            "content_index": -1,
            "audio_end_ms": 0,
        },
        "invalid_value",
        "content_index",
    ),
    (
        "end not int",
        {
            "type": "conversation.item.truncate",
            "item_id": "i",
            "content_index": 0,
            "audio_end_ms": "1",
        },
        "invalid_value",
        "audio_end_ms",
    ),
    (
        "negative end",
        {
            "type": "conversation.item.truncate",
            "item_id": "i",
            "content_index": 0,
            "audio_end_ms": -5,
        },
        "invalid_value",
        "audio_end_ms",
    ),
    ("delete without item", {"type": "conversation.item.delete"}, "invalid_value", "item_id"),
    # response.create, response.cancel
    ("response not an object", response_create([]), "invalid_value", "response"),
    ("response input", response_create({"input": []}), "invalid_value", "response.input"),
    ("response tools", response_create({"tools": []}), "invalid_value", "response.tools"),
    (
        "response field",
        response_create({"temperature": 0.8}),
        "invalid_value",
        "response.temperature",
    ),
    (
        "out-of-band",
        response_create({"conversation": "none"}),
        "invalid_value",
        "response.conversation",
    ),
    (
        "response modalities",
        response_create({"output_modalities": ["text", "audio"]}),
        "invalid_value",
        "response.output_modalities",
    ),
    (
        "response tool_choice",
        response_create({"tool_choice": "required"}),
        "invalid_value",
        "response.tool_choice",
    ),
    (
        "response tokens",
        response_create({"max_output_tokens": 0}),
        "invalid_value",
        "response.max_output_tokens",
    ),
    (
        "response voice",
        response_create({"audio": {"output": {"voice": {"id": "v"}}}}),
        "invalid_value",
        "response.audio.output.voice",
    ),
    (
        "response format rate",
        response_create({"audio": {"output": {"format": {"type": "audio/pcmu", "rate": 24000}}}}),
        "invalid_value",
        "response.audio.output.format.rate",
    ),
    (
        "response format type",
        response_create({"audio": {"output": {"format": {"type": "audio/opus"}}}}),
        "invalid_value",
        "response.audio.output.format.type",
    ),
    (
        "response speed",
        response_create({"audio": {"output": {"speed": 1.0}}}),
        "invalid_value",
        "response.audio.output.speed",
    ),
    (
        "response audio field",
        response_create({"audio": {"input": {}}}),
        "invalid_value",
        "response.audio.input",
    ),
    (
        "metadata size",
        response_create({"metadata": {str(n): "v" for n in range(17)}}),
        "invalid_value",
        "response.metadata",
    ),
    (
        "metadata value",
        response_create({"metadata": {"k": 5}}),
        "invalid_value",
        "response.metadata.k",
    ),
    (
        "metadata key",
        response_create({"metadata": {"k" * 65: "v"}}),
        "invalid_value",
        "response.metadata." + "k" * 65,
    ),
    ("cancel id", {"type": "response.cancel", "response_id": 5}, "invalid_value", "response_id"),
]


@pytest.mark.parametrize(
    ("event", "code", "param"),
    [case[1:] for case in REJECTIONS],
    ids=[case[0] for case in REJECTIONS],
)
def test_rejections_name_their_param(event: object, code: str, param: str | None) -> None:
    command = decode(event)
    assert isinstance(command, RejectEvent)
    error = command.error
    assert (error.type, error.code, error.param) == ("invalid_request_error", code, param)
    assert error.message
    expected_event_id = event.get("event_id") if isinstance(event, dict) else None
    assert error.event_id == (expected_event_id if isinstance(expected_event_id, str) else None)


@pytest.mark.parametrize("frame", ["not json", "[1, 2]", '"text"', "{" * 100_000])
def test_frames_that_are_not_objects_are_invalid_events(frame: str) -> None:
    command = decode_text_frame(frame)
    assert isinstance(command, RejectEvent)
    assert (command.error.code, command.error.param, command.error.event_id) == (
        "invalid_event",
        None,
        None,
    )


def test_binary_frames_are_invalid_events() -> None:
    assert reject_binary_frame().error.code == "invalid_event"


def test_append_decodes_whole_samples() -> None:
    assert decode({"type": "input_audio_buffer.append", "audio": PCM}) == AppendAudio(
        b"\x01\x00\x02\x00"
    )
    assert decode({"type": "input_audio_buffer.append", "audio": ""}) == AppendAudio(b"")


def test_simple_events() -> None:
    assert decode({"type": "input_audio_buffer.commit", "event_id": "e"}) == CommitAudio("e")
    assert decode({"type": "input_audio_buffer.clear"}) == ClearAudio()
    assert decode({"type": "conversation.item.delete", "item_id": "i"}) == DeleteItem("i")
    assert decode({"type": "response.cancel"}) == CancelResponse()
    assert decode({"type": "response.cancel", "response_id": "resp_1"}) == CancelResponse("resp_1")
    assert decode({"type": "response.create"}) == CreateResponse()
    assert decode({"type": "response.create", "response": None}) == CreateResponse()
    truncate = {
        "type": "conversation.item.truncate",
        "item_id": "i",
        "content_index": 0,
        "audio_end_ms": 120,
    }
    assert decode(truncate) == TruncateItem("i", 0, 120)


def test_full_ga_session_update() -> None:
    command = decode(
        session_update(
            {
                "model": "gpt-realtime",
                "instructions": "Be brief.",
                "output_modalities": ["text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {
                            "model": "whisper-1",
                            "language": "zh",
                            "prompt": "names",
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.6,
                            "prefix_padding_ms": 200,
                            "silence_duration_ms": 400,
                            "create_response": False,
                            "interrupt_response": True,
                            "idle_timeout_ms": None,
                        },
                        "noise_reduction": None,
                    },
                    "output": {"format": {"type": "audio/pcm"}, "voice": "marin", "speed": 1.0},
                },
                "tools": [
                    {
                        "type": "function",
                        "name": "f",
                        "description": "d",
                        "parameters": {"type": "object"},
                    }
                ],
                "tool_choice": "none",
                "max_output_tokens": "inf",
                "truncation": "auto",
                "tracing": None,
                "prompt": None,
                "include": [],
            }
        )
    )
    assert command == UpdateSession(
        SessionPatch(
            type="realtime",
            model="gpt-realtime",
            instructions="Be brief.",
            output_modality="text",
            input_format=PCM_FORMAT,
            output_format=PCM_FORMAT,
            transcription=TranscriptionPatch(model="whisper-1", language="zh", prompt="names"),
            turn_detection=ServerVadPatch(0.6, 200, 400, False, True),
            voice="marin",
            tools=(ToolDef("f", "d", {"type": "object"}),),
            tool_choice="none",
            max_output_tokens=None,
        ),
        "c1",
    )


def test_a_prompt_of_exactly_the_limit_is_accepted() -> None:
    prompt = "术" * 1024  # characters, not bytes
    command = decode(session_update({"audio": {"input": {"transcription": {"prompt": prompt}}}}))
    assert command == UpdateSession(
        SessionPatch(type="realtime", transcription=TranscriptionPatch(prompt=prompt)), "c1"
    )


def test_a_prompt_over_the_limit_says_so() -> None:
    command = decode(
        session_update({"audio": {"input": {"transcription": {"prompt": "a" * 5000}}}})
    )
    assert isinstance(command, RejectEvent)
    assert "at most 1024 characters" in command.error.message
    assert "got 5000" in command.error.message


def test_nulls_clear_and_absent_fields_stay_unset() -> None:
    command = decode(
        session_update({"audio": {"input": {"transcription": None, "turn_detection": None}}})
    )
    assert command == UpdateSession(
        SessionPatch(type="realtime", transcription=None, turn_detection=None), "c1"
    )
    command = decode(session_update({"instructions": ""}))
    assert command == UpdateSession(SessionPatch(type="realtime", instructions=""), "c1")


@pytest.mark.parametrize("trace", sorted(TRACES.glob("*.jsonl")), ids=lambda path: path.stem)
def test_every_aicc_frame_decodes(trace: Path) -> None:
    for line in trace.read_text(encoding="utf-8").splitlines():
        command = decode_text_frame(line)
        assert not isinstance(command, RejectEvent), (line[:120], command)


def test_aicc_session_update_shape() -> None:
    frame = (
        (TRACES / "opening_gateway_with_voice.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    command = decode_text_frame(frame)
    assert isinstance(command, UpdateSession)
    patch = command.patch
    assert patch.turn_detection == ServerVadPatch(silence_duration_ms=500)
    assert patch.voice == "flow-voice"
    assert patch.transcription is UNSET
    assert patch.tools != UNSET
    assert [tool.name for tool in patch.tools] == ["lookup_balance"]
    settings = SessionSettings().apply(patch, SessionSettings())
    assert settings.turn_detection is not None
    assert settings.turn_detection.silence_duration_ms == 500
    assert settings.turn_detection.prefix_padding_ms == 300


def test_item_create_variants() -> None:
    command = decode(
        item_create(
            {
                **message(
                    "user", {"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}
                ),
                "id": "mine",
                "object": "realtime.item",
                "status": "completed",
            },
            previous_item_id="root",
        )
    )
    assert command == CreateItem(NewMessage("user", ("a", "b"), "mine"), "root", "c1")
    command = decode(item_create(message("assistant", {"type": "output_text", "text": "hi"})))
    assert command == CreateItem(NewMessage("assistant", ("hi",)), None, "c1")
    command = decode(
        item_create({"type": "function_call_output", "call_id": "call_1", "output": ""})
    )
    assert command == CreateItem(NewFunctionCallOutput("call_1", ""), None, "c1")


def test_response_create_fields() -> None:
    command = decode(
        response_create(
            {
                "instructions": "Say hi.",
                "output_modalities": ["text"],
                "max_output_tokens": 50,
                "tool_choice": "auto",
                "audio": {
                    "output": {"voice": "cedar", "format": {"type": "audio/pcm", "rate": 24000}}
                },
                "metadata": {"purpose": "greeting"},
                "conversation": "auto",
            }
        )
    )
    assert command == CreateResponse(
        ResponseParams(
            "Say hi.",
            "text",
            50,
            "auto",
            "cedar",
            {"purpose": "greeting"},
            output_format=PCM_FORMAT,
        ),
        "c1",
    )
    assert decode(response_create({"max_output_tokens": "inf"})) == CreateResponse(
        ResponseParams(max_output_tokens=None), "c1"
    )


def test_session_defaults_need_no_type() -> None:
    assert decode_session(
        {"instructions": "x"}, "session_defaults", require_type=False
    ) == SessionPatch(instructions="x")
    assert (
        decode_session({"type": "realtime"}, "session_defaults", require_type=False).type
        == "realtime"
    )


def test_a_transcription_session_update_is_decoded() -> None:
    """AICC's exact update (internal/transcribe/openairt.go): the input format, the
    transcription's model, language and prompt, and turn detection null."""
    command = decode(
        transcription_update(
            {
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": {"model": "m", "language": "zh", "prompt": "NovaNet"},
                        "turn_detection": None,
                    }
                }
            }
        )
    )
    assert command == UpdateSession(
        SessionPatch(
            type="transcription",
            input_format=PCM_FORMAT,
            transcription=TranscriptionPatch(model="m", language="zh", prompt="NovaNet"),
        ),
        "c1",
    )
    # G.711 is accepted like in realtime sessions; an unknown model is echoed, never judged.
    g711 = decode(
        transcription_update(
            {
                "audio": {
                    "input": {"format": {"type": "audio/pcmu"}, "transcription": {"model": "?"}}
                }
            }
        )
    )
    assert isinstance(g711, UpdateSession)
    assert g711.patch.input_format == PCMU
    assert g711.patch.transcription == TranscriptionPatch(model="?")


def test_transcription_sessions_are_not_a_session_default() -> None:
    with pytest.raises(ProtocolError, match="transcription"):
        decode_session_defaults({"type": "transcription"})


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"type": "audio/pcmu"}, PCMU),
        ({"type": "audio/pcmu", "rate": 8000}, PCMU),
        ({"type": "audio/pcma"}, PCMA),
        ({"type": "audio/pcma", "rate": 8000}, PCMA),
        ({"type": "audio/pcm"}, PCM_FORMAT),
        ({"rate": 24000}, PCM_FORMAT),  # the type defaults to audio/pcm
        ({}, PCM_FORMAT),
    ],
)
def test_g711_formats_are_accepted_with_no_rate_or_8000(
    given: dict[str, object], expected: AudioFormat
) -> None:
    """M7 (D2, D3): GA's G.711 names, at 8 kHz; input and output may differ."""
    command = decode(
        session_update({"audio": {"input": {"format": given}, "output": {"format": given}}})
    )
    assert command == UpdateSession(
        SessionPatch(type="realtime", input_format=expected, output_format=expected), "c1"
    )
    mixed = decode(
        session_update(
            {
                "audio": {
                    "input": {"format": {"type": "audio/pcmu"}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                }
            }
        )
    )
    assert mixed == UpdateSession(
        SessionPatch(type="realtime", input_format=PCMU, output_format=PCM_FORMAT), "c1"
    )


def test_a_response_may_set_its_own_output_format() -> None:
    command = decode(response_create({"audio": {"output": {"format": {"type": "audio/pcma"}}}}))
    assert command == CreateResponse(ResponseParams(output_format=PCMA), "c1")


def test_appends_are_checked_for_whole_samples_by_the_session() -> None:
    """Any byte count is whole G.711 samples, so the protocol layer only decodes base64; the
    session rejects half a PCM sample (``protocol_errors.jsonl``)."""
    assert decode({"type": "input_audio_buffer.append", "audio": "AAAA", "event_id": "a"}) == (
        AppendAudio(b"\x00\x00\x00", "a")
    )

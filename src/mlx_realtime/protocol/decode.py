"""Client events to commands: validation of every GA client event in scope (docs/PROTOCOL.md).

Decoding never raises. A frame that fails validation becomes a ``RejectEvent`` carrying the
GA ``error`` payload, with the client's ``event_id`` when the frame had a readable one.
``input_audio_buffer.append`` takes a short path: only its ``audio`` is checked and decoded.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import cast

from mlx_realtime.protocol.fields import (
    ProtocolError,
    check_keys,
    child,
    element,
    expect_choice,
    expect_int,
    expect_list,
    expect_object,
    expect_str,
    require,
    value_error,
)
from mlx_realtime.protocol.session import (
    decode_audio_format,
    decode_max_output_tokens,
    decode_modalities,
    decode_session,
    decode_tool_choice,
    decode_voice,
)
from mlx_realtime.providers.base import ToolChoice
from mlx_realtime.session.commands import (
    AppendAudio,
    CancelResponse,
    ClearAudio,
    Command,
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
from mlx_realtime.session.errors import ErrorInfo
from mlx_realtime.session.settings import UNSET, Maybe, Modality

_UNSUPPORTED_EVENTS = {
    "conversation.item.retrieve": "items are not retrievable from this server.",
    "output_audio_buffer.clear": "it is a WebRTC/SIP event; over WebSocket send response.cancel.",
    "transcription_session.update": "transcription sessions are not supported.",
}
_SERVER_CREATED_ITEMS = frozenset(
    {"function_call", "mcp_call", "mcp_list_tools", "mcp_approval_request", "mcp_approval_response"}
)
_ITEM_STATUSES = ("completed", "incomplete", "in_progress")
_METADATA_PAIRS = 16
_METADATA_KEY_CHARS = 64
_METADATA_VALUE_CHARS = 512


def decode_text_frame(raw: str) -> Command:
    """One client text frame to a command."""
    event_id: str | None = None
    try:
        try:
            event: object = json.loads(raw)
        except (ValueError, RecursionError):
            raise ProtocolError("invalid_event", "The event is not valid JSON.") from None
        if not isinstance(event, dict):
            raise ProtocolError("invalid_event", "The event must be a JSON object.")
        obj = cast(dict[str, object], event)
        if "event_id" in obj:
            event_id = expect_str(obj["event_id"], "event_id")
        return _decode(obj, event_id)
    except ProtocolError as error:
        return RejectEvent(
            ErrorInfo("invalid_request_error", error.code, error.message, error.param, event_id)
        )


def reject_binary_frame() -> RejectEvent:
    message = "Binary frames are not supported; send each event as a JSON text frame."
    return RejectEvent(ErrorInfo("invalid_request_error", "invalid_event", message))


def _decode(obj: dict[str, object], event_id: str | None) -> Command:
    if "type" not in obj:
        raise ProtocolError("invalid_event", "The 'type' field is missing.")
    event_type = expect_str(obj["type"], "type")
    match event_type:
        case "input_audio_buffer.append":
            check_keys(obj, "", ("type", "event_id", "audio"))
            return AppendAudio(_decode_pcm(require(obj, "audio", "")))
        case "input_audio_buffer.commit":
            check_keys(obj, "", ("type", "event_id"))
            return CommitAudio(event_id)
        case "input_audio_buffer.clear":
            check_keys(obj, "", ("type", "event_id"))
            return ClearAudio(event_id)
        case "session.update":
            check_keys(obj, "", ("type", "event_id", "session"))
            return UpdateSession(decode_session(require(obj, "session", ""), "session"), event_id)
        case "conversation.item.create":
            return _create_item(obj, event_id)
        case "conversation.item.truncate":
            check_keys(obj, "", ("type", "event_id", "item_id", "content_index", "audio_end_ms"))
            return TruncateItem(
                item_id=expect_str(require(obj, "item_id", ""), "item_id"),
                content_index=expect_int(
                    require(obj, "content_index", ""), "content_index", minimum=0
                ),
                audio_end_ms=expect_int(
                    require(obj, "audio_end_ms", ""), "audio_end_ms", minimum=0
                ),
                event_id=event_id,
            )
        case "conversation.item.delete":
            check_keys(obj, "", ("type", "event_id", "item_id"))
            return DeleteItem(expect_str(require(obj, "item_id", ""), "item_id"), event_id)
        case "response.create":
            check_keys(obj, "", ("type", "event_id", "response"))
            response = obj.get("response")
            params = (
                ResponseParams() if response is None else _response_params(response, "response")
            )
            return CreateResponse(params, event_id)
        case "response.cancel":
            check_keys(obj, "", ("type", "event_id", "response_id"))
            response_id = obj.get("response_id")
            return CancelResponse(
                None if response_id is None else expect_str(response_id, "response_id"), event_id
            )
        case _ if event_type in _UNSUPPORTED_EVENTS:
            message = f"'{event_type}' is not supported: {_UNSUPPORTED_EVENTS[event_type]}"
            raise ProtocolError("invalid_event", message, "type")
        case _:
            message = f"Invalid value: '{event_type}'. It is not a supported client event type."
            raise ProtocolError("invalid_value", message, "type")


def _decode_pcm(value: object) -> bytes:
    text = expect_str(value, "audio")
    try:
        pcm = base64.b64decode(text, validate=True)
    except binascii.Error:
        raise value_error("audio", "expected base64-encoded audio.") from None
    if len(pcm) % 2:
        raise value_error("audio", "audio/pcm needs whole 16-bit samples (an even byte count).")
    return pcm


def _create_item(obj: dict[str, object], event_id: str | None) -> CreateItem:
    check_keys(obj, "", ("type", "event_id", "item", "previous_item_id"))
    previous = obj.get("previous_item_id")
    return CreateItem(
        item=_item(require(obj, "item", ""), "item"),
        previous_item_id=None if previous is None else expect_str(previous, "previous_item_id"),
        event_id=event_id,
    )


def _item(value: object, path: str) -> NewMessage | NewFunctionCallOutput:
    obj = expect_object(value, path)
    type_param = child(path, "type")
    item_type = expect_str(require(obj, "type", path), type_param)
    if item_type in _SERVER_CREATED_ITEMS:
        message = f"'{item_type}' items cannot be created by the client."
        raise ProtocolError("invalid_value", message, type_param)
    expect_choice(item_type, type_param, ("message", "function_call_output"))
    item_id: str | None = None
    if obj.get("id") is not None:
        item_id = expect_str(obj["id"], child(path, "id"))
        if not item_id:
            raise value_error(child(path, "id"), "the id must not be empty.")
    if "object" in obj:
        expect_choice(obj["object"], child(path, "object"), ("realtime.item",))
    if "status" in obj:
        expect_choice(obj["status"], child(path, "status"), _ITEM_STATUSES)
    if item_type == "function_call_output":
        check_keys(obj, path, ("type", "id", "object", "status", "call_id", "output"))
        return NewFunctionCallOutput(
            call_id=expect_str(require(obj, "call_id", path), child(path, "call_id")),
            output=expect_str(require(obj, "output", path), child(path, "output")),
            id=item_id,
        )
    check_keys(obj, path, ("type", "id", "object", "status", "role", "content"))
    role_param = child(path, "role")
    role = expect_choice(require(obj, "role", path), role_param, ("system", "user", "assistant"))
    content_path = child(path, "content")
    content = expect_list(require(obj, "content", path), content_path)
    if not content:
        raise value_error(content_path, "a message needs at least one content part.")
    part_type = "output_text" if role == "assistant" else "input_text"
    texts = tuple(
        _text_part(part, element(content_path, index), part_type)
        for index, part in enumerate(content)
    )
    return NewMessage(role, texts, item_id)


def _text_part(value: object, path: str, expected: str) -> str:
    part = expect_object(value, path)
    type_param = child(path, "type")
    part_type = expect_str(require(part, "type", path), type_param)
    unsupported = {
        "input_audio": "user audio enters only through input_audio_buffer.append.",
        "input_image": "images are not supported.",
        "output_audio": "assistant audio cannot be created by the client.",
    }
    if part_type in unsupported:
        raise ProtocolError(
            "invalid_value", f"'{part_type}' is not supported: {unsupported[part_type]}", type_param
        )
    expect_choice(part_type, type_param, (expected,))
    check_keys(part, path, ("type", "text"))
    return expect_str(require(part, "text", path), child(path, "text"))


def _response_params(value: object, path: str) -> ResponseParams:
    obj = expect_object(value, path)
    check_keys(
        obj,
        path,
        (
            "instructions",
            "output_modalities",
            "max_output_tokens",
            "tool_choice",
            "audio",
            "metadata",
            "conversation",
        ),
        unsupported={
            "input": "out-of-band input is not supported; responses use the conversation.",
            "tools": "tools are set on the session.",
            "prompt": "prompt templates are not supported.",
            "reasoning": "the model's reasoning is always off.",
            "parallel_tool_calls": "tool calls are made one at a time.",
        },
    )
    conversation = obj.get("conversation")
    if conversation is not None and conversation != "auto":
        param = child(path, "conversation")
        message = f"'{param}' is not supported: only 'auto' (the default conversation) is served."
        raise ProtocolError("invalid_value", message, param)
    instructions: str | None = None
    modality: Modality | None = None
    max_output_tokens: Maybe[int | None] = UNSET
    tool_choice: ToolChoice | None = None
    if "instructions" in obj:
        instructions = expect_str(obj["instructions"], child(path, "instructions"))
    if "output_modalities" in obj:
        modality = decode_modalities(obj["output_modalities"], child(path, "output_modalities"))
    if "max_output_tokens" in obj:
        max_output_tokens = decode_max_output_tokens(
            obj["max_output_tokens"], child(path, "max_output_tokens")
        )
    if "tool_choice" in obj:
        tool_choice = decode_tool_choice(obj["tool_choice"], child(path, "tool_choice"))
    return ResponseParams(
        instructions=instructions,
        output_modality=modality,
        max_output_tokens=max_output_tokens,
        tool_choice=tool_choice,
        voice=_response_voice(obj.get("audio"), child(path, "audio")),
        metadata=_metadata(obj.get("metadata"), child(path, "metadata")),
    )


def _response_voice(value: object, path: str) -> str | None:
    if value is None:
        return None
    audio = expect_object(value, path)
    check_keys(audio, path, ("output",))
    if audio.get("output") is None:
        return None
    output_path = child(path, "output")
    output = expect_object(audio["output"], output_path)
    check_keys(output, output_path, ("voice", "format"))
    if "format" in output:
        decode_audio_format(output["format"], child(output_path, "format"))
    return (
        None
        if "voice" not in output
        else decode_voice(output["voice"], child(output_path, "voice"))
    )


def _metadata(value: object, path: str) -> dict[str, str] | None:
    if value is None:
        return None
    obj = expect_object(value, path)
    if len(obj) > _METADATA_PAIRS:
        raise value_error(path, f"at most {_METADATA_PAIRS} key-value pairs are allowed.")
    metadata: dict[str, str] = {}
    for key, entry in obj.items():
        param = child(path, key)
        if len(key) > _METADATA_KEY_CHARS:
            raise value_error(param, f"keys are at most {_METADATA_KEY_CHARS} characters.")
        text = expect_str(entry, param)
        if len(text) > _METADATA_VALUE_CHARS:
            raise value_error(param, f"values are at most {_METADATA_VALUE_CHARS} characters.")
        metadata[key] = text
    return metadata

"""The GA session object: ``session.update`` validation and the effective-session echo.

Validation rejects the whole update on the first problem, so a failed update changes nothing.
The echo always carries the full effective session, with ``id`` and ``object``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from mlx_realtime.protocol.fields import (
    ProtocolError,
    check_keys,
    child,
    element,
    expect_bool,
    expect_choice,
    expect_int,
    expect_list,
    expect_number,
    expect_object,
    expect_str,
    require,
    value_error,
)
from mlx_realtime.providers.base import ToolChoice, ToolDef
from mlx_realtime.session.settings import (
    UNSET,
    Maybe,
    Modality,
    ServerVad,
    ServerVadPatch,
    SessionPatch,
    SessionSettings,
    Transcription,
    TranscriptionPatch,
)

WIRE_FORMAT: dict[str, object] = {"type": "audio/pcm", "rate": 24000}
MAX_OUTPUT_TOKENS = 4096

_TOOL_CHOICES: tuple[ToolChoice, ...] = ("auto", "none")

_SESSION_KEYS = frozenset(
    {"type", "model", "instructions", "output_modalities", "audio", "tools", "tool_choice"}
    | {"max_output_tokens", "truncation", "tracing", "prompt", "include"}
)
_SESSION_UNSUPPORTED = {
    "parallel_tool_calls": "tool calls are made one at a time.",
    "reasoning": "the model's reasoning is always off.",
}
_BETA_SESSION_KEYS = frozenset(
    {"modalities", "voice", "turn_detection", "input_audio_format", "output_audio_format"}
    | {"input_audio_transcription", "temperature", "max_response_output_tokens"}
)


def decode_session(value: object, path: str, *, require_type: bool = True) -> SessionPatch:
    """Validate a GA session object (``session.update``'s ``session``) into a patch."""
    obj = expect_object(value, path)
    check_keys(obj, path, _SESSION_KEYS, unsupported=_SESSION_UNSUPPORTED, beta=_BETA_SESSION_KEYS)
    if require_type or "type" in obj:
        _decode_session_type(require(obj, "type", path), child(path, "type"))

    model: Maybe[str] = UNSET
    instructions: Maybe[str] = UNSET
    modality: Maybe[Modality] = UNSET
    transcription: Maybe[TranscriptionPatch | None] = UNSET
    turn_detection: Maybe[ServerVadPatch | None] = UNSET
    voice: Maybe[str] = UNSET
    tools: Maybe[tuple[ToolDef, ...]] = UNSET
    tool_choice: Maybe[ToolChoice] = UNSET
    max_output_tokens: Maybe[int | None] = UNSET
    if "model" in obj:
        model = expect_str(obj["model"], child(path, "model"))
    if "instructions" in obj:
        instructions = expect_str(obj["instructions"], child(path, "instructions"))
    if "output_modalities" in obj:
        modality = decode_modalities(obj["output_modalities"], child(path, "output_modalities"))
    if "audio" in obj:
        transcription, turn_detection, voice = _decode_audio(obj["audio"], child(path, "audio"))
    if "tools" in obj:
        tools = _decode_tools(obj["tools"], child(path, "tools"))
    if "tool_choice" in obj:
        tool_choice = decode_tool_choice(obj["tool_choice"], child(path, "tool_choice"))
    if "max_output_tokens" in obj:
        max_output_tokens = decode_max_output_tokens(
            obj["max_output_tokens"], child(path, "max_output_tokens")
        )
    _expect_default(obj, path, "truncation", "auto")
    _expect_default(obj, path, "tracing", None)
    _expect_default(obj, path, "prompt", None)
    if obj.get("include") not in (None, []):
        raise value_error(child(path, "include"), "only null or an empty array is supported.")
    return SessionPatch(
        model=model,
        instructions=instructions,
        output_modality=modality,
        transcription=transcription,
        turn_detection=turn_detection,
        voice=voice,
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
    )


def decode_session_defaults(
    raw: Mapping[str, object], *, base: SessionSettings | None = None
) -> SessionSettings:
    """The configured ``session_defaults``, validated exactly like a ``session.update``, over
    ``base`` (the built-in settings by default)."""
    builtin = base or SessionSettings()
    return builtin.apply(decode_session(dict(raw), "session_defaults", require_type=False), builtin)


def decode_modalities(value: object, param: str) -> Modality:
    items = [
        expect_str(item, element(param, index))
        for index, item in enumerate(expect_list(value, param))
    ]
    if items == ["audio"]:
        return "audio"
    if items == ["text"]:
        return "text"
    raise value_error(param, 'expected ["audio"] or ["text"].')


def decode_tool_choice(value: object, param: str) -> ToolChoice:
    """``required`` and forced functions stay unsupported until M5 (ARCHITECTURE §5.2)."""
    if value == "required" or isinstance(value, dict):
        message = f"'{param}' is not supported: only 'auto' and 'none' are accepted."
        raise ProtocolError("invalid_value", message, param)
    return expect_choice(value, param, _TOOL_CHOICES)


def decode_max_output_tokens(value: object, param: str) -> int | None:
    if value == "inf":
        return None
    if isinstance(value, str):
        raise value_error(param, f"expected 'inf' or an integer, but got '{value}'.")
    return expect_int(value, param, minimum=1, maximum=MAX_OUTPUT_TOKENS)


def decode_audio_format(value: object, path: str) -> None:
    """Only ``audio/pcm`` at 24 kHz is served; G.711 arrives later (ARCHITECTURE §13, M7)."""
    obj = expect_object(value, path)
    check_keys(obj, path, ("type", "rate"))
    param = child(path, "type")
    audio_type = expect_str(obj.get("type", "audio/pcm"), param)
    if audio_type in ("audio/pcmu", "audio/pcma"):
        message = (
            f"Unsupported value: '{audio_type}' is not supported yet; use 'audio/pcm' at 24000 Hz."
        )
        raise ProtocolError("invalid_value", message, param)
    expect_choice(audio_type, param, ("audio/pcm",))
    if "rate" in obj and expect_int(obj["rate"], child(path, "rate")) != 24000:
        raise value_error(child(path, "rate"), "only 24000 is supported.")


def decode_voice(value: object, param: str) -> str:
    """Any voice name is accepted (D19); custom voice objects are not."""
    if isinstance(value, dict):
        raise ProtocolError(
            "invalid_value", f"'{param}' is not supported as an object: pass a voice name.", param
        )
    return expect_str(value, param)


def _decode_session_type(value: object, param: str) -> None:
    if value == "transcription":
        message = "Transcription sessions are not supported; use 'realtime'."
        raise ProtocolError("invalid_value", message, param)
    expect_choice(value, param, ("realtime",))


def _decode_audio(
    value: object, path: str
) -> tuple[Maybe[TranscriptionPatch | None], Maybe[ServerVadPatch | None], Maybe[str]]:
    audio = expect_object(value, path)
    check_keys(audio, path, ("input", "output"))
    transcription: Maybe[TranscriptionPatch | None] = UNSET
    turn_detection: Maybe[ServerVadPatch | None] = UNSET
    voice: Maybe[str] = UNSET
    if "input" in audio:
        path_in = child(path, "input")
        audio_in = expect_object(audio["input"], path_in)
        check_keys(
            audio_in, path_in, ("format", "transcription", "turn_detection", "noise_reduction")
        )
        if "format" in audio_in:
            decode_audio_format(audio_in["format"], child(path_in, "format"))
        if audio_in.get("noise_reduction") is not None:
            raise value_error(child(path_in, "noise_reduction"), "only null is supported.")
        if "transcription" in audio_in:
            transcription = _decode_transcription(
                audio_in["transcription"], child(path_in, "transcription")
            )
        if "turn_detection" in audio_in:
            turn_detection = _decode_turn_detection(
                audio_in["turn_detection"], child(path_in, "turn_detection")
            )
    if "output" in audio:
        path_out = child(path, "output")
        audio_out = expect_object(audio["output"], path_out)
        check_keys(audio_out, path_out, ("format", "voice", "speed"))
        if "format" in audio_out:
            decode_audio_format(audio_out["format"], child(path_out, "format"))
        if "voice" in audio_out:
            voice = decode_voice(audio_out["voice"], child(path_out, "voice"))
        if "speed" in audio_out:
            param = child(path_out, "speed")
            if expect_number(audio_out["speed"], param) != 1.0:
                raise value_error(param, "only 1.0 is supported.")
    return transcription, turn_detection, voice


def _decode_transcription(value: object, path: str) -> TranscriptionPatch | None:
    if value is None:
        return None
    obj = expect_object(value, path)
    check_keys(
        obj,
        path,
        ("model", "language", "prompt"),
        unsupported={
            "delay": "the transcription delay is fixed.",
            "keywords": "keyword hints are not supported.",
            "languages": "pass a single 'language'.",
        },
    )
    model: Maybe[str] = UNSET
    language: Maybe[str] = UNSET
    prompt: Maybe[str] = UNSET
    if "model" in obj:
        model = expect_str(obj["model"], child(path, "model"))
    if "language" in obj:
        language = expect_str(obj["language"], child(path, "language"))
    if "prompt" in obj:
        prompt = expect_str(obj["prompt"], child(path, "prompt"))
    return TranscriptionPatch(model=model, language=language, prompt=prompt)


def _decode_turn_detection(value: object, path: str) -> ServerVadPatch | None:
    if value is None:
        return None
    obj = expect_object(value, path)
    param = child(path, "type")
    vad_type = expect_str(require(obj, "type", path), param)
    if vad_type == "semantic_vad":
        message = "Invalid value: 'semantic_vad'. Supported values are: 'server_vad' (or null)."
        raise ProtocolError("invalid_value", message, param)
    expect_choice(vad_type, param, ("server_vad",))
    check_keys(
        obj,
        path,
        (
            "type",
            "threshold",
            "prefix_padding_ms",
            "silence_duration_ms",
            "create_response",
            "interrupt_response",
            "idle_timeout_ms",
        ),
    )
    if obj.get("idle_timeout_ms") is not None:
        raise value_error(child(path, "idle_timeout_ms"), "only null is supported.")
    threshold: Maybe[float] = UNSET
    prefix_padding_ms: Maybe[int] = UNSET
    silence_duration_ms: Maybe[int] = UNSET
    create_response: Maybe[bool] = UNSET
    interrupt_response: Maybe[bool] = UNSET
    if "threshold" in obj:
        threshold = expect_number(
            obj["threshold"], child(path, "threshold"), minimum=0.0, maximum=1.0
        )
    if "prefix_padding_ms" in obj:
        prefix_padding_ms = expect_int(
            obj["prefix_padding_ms"], child(path, "prefix_padding_ms"), minimum=0
        )
    if "silence_duration_ms" in obj:
        silence_duration_ms = expect_int(
            obj["silence_duration_ms"], child(path, "silence_duration_ms"), minimum=0
        )
    if "create_response" in obj:
        create_response = expect_bool(obj["create_response"], child(path, "create_response"))
    if "interrupt_response" in obj:
        interrupt_response = expect_bool(
            obj["interrupt_response"], child(path, "interrupt_response")
        )
    return ServerVadPatch(
        threshold=threshold,
        prefix_padding_ms=prefix_padding_ms,
        silence_duration_ms=silence_duration_ms,
        create_response=create_response,
        interrupt_response=interrupt_response,
    )


def _decode_tools(value: object, path: str) -> tuple[ToolDef, ...]:
    tools: list[ToolDef] = []
    for index, entry in enumerate(expect_list(value, path)):
        tool_path = element(path, index)
        tool = expect_object(entry, tool_path)
        type_param = child(tool_path, "type")
        tool_type = require(tool, "type", tool_path)
        if tool_type == "mcp":
            raise ProtocolError(
                "invalid_value", f"'{type_param}' is not supported: MCP tools.", type_param
            )
        expect_choice(tool_type, type_param, ("function",))
        check_keys(tool, tool_path, ("type", "name", "description", "parameters"))
        name_param = child(tool_path, "name")
        name = expect_str(require(tool, "name", tool_path), name_param)
        if not name:
            raise value_error(name_param, "the name must not be empty.")
        if any(existing.name == name for existing in tools):
            raise value_error(name_param, f"duplicate tool name '{name}'.")
        description = tool.get("description")
        parameters = tool.get("parameters")
        tools.append(
            ToolDef(
                name=name,
                description=None
                if description is None
                else expect_str(description, child(tool_path, "description")),
                parameters=None
                if parameters is None
                else expect_object(parameters, child(tool_path, "parameters")),
            )
        )
    return tuple(tools)


def _expect_default(obj: Mapping[str, object], path: str, key: str, default: str | None) -> None:
    """Fields accepted only at their GA default."""
    if key in obj and obj[key] != default:
        raise value_error(child(path, key), f"only {json.dumps(default)} is supported.")


def render_session(session_id: str, settings: SessionSettings) -> dict[str, object]:
    """The full effective session, as ``session.created`` and ``session.updated`` carry it."""
    return {
        "type": "realtime",
        "object": "realtime.session",
        "id": session_id,
        "model": settings.model,
        "output_modalities": [settings.output_modality],
        "instructions": settings.instructions,
        "tools": [_render_tool(tool) for tool in settings.tools],
        "tool_choice": settings.tool_choice,
        "max_output_tokens": render_max_output_tokens(settings.max_output_tokens),
        "tracing": None,
        "prompt": None,
        "truncation": "auto",
        "include": None,
        "audio": {
            "input": {
                "format": dict(WIRE_FORMAT),
                "transcription": _render_transcription(settings.transcription),
                "noise_reduction": None,
                "turn_detection": _render_turn_detection(settings.turn_detection),
            },
            "output": {"format": dict(WIRE_FORMAT), "voice": settings.voice, "speed": 1.0},
        },
    }


def render_max_output_tokens(value: int | None) -> int | str:
    return "inf" if value is None else value


def _render_transcription(transcription: Transcription | None) -> dict[str, object] | None:
    if transcription is None:
        return None
    fields = {
        "model": transcription.model,
        "language": transcription.language,
        "prompt": transcription.prompt,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _render_turn_detection(vad: ServerVad | None) -> dict[str, object] | None:
    if vad is None:
        return None
    return {
        "type": "server_vad",
        "threshold": vad.threshold,
        "prefix_padding_ms": vad.prefix_padding_ms,
        "silence_duration_ms": vad.silence_duration_ms,
        "idle_timeout_ms": None,
        "create_response": vad.create_response,
        "interrupt_response": vad.interrupt_response,
    }


def _render_tool(tool: ToolDef) -> dict[str, object]:
    rendered: dict[str, object] = {"type": "function", "name": tool.name}
    if tool.description is not None:
        rendered["description"] = tool.description
    if tool.parameters is not None:
        rendered["parameters"] = dict(tool.parameters)
    return rendered

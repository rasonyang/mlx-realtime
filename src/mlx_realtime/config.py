"""Configuration: one YAML file, environment overrides, strict validation (ARCHITECTURE §10).

Every key can be overridden by an environment variable ``MLXRT__<SECTION>__<KEY>``; a double
underscore marks nesting and the value is parsed as YAML. An unknown key or a bad value stops
startup with an error that names the dotted path of the offending key.

``session_defaults`` is GA-shaped and is validated by the protocol layer when the gateway is
built, so the defaults can never hold a value that a ``session.update`` would reject.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ENV_PREFIX = "MLXRT__"


class ConfigError(Exception):
    """The configuration is invalid; the message names every offending dotted path."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GatewayConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=0, le=65535)
    max_sessions: int = Field(default=8, ge=1)
    api_key: str = ""
    max_session_s: int = Field(default=3600, gt=0)


class LimitsConfig(_Section):
    client_max_message_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    inbox_max_ms: int = Field(default=2000, gt=0)
    inbox_max_commands: int = Field(default=256, gt=0)
    input_audio_buffer_max_ms: int = Field(default=60_000, gt=0)
    outbound_queue_events: int = Field(default=256, ge=2)
    client_write_timeout_ms: int = Field(default=5000, gt=0)
    max_response_audio_ms: int = Field(default=120_000, gt=0)


class OutputConfig(_Section):
    max_delta_ms: int = Field(default=100, gt=0)


class VadConfig(_Section):
    type: Literal["mock"] = "mock"
    min_speech_ms: int = Field(default=150, ge=0)


class SmartTurnConfig(_Section):
    enabled: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_silence_ms: int = Field(default=3000, gt=0)
    window_s: int = Field(default=8, gt=0)


class TurnConfig(_Section):
    vad: VadConfig = VadConfig()
    smart_turn: SmartTurnConfig = SmartTurnConfig()


class SegmenterConfig(_Section):
    first_min_chars: int = Field(default=4, ge=1)
    first_max_wait_ms: int = Field(default=400, ge=0)
    min_chars: int = Field(default=8, ge=1)
    max_chars: int = Field(default=80, ge=2)


class TextConfig(_Section):
    segmenter: SegmenterConfig = SegmenterConfig()


class MockLLMConfig(_Section):
    type: Literal["mock"] = "mock"
    first_token_delay_ms: int = Field(default=0, ge=0)
    token_delay_ms: int = Field(default=20, ge=0)
    greeting: str = "Hello."


class MockASRConfig(_Section):
    type: Literal["mock"] = "mock"
    delay_ms: int = Field(default=0, ge=0)
    transcripts: list[str] = Field(default_factory=lambda: ["你好。"])


class MockTTSConfig(_Section):
    type: Literal["mock"] = "mock"
    first_chunk_delay_ms: int = Field(default=0, ge=0)
    chunk_delay_ms: int = Field(default=0, ge=0)
    chunk_ms: int = Field(default=80, gt=0)
    ms_per_char: int = Field(default=60, gt=0)
    frequency_hz: float = Field(default=440.0, gt=0)


class ProvidersConfig(_Section):
    llm: MockLLMConfig = MockLLMConfig()
    asr: MockASRConfig = MockASRConfig()
    tts: MockTTSConfig = MockTTSConfig()


class ObservabilityConfig(_Section):
    log_level: Literal["debug", "info", "warning", "error"] = "info"


class Config(_Section):
    gateway: GatewayConfig = GatewayConfig()
    limits: LimitsConfig = LimitsConfig()
    output: OutputConfig = OutputConfig()
    session_defaults: dict[str, object] = Field(default_factory=dict[str, object])
    turn: TurnConfig = TurnConfig()
    text: TextConfig = TextConfig()
    providers: ProvidersConfig = ProvidersConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


def load_config(path: Path, environ: Mapping[str, str]) -> Config:
    """Read ``path``, apply ``MLXRT__`` overrides from ``environ`` and validate the result."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        document: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    raw = {str(key): value for key, value in cast(dict[object, object], document).items()}
    overrides = _apply_env_overrides(raw, environ)
    return validate_config(raw, overrides)


def validate_config(
    raw: Mapping[str, object], overrides: Mapping[str, str] | None = None
) -> Config:
    """Validate a raw mapping; ``overrides`` maps dotted paths to the env variable that set them."""
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        lines = [_describe(error["loc"], error["msg"], overrides or {}) for error in exc.errors()]
        raise ConfigError("invalid configuration:\n" + "\n".join(lines)) from None


def _apply_env_overrides(raw: dict[str, object], environ: Mapping[str, str]) -> dict[str, str]:
    applied: dict[str, str] = {}
    for name in sorted(environ):
        if not name.startswith(ENV_PREFIX):
            continue
        keys = name[len(ENV_PREFIX) :].lower().split("__")
        if not all(keys):
            raise ConfigError(f"{name}: malformed override name")
        try:
            value: object = yaml.safe_load(environ[name])
        except yaml.YAMLError as exc:
            raise ConfigError(f"{name}: value is not valid YAML: {exc}") from exc
        node = raw
        for depth, key in enumerate(keys[:-1]):
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                path = ".".join(keys[: depth + 1])
                raise ConfigError(f"{name}: {path} is not a section")
            node = cast(dict[str, object], child)
        node[keys[-1]] = value
        applied[".".join(keys)] = name
    return applied


def _describe(loc: tuple[int | str, ...], message: str, overrides: Mapping[str, str]) -> str:
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else part)
    source = next(
        (env for key, env in overrides.items() if _related(path, key)),
        None,
    )
    suffix = f" (set by {source})" if source else ""
    return f"  {path or '<root>'}: {message}{suffix}"


def _related(path: str, key: str) -> bool:
    """Whether an error at ``path`` concerns the value an override set at ``key``."""
    return path == key or path.startswith(key + ".") or key.startswith(path + ".")

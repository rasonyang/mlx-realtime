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
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    field_validator,
    model_validator,
)

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
    """``mock`` never hears speech (tests, mock runs); ``silero`` runs Silero VAD v6.2.3."""

    type: Literal["mock", "silero"] = "mock"
    model: str = "models/silero_vad.onnx"
    min_speech_ms: int = Field(default=150, ge=0)
    # pipecat's periodic reset of Silero's recurrent state during clear non-speech; null: never.
    model_reset_interval_s: float | None = Field(default=None, gt=0)


class SmartTurnConfig(_Section):
    """Smart Turn v3.2. The defaults are G3's: the fp32 file on CPU, 2 threads, 2 workers."""

    enabled: bool = False
    model: str = "models/smart-turn-v3.2-gpu.onnx"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_silence_ms: int = Field(default=3000, gt=0)
    window_s: int = Field(default=8, gt=0, le=8)
    workers: int = Field(default=2, ge=1)
    intra_op_threads: int = Field(default=2, ge=1)


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


class HttpTimeouts(_Section):
    """The deadlines of ARCHITECTURE §5.2; the first item is the first token or the transcript."""

    connect_ms: int = Field(default=500, gt=0)
    first_item_ms: int = Field(default=5000, gt=0)
    idle_ms: int = Field(default=3000, gt=0)
    total_ms: int = Field(default=60_000, gt=0)
    warmup_ms: int = Field(default=600_000, gt=0)  # a backend may still be loading its model


class MlxLmChatConfig(_Section):
    type: Literal["mlx_lm_chat"]
    base_url: str = "http://127.0.0.1:8101"
    model: str | None = None  # omitted from requests, or exactly the server's --model value
    # Sampling and length: placeholders until the M0 LLM spike (G2) decides them.
    max_completion_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    empty_conversation_cue: str = "(The conversation has just started.)"
    system_note: str = "(System note: {text})"  # how a later system item reaches the model
    max_in_flight: int = Field(default=8, ge=1)
    queue_timeout_ms: int = Field(default=1000, ge=0)
    timeouts: HttpTimeouts = HttpTimeouts()

    @field_validator("system_note")
    @classmethod
    def _has_text_field(cls, value: str) -> str:
        if "{text}" not in value:
            raise ValueError("must contain {text}")
        return value


class MlxAudioAsrConfig(_Section):
    type: Literal["mlx_audio_transcription"]
    base_url: str = "http://127.0.0.1:8102"
    model: str = "mlx-community/Qwen3-ASR-1.7B-8bit"
    max_in_flight: int = Field(default=4, ge=1)
    queue_timeout_ms: int = Field(default=1000, ge=0)
    timeouts: HttpTimeouts = HttpTimeouts(first_item_ms=5000, idle_ms=2000, total_ms=10_000)


_PROVIDER_TAGS = frozenset({"mock", "mlx_lm_chat", "mlx_audio_transcription"})


def _provider_type(value: object) -> str:
    """The ``type`` of a provider section; sections without one are the M1 mocks."""
    if isinstance(value, dict):
        return str(cast(dict[str, object], value).get("type", "mock"))
    return str(getattr(value, "type", "mock"))


type LLMConfig = Annotated[
    Annotated[MockLLMConfig, Tag("mock")] | Annotated[MlxLmChatConfig, Tag("mlx_lm_chat")],
    Discriminator(_provider_type),
]
type ASRConfig = Annotated[
    Annotated[MockASRConfig, Tag("mock")]
    | Annotated[MlxAudioAsrConfig, Tag("mlx_audio_transcription")],
    Discriminator(_provider_type),
]


class ProvidersConfig(_Section):
    llm: LLMConfig = MockLLMConfig()
    asr: ASRConfig = MockASRConfig()
    tts: MockTTSConfig = MockTTSConfig()


class HfRepo(_Section):
    """A Hugging Face model repository pinned to a revision (a commit hash, or a branch or tag)."""

    repo: str
    revision: str


class BackendConfig(_Section):
    """One model process, started by ``mlxrt dev`` (ARCHITECTURE §2)."""

    project: str  # the uv project the process runs in, e.g. backends/llm
    command: list[str] = Field(min_length=1)  # run as: uv run --locked --project <project> ...
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    model: HfRepo | None = None  # served model: passed as --model <its local snapshot>
    fetch: list[HfRepo] = Field(default_factory=list[HfRepo])  # models requested by name
    args: list[str] = Field(default_factory=list[str])
    env: dict[str, str] = Field(default_factory=dict[str, str])


class BackendsConfig(_Section):
    env: dict[str, str] = Field(default_factory=dict[str, str])  # for every backend process
    llm: BackendConfig | None = None
    asr: BackendConfig | None = None
    tts: BackendConfig | None = None

    def processes(self) -> dict[str, BackendConfig]:
        named = {"llm": self.llm, "asr": self.asr, "tts": self.tts}
        return {name: backend for name, backend in named.items() if backend is not None}


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
    backends: BackendsConfig = BackendsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @model_validator(mode="after")
    def _llm_model_is_exact_or_omitted(self) -> Self:
        llm, backend = self.providers.llm, self.backends.llm
        if (
            isinstance(llm, MlxLmChatConfig)
            and llm.model is not None
            and backend is not None
            and backend.model is not None
        ):
            # The server runs on a local snapshot path; any other name makes it look the
            # model up on Hugging Face and swap models (ARCHITECTURE §5.2).
            raise ValueError("providers.llm.model must be omitted when backends.llm.model is set")
        return self


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
    previous: int | str | None = None
    for part in loc:
        # A discriminated union adds its tag to the location: providers.llm.mlx_lm_chat.x.
        if part in _PROVIDER_TAGS and previous in ("llm", "asr", "tts"):
            previous = part
            continue
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else part)
        previous = part
    source = next(
        (env for key, env in overrides.items() if _related(path, key)),
        None,
    )
    suffix = f" (set by {source})" if source else ""
    return f"  {path or '<root>'}: {message}{suffix}"


def _related(path: str, key: str) -> bool:
    """Whether an error at ``path`` concerns the value an override set at ``key``."""
    return path == key or path.startswith(key + ".") or key.startswith(path + ".")

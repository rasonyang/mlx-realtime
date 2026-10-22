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
    """Output audio: deltas of at most ``max_delta_ms``, sent as soon as they are synthesized but
    never more than ``max_lead_ms`` ahead of real time, counted from a response's first delta
    (D6). A delta stays under 1 s, so paced deltas never leave a 2 s gap (AICC's watchdog)."""

    max_delta_ms: int = Field(default=100, gt=0, le=1000)
    max_lead_ms: int = Field(default=10_000, gt=0)

    @model_validator(mode="after")
    def _lead_holds_a_delta(self) -> Self:
        if self.max_lead_ms < self.max_delta_ms:
            raise ValueError("max_lead_ms must be at least max_delta_ms")
        return self


DEFAULT_VERBATIM_PREFIXES = (
    "Say exactly this, word for word, and add nothing else:\n",
    "请一字不差地说出下面这句话，不要添加任何其它内容：\n",
)
"""AICC's line directions (``SayExactly``), in English and Chinese."""


class LinesConfig(_Section):
    """Verbatim lines (D16): a response whose own ``instructions`` start with one of these
    prefixes skips the LLM and speaks the rest as written. An empty list turns lines off."""

    verbatim_prefixes: list[str] = Field(default_factory=lambda: list(DEFAULT_VERBATIM_PREFIXES))

    @field_validator("verbatim_prefixes")
    @classmethod
    def _no_empty_prefix(cls, value: list[str]) -> list[str]:
        if any(not prefix for prefix in value):
            raise ValueError("a prefix must not be empty")
        return value


type SegmentLang = Literal["zh", "en"]
"""A segment's language tag (mixed text is ``zh``); the TTS models map it to their lang_code."""


class VoiceConfig(_Section):
    """How the TTS models speak one session voice name (``audio.output.voice``, D19)."""

    speaker: str = Field(min_length=1)  # the primary model's speaker
    # "segment": each segment's language tag through the primary model's lang_codes. Any
    # other value is sent as the lang_code itself ("auto", "chinese", "english", ...).
    lang: str = Field(default="segment", min_length=1)
    # The fallback model's voice for each segment language (Kokoro voices are per language).
    fallback: dict[SegmentLang, str] = Field(default_factory=dict[SegmentLang, str])


def _builtin_voices() -> dict[str, VoiceConfig]:
    """Qwen3-TTS-12Hz-0.6B-CustomVoice's Mandarin and English speakers, each with Kokoro voices
    of the same gender. ``dylan`` and ``eric`` are left out: with chinese or auto they turn
    into Beijing and Sichuan dialect."""

    def voice(speaker: str, zh: str, en: str) -> VoiceConfig:
        return VoiceConfig(speaker=speaker, fallback={"zh": zh, "en": en})

    return {
        "vivian": voice("vivian", "zf_xiaobei", "af_heart"),
        "serena": voice("serena", "zf_xiaoxiao", "af_bella"),
        "uncle_fu": voice("uncle_fu", "zm_yunjian", "am_adam"),
        "ryan": voice("ryan", "zm_yunxi", "am_michael"),
        "aiden": voice("aiden", "zm_yunyang", "am_puck"),
    }


class VoicesConfig(_Section):
    """The session voice names this service offers (D19). Any other name a client sends is
    accepted, mapped to ``default`` with a warning, and the effective name is echoed."""

    default: str = "vivian"
    map: dict[str, VoiceConfig] = Field(default_factory=_builtin_voices)

    @model_validator(mode="after")
    def _default_is_mapped(self) -> Self:
        if not self.map:
            raise ValueError("voices.map must name at least one voice")
        if self.default not in self.map:
            raise ValueError(f"the default voice '{self.default}' is not in voices.map")
        return self


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


class NormalizerConfig(_Section):
    """TTS input normalization by category (ARCHITECTURE §7); transcripts keep the original
    text. Phone numbers and IDs are on, the rest off, until the S9 probe decides (G5)."""

    enabled: bool = True
    numbers: bool = False  # cardinal, decimal, percent; read in Chinese or English by context
    money: bool = False
    dates: bool = False  # dates and times
    phones: bool = True  # mobile, landline, 400/800: digit by digit, in groups
    ids: bool = True  # order and tracking IDs: character by character
    units: bool = False
    spacing: bool = False  # zh/en spacing
    phone_one_as_yao: bool = True  # read 1 as 幺 in Chinese phone numbers


class TextConfig(_Section):
    segmenter: SegmenterConfig = SegmenterConfig()
    normalizer: NormalizerConfig = NormalizerConfig()


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


type TtsFamily = Literal["qwen3_tts", "kokoro"]
"""How a TTS model is asked: Qwen3-TTS takes a speaker name, Kokoro a voice-pack file."""

_TTS_TIMEOUTS = HttpTimeouts(first_item_ms=1500, idle_ms=1000, total_ms=60_000, warmup_ms=300_000)


class TtsModelConfig(_Section):
    """One model of the TTS backend, and the fields its requests carry (ARCHITECTURE §5.3).

    ``streaming_interval``: seconds of audio per streamed chunk (Qwen3-TTS streams
    ``max(1, int(interval × 12.5))`` codec frames of 80 ms; at 0.32 the first chunk arrived
    about 110 ms after the request in M0). The sampling fields are sent only when set; unset,
    the model's own generation_config.json applies.
    """

    model: str = Field(min_length=1)  # the name sent in requests
    family: TtsFamily = "qwen3_tts"
    lang_codes: dict[SegmentLang, str] = Field(default_factory=dict[SegmentLang, str])
    streaming_interval: float = Field(default=0.32, gt=0)
    voices_dir: str | None = None  # Kokoro voice packs; None: voices/ of the pinned snapshot
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)  # None: the server's 1200 codec frames
    timeouts: HttpTimeouts = _TTS_TIMEOUTS


class TtsFallbackConfig(TtsModelConfig):
    """The model a segment is retried on when the primary fails before its first chunk (D13).
    After ``breaker_failures`` consecutive primary failures every segment goes straight to the
    fallback for ``cooldown_s``."""

    enabled: bool = True
    breaker_failures: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=30.0, gt=0)


class MlxAudioSpeechConfig(_Section):
    """``mlx_audio.server``'s streaming speech route, one model per request (primary, fallback).
    The server runs one stream at a time, so requests queue in the gateway, at most
    ``queue_timeout_ms`` (then ``backend_busy``)."""

    type: Literal["mlx_audio_speech"]
    base_url: str = "http://127.0.0.1:8103"
    sample_rate: int = Field(default=24_000, gt=0)  # the PCM the server streams
    max_in_flight: int = Field(default=1, ge=1)
    queue_timeout_ms: int = Field(default=2000, ge=0)
    warmup_texts: dict[SegmentLang, str] = Field(
        default_factory=lambda: {
            "zh": "您好，请问有什么可以帮您？",
            "en": "Hello, how can I help you?",
        }
    )
    primary: TtsModelConfig = TtsModelConfig(
        model="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", family="qwen3_tts"
    )
    fallback: TtsFallbackConfig = TtsFallbackConfig(
        model="mlx-community/Kokoro-82M-bf16",
        family="kokoro",
        timeouts=HttpTimeouts(first_item_ms=2000, idle_ms=2000, total_ms=60_000, warmup_ms=300_000),
    )

    @field_validator("warmup_texts")
    @classmethod
    def _both_languages(cls, value: dict[SegmentLang, str]) -> dict[SegmentLang, str]:
        if set(value) != {"zh", "en"} or not all(text.strip() for text in value.values()):
            raise ValueError("needs a non-empty zh and en sentence")
        return value


_PROVIDER_TAGS = frozenset({"mock", "mlx_lm_chat", "mlx_audio_transcription", "mlx_audio_speech"})


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
type TTSConfig = Annotated[
    Annotated[MockTTSConfig, Tag("mock")]
    | Annotated[MlxAudioSpeechConfig, Tag("mlx_audio_speech")],
    Discriminator(_provider_type),
]


class ProvidersConfig(_Section):
    llm: LLMConfig = MockLLMConfig()
    asr: ASRConfig = MockASRConfig()
    tts: TTSConfig = MockTTSConfig()


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

    def pins(self) -> dict[str, str]:
        """Every pinned model repository (served or fetched by name) and its revision."""
        pinned: dict[str, str] = {}
        for backend in self.processes().values():
            served = [] if backend.model is None else [backend.model]
            for pin in [*backend.fetch, *served]:
                pinned[pin.repo] = pin.revision
        return pinned


class ObservabilityConfig(_Section):
    log_level: Literal["debug", "info", "warning", "error"] = "info"


class Config(_Section):
    gateway: GatewayConfig = GatewayConfig()
    limits: LimitsConfig = LimitsConfig()
    output: OutputConfig = OutputConfig()
    lines: LinesConfig = LinesConfig()
    voices: VoicesConfig = VoicesConfig()
    session_defaults: dict[str, object] = Field(default_factory=dict[str, object])
    turn: TurnConfig = TurnConfig()
    text: TextConfig = TextConfig()
    providers: ProvidersConfig = ProvidersConfig()
    backends: BackendsConfig = BackendsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @model_validator(mode="after")
    def _voices_have_fallback_voices(self) -> Self:
        tts = self.providers.tts
        if not isinstance(tts, MlxAudioSpeechConfig):
            return self
        fallback = tts.fallback
        if not fallback.enabled or fallback.family != "kokoro":
            return self
        for name, voice in self.voices.map.items():
            missing = sorted({"zh", "en"} - set(voice.fallback))
            if missing:
                raise ValueError(
                    f"voices.map.{name}.fallback needs a Kokoro voice for {', '.join(missing)} "
                    "while providers.tts.fallback is enabled"
                )
        return self

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

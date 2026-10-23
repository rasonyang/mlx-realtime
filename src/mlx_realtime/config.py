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


class SilenceTrimConfig(_Section):
    """Edge silence of each synthesized segment (M0 S3: Qwen3-TTS adds ~0.45 s before and ~0.5 s
    after every output). An energy gate on ``frame_ms`` frames (voiced from ``threshold_dbfs``
    RMS) keeps at most ``max_lead_ms`` before the first voiced frame and ``max_tail_ms`` after
    the last; inner pauses are kept whole. The first voiced frame goes out at once."""

    enabled: bool = True
    threshold_dbfs: float = Field(default=-50.0, lt=0)
    frame_ms: int = Field(default=20, gt=0, le=100)
    max_lead_ms: int = Field(default=60, ge=0)
    max_tail_ms: int = Field(default=100, ge=0)


class AudioReserveConfig(_Section):
    """The per-session audio reserve (M6, DECISIONS 2026-10-23): while a response still has
    segments to come, the sender holds back up to ``hold_ms`` of its synthesized audio, so deltas
    keep flowing while the next segment waits in the shared TTS queue. A held delta goes out
    when the wire has been quiet for ``keepalive_ms`` (under AICC's 2 s gap watchdog), or when
    the client's modelled buffer drops below ``min_lead_ms`` (so playback never starves for
    it). Once the last segment is synthesized, everything left goes out paced as usual."""

    enabled: bool = True
    hold_ms: int = Field(default=1500, ge=0)
    keepalive_ms: int = Field(default=1000, gt=0, le=1900)
    min_lead_ms: int = Field(default=400, ge=0)


class OutputConfig(_Section):
    """Output audio: deltas of at most ``max_delta_ms``, sent as soon as they are synthesized but
    never more than ``max_lead_ms`` ahead of real time, counted from a response's first delta
    (D6). A delta stays under 1 s, so paced deltas never leave a 2 s gap (AICC's watchdog)."""

    max_delta_ms: int = Field(default=100, gt=0, le=1000)
    max_lead_ms: int = Field(default=10_000, gt=0)
    trim_silence: SilenceTrimConfig = SilenceTrimConfig()
    reserve: AudioReserveConfig = AudioReserveConfig()

    @model_validator(mode="after")
    def _lead_holds_a_delta(self) -> Self:
        if self.max_lead_ms < self.max_delta_ms:
            raise ValueError("max_lead_ms must be at least max_delta_ms")
        return self


type ResampleQuality = Literal["QQ", "LQ", "MQ", "HQ", "VHQ"]


class G711Config(_Section):
    """G.711 sessions (``audio/pcmu``, ``audio/pcma`` at 8 kHz, D3; M7): the soxr quality of each
    resampling step. ``input_quality`` resamples the caller's 8 kHz audio to the canonical
    16 kHz before the echo canceller and the VAD (M0 S7: LQ lags 32 ms on average, HQ 64 ms; LQ
    rolls off above 3 kHz, -8 dB at 3.4 kHz); the echo reference of G.711 output takes the
    same path. ``output_quality`` resamples the TTS's 24 kHz to 8 kHz (its lag delays no
    delta: a segment's tail is flushed when it ends)."""

    input_quality: ResampleQuality = "LQ"
    output_quality: ResampleQuality = "HQ"


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
    # The primary model's speaker for segments of one language, instead of `speaker` (S3:
    # vivian speaks zh and mixed segments, aiden en segments).
    speakers: dict[SegmentLang, str] = Field(default_factory=dict[SegmentLang, str])
    # "segment": each segment's language tag through the primary model's lang_codes. Any
    # other value is sent as the lang_code itself ("auto", "chinese", "english", ...).
    lang: str = Field(default="segment", min_length=1)
    # The fallback model's voice for each segment language (Kokoro voices are per language).
    fallback: dict[SegmentLang, str] = Field(default_factory=dict[SegmentLang, str])


def _builtin_voices() -> dict[str, VoiceConfig]:
    """Qwen3-TTS-12Hz-0.6B-CustomVoice's Mandarin and English speakers, each with Kokoro voices
    of the same gender. ``dylan`` and ``eric`` are left out: with chinese or auto they turn
    into Beijing and Sichuan dialect. ``ryan`` is left out: M0 S3 found the runaway defect
    (mlx-audio#1002) follows it on every model. The default voice, vivian, hands English
    segments to aiden (S3: vivian for zh and mixed, aiden for en)."""

    def voice(
        speaker: str, zh: str, en: str, speakers: dict[SegmentLang, str] | None = None
    ) -> VoiceConfig:
        return VoiceConfig(speaker=speaker, speakers=speakers or {}, fallback={"zh": zh, "en": en})

    return {
        "vivian": voice("vivian", "zf_xiaobei", "af_heart", {"en": "aiden"}),
        "serena": voice("serena", "zf_xiaoxiao", "af_bella"),
        "uncle_fu": voice("uncle_fu", "zm_yunjian", "am_adam"),
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


class AecConfig(_Section):
    """The per-session echo canceller (D17; engine and defaults from M0's G7).

    Its reference is the session's own output audio, resampled to 16 kHz and placed on a model
    of the client's playout. The defaults model AICC (voice/rtp.go): a 20 ms tick, playback
    starting at ``client_prebuffer_ms`` of queued frames (3) or that long after the queue was
    last empty, and a re-buffer after more than ``client_underrun_ticks`` empty ticks. The
    reference leads the modelled playout by ``reference_lead_ms`` so it precedes its echo.
    A GCC-PHAT delay estimator re-aligns the reference only when the residual delay leaves
    [``min_residual_delay_ms``, ``max_residual_delay_ms``], and learns AICC's tick phase.
    """

    enabled: bool = True
    engine: Literal["webrtc_aec3"] = "webrtc_aec3"  # AEC3 via pywebrtc-audio 0.2.0
    # What the ASR gets: the canceller's output, or the input as received. Speech and
    # end-of-turn detection always run on the output. Pending M0's double-talk ΔCER.
    asr_input: Literal["processed", "raw"] = "processed"
    workers: int = Field(default=2, ge=1)  # the canceller executor's threads
    client_tick_ms: int = Field(default=20, gt=0)
    client_prebuffer_ms: int = Field(default=60, ge=0)
    client_underrun_ticks: int = Field(default=2, ge=0)
    reference_lead_ms: int = Field(default=60, ge=0)
    learn_tick_phase: bool = True
    delay_guard: bool = True  # the GCC-PHAT guard (the tick phase is learned from it)
    min_residual_delay_ms: int = Field(default=5, ge=0)
    max_residual_delay_ms: int = Field(default=450, gt=0)
    # With no echo on the line, AEC3's suppressor takes a caller talking over our audio for
    # echo and silences them (M4). Once the delay estimator has seen this much playing
    # reference without any echo of it, the canceller gets a silent reference until echo shows
    # up again (null: never bypass).
    bypass_without_echo_ms: int | None = Field(default=1500, gt=0)

    @model_validator(mode="after")
    def _keep_range(self) -> Self:
        if self.min_residual_delay_ms >= self.max_residual_delay_ms:
            raise ValueError("min_residual_delay_ms must be below max_residual_delay_ms")
        if not self.min_residual_delay_ms <= self.reference_lead_ms <= self.max_residual_delay_ms:
            raise ValueError("reference_lead_ms must lie within the residual delay keep range")
        return self


class BargeInConfig(_Section):
    """The second line of defence against echo, while the client is modelled as playing (D17).

    - ``echo_guard_ms``: in the first part of a response's playback, a speech onset never
      interrupts; speech that began there counts only once the guard is over (0: off).
    - ``min_speech_ms_while_playing``: during playback, speech must last this long before
      ``speech_started`` (null: ``turn.vad.min_speech_ms`` applies throughout).
    - ``echo_turn_suppression``: a committed turn whose transcript closely matches assistant
      text played around it (normalized overlap of at least ``echo_match_threshold``, counted in
      CJK characters and Latin words; transcripts of at least ``echo_match_min_units`` of them) is
      treated like noise (D7): its response completes with no output and no LLM call, and the
      LLM never sees the turn.
    """

    echo_guard_ms: int = Field(default=2000, ge=0)
    min_speech_ms_while_playing: int | None = Field(default=400, ge=0)
    echo_turn_suppression: bool = True
    echo_match_threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    echo_match_min_units: int = Field(default=4, ge=1)
    # Assistant audio modelled as playing up to this long before the turn's audio still counts.
    echo_match_window_ms: int = Field(default=3000, ge=0)

    @property
    def active(self) -> bool:
        """Whether any rule needs the playout model."""
        return (
            self.echo_guard_ms > 0
            or self.min_speech_ms_while_playing is not None
            or self.echo_turn_suppression
        )


class SegmenterConfig(_Section):
    first_min_chars: int = Field(default=4, ge=1)
    first_max_wait_ms: int = Field(default=400, ge=0)
    min_chars: int = Field(default=8, ge=1)
    max_chars: int = Field(default=80, ge=2)


class NormalizerConfig(_Section):
    """TTS input normalization by category (ARCHITECTURE §7); transcripts keep the original
    text. The G5 scope (M0 S9) is the default: phone numbers (with 尾号 and extensions), IDs,
    e-mail addresses, cardinals of ``min_cardinal_digits`` or more digits, and in English text
    numeric dates and amounts from ``en_amount_min``. Money, decimals, percentages, ordinals,
    dates and times, units, URLs and brand names stay as written. The keys are the fields of
    ``NormalizerOptions``."""

    enabled: bool = True
    numbers: bool = False  # every free-standing number: cardinal, decimal, percent, ordinal
    cardinals: bool = True  # free-standing plain cardinals of min_cardinal_digits or more
    min_cardinal_digits: int = Field(default=5, ge=1)
    money: bool = False
    dates: bool = False  # dates and times
    phones: bool = True  # every form, digit by digit, with 尾号 and extensions
    ids: bool = True  # order and tracking IDs: character by character
    emails: bool = True  # "@" and "." spoken
    units: bool = False
    spacing: bool = False  # zh/en spacing
    en_dates: bool = True  # numeric dates read in English text, even with dates off
    en_amounts: bool = True  # amounts of en_amount_min or more in English text, money off
    en_amount_min: int = Field(default=1000, ge=0)
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


PENDING_TOOL_OUTPUT = '{"status": "pending", "note": "The result has not arrived yet."}'
"""The placeholder result of a call that is still unanswered when an LLM request is built."""


class MlxLmChatConfig(_Section):
    type: Literal["mlx_lm_chat"]
    base_url: str = "http://127.0.0.1:8101"
    model: str | None = None  # omitted from requests, or exactly the server's --model value
    # Sampling and length (G2, M0 S1): requests with tools attached go at tool_temperature /
    # tool_top_p (greedy: 19/20 expected calls), the others at temperature / top_p, which match
    # the server's flags (--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0).
    max_completion_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    tool_temperature: float = Field(default=0.0, ge=0.0)
    tool_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # A call the server's parser dropped (finish_reason tool_calls, no call) before any text is
    # asked for once more, in the same response, at temperature / top_p (M0 S1).
    retry_dropped_call: bool = True
    # Warm the server's prompt cache when a session's instructions or tools change (M0 S2), at
    # most max_warmups_in_flight warm-ups at a time (one per prefix).
    warm_on_update: bool = True
    max_warmups_in_flight: int = Field(default=1, ge=1)
    empty_conversation_cue: str = "(The call has connected. Greet the caller as instructed.)"
    system_note: str = "(System note: {text})"  # how a later system item reaches the model
    # M5 history shaping for the Qwen template (providers/chat_wire.py). A tool result always
    # follows its call. tool_interlude: an assistant text between a call and its result (AICC's
    # wait line) is merged into the call's message (merge) or placed after the result
    # (after_output). unanswered_calls: a call with no result yet gets pending_tool_output as
    # a placeholder result (pending), is replayed without one (keep), or is left out (drop).
    tool_interlude: Literal["merge", "after_output"] = "merge"
    unanswered_calls: Literal["pending", "keep", "drop"] = "pending"
    pending_tool_output: str = Field(default=PENDING_TOOL_OUTPUT, min_length=1)
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


class TokenCapConfig(_Section):
    """The codec frames (12.5 per second) a segment may take: ``base`` plus ``per_cjk`` per CJK
    character and ``per_other`` per other letter or digit, never more than the model's
    ``max_tokens`` (or the server's 1200). Normal speech needs about 3 frames per CJK
    character and 1 per Latin letter; the defaults leave twice that, plus 2 s for the edge
    silence."""

    base: int = Field(default=25, ge=1)
    per_cjk: float = Field(default=6.0, gt=0)
    per_other: float = Field(default=2.0, gt=0)


class TtsModelConfig(_Section):
    """One model of the TTS backend, and the fields its requests carry (ARCHITECTURE §5.3).

    ``streaming_interval``: seconds of audio per streamed chunk (Qwen3-TTS streams
    ``max(1, int(interval × 12.5))`` codec frames of 80 ms; M0 S3 chose 0.16, whose first chunk
    came about 76 ms after the request, uncapped). The sampling fields are sent only when set;
    unset, the model's own generation_config.json applies. ``token_cap`` bounds ``max_tokens``
    by the text (Qwen3-TTS only).
    """

    model: str = Field(min_length=1)  # the name sent in requests
    family: TtsFamily = "qwen3_tts"
    lang_codes: dict[SegmentLang, str] = Field(default_factory=dict[SegmentLang, str])
    streaming_interval: float = Field(default=0.16, gt=0)  # S3: 2 codec frames per chunk
    voices_dir: str | None = None  # Kokoro voice packs; None: voices/ of the pinned snapshot
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)  # None: the server's 1200 codec frames
    # A max_tokens per request in proportion to its text (S3: a runaway otherwise runs to the
    # 1200-frame limit, 96 s); None sends max_tokens alone. Qwen3-TTS only.
    token_cap: TokenCapConfig | None = TokenCapConfig()
    timeouts: HttpTimeouts = _TTS_TIMEOUTS


class TtsFallbackConfig(TtsModelConfig):
    """The model a segment is retried on when the primary fails before its first chunk (D13).
    After ``breaker_failures`` consecutive primary failures every segment goes straight to the
    fallback for ``cooldown_s``."""

    enabled: bool = True
    breaker_failures: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=30.0, gt=0)
    # D13 (owner review pending): Kokoro garbles English inside Chinese (S3: 15.9% errors on
    # mixed text), so by default it never speaks a segment with Latin letters: a primary
    # failure on one fails that segment instead.
    latin_segments: bool = False


class MlxAudioSpeechConfig(_Section):
    """``mlx_audio.server``'s streaming speech route, one model per request (primary, fallback).
    The server runs one stream at a time, so requests queue in the gateway.

    ``scheduling`` orders that queue (M6): ``edf`` serves first the request whose session runs
    out of audio soonest (a first segment, whose session has no audio yet, counts from when its
    response started, so it goes ahead of every session that still has audio); ``fifo`` serves
    in arrival order. ``first_segment_grace_ms`` places a first segment's deadline that long
    after its response started (0: before every session that still has audio). A request waits
    at most ``queue_timeout_ms`` past its deadline (then ``backend_busy``)."""

    type: Literal["mlx_audio_speech"]
    base_url: str = "http://127.0.0.1:8103"
    sample_rate: int = Field(default=24_000, gt=0)  # the PCM the server streams
    max_in_flight: int = Field(default=1, ge=1)
    queue_timeout_ms: int = Field(default=2000, ge=0)
    scheduling: Literal["edf", "fifo"] = "edf"
    first_segment_grace_ms: int = Field(default=0, ge=0, le=10_000)
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
    """Logs, ``/metrics`` (always served) and ``GET /debug/turns`` (ARCHITECTURE §9).

    - ``quantile_window_turns``: the windowed P50/P95 gauges cover each stage's last N turns.
    - ``debug_endpoints``: serve ``/debug/turns``, each turn's marks and stage durations for
      the benchmark (off by default; it is unauthenticated, like ``/metrics``). It keeps the
      ``debug_sessions`` most recent sessions, and at most ``debug_turns_per_session`` turns
      of each.
    """

    log_level: Literal["debug", "info", "warning", "error"] = "info"
    quantile_window_turns: int = Field(default=200, ge=1, le=100_000)
    debug_endpoints: bool = False
    debug_sessions: int = Field(default=64, ge=1, le=10_000)
    debug_turns_per_session: int = Field(default=500, ge=1, le=100_000)


class Config(_Section):
    gateway: GatewayConfig = GatewayConfig()
    limits: LimitsConfig = LimitsConfig()
    output: OutputConfig = OutputConfig()
    g711: G711Config = G711Config()
    lines: LinesConfig = LinesConfig()
    voices: VoicesConfig = VoicesConfig()
    session_defaults: dict[str, object] = Field(default_factory=dict[str, object])
    turn: TurnConfig = TurnConfig()
    aec: AecConfig = AecConfig()
    barge_in: BargeInConfig = BargeInConfig()
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

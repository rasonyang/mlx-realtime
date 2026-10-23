from __future__ import annotations

from pathlib import Path

import pytest

from mlx_realtime.config import (
    DEFAULT_VERBATIM_PREFIXES,
    AudioReserveConfig,
    ConfigError,
    MlxAudioAsrConfig,
    MlxAudioSpeechConfig,
    MlxLmChatConfig,
    MockASRConfig,
    MockLLMConfig,
    NormalizerConfig,
    load_config,
    validate_config,
)
from mlx_realtime.providers.fallback import FallbackTTS
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.mock import EchoLLM, SineTTS
from mlx_realtime.providers.registry import build_providers
from mlx_realtime.server.gateway import Gateway, normalizer_options
from mlx_realtime.text.normalize import NormalizerOptions
from tests.speech import needs_models

REPO = Path(__file__).parents[2]
REPO_CONFIG = REPO / "config.yaml"
MOCK_CONFIG = REPO / "config.mock.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@needs_models
def test_repository_config_points_at_the_real_backends() -> None:
    config = load_config(REPO_CONFIG, {})
    gateway = Gateway(config)  # builds the providers and the CPU detectors; starts nothing
    assert config.gateway.port == 8000
    assert isinstance(gateway.providers.llm, MlxLmChatLLM)
    assert isinstance(gateway.providers.asr, MlxAudioASR)
    assert (config.turn.vad.type, config.turn.smart_turn.enabled) == ("silero", True)
    assert config.turn.smart_turn.model.endswith("smart-turn-v3.2-gpu.onnx")  # G3
    assert (config.turn.smart_turn.workers, config.turn.smart_turn.intra_op_threads) == (2, 2)
    assert gateway.defaults.turn_detection is not None
    assert gateway.defaults.turn_detection.silence_duration_ms == 200  # D4
    assert gateway.defaults.transcription is not None  # D15
    assert set(config.backends.processes()) == {"llm", "asr", "tts"}
    llm = config.backends.llm
    assert llm is not None
    assert llm.args[:2] == ["--chat-template-args", '{"enable_thinking":false}']
    assert isinstance(config.providers.llm, MlxLmChatConfig)
    assert config.providers.llm.model is None  # the backend serves a local snapshot
    assert isinstance(config.providers.asr, MlxAudioAsrConfig)
    asr = config.backends.asr
    assert asr is not None
    assert [pin.repo for pin in asr.fetch] == [config.providers.asr.model]
    # M3: the TTS backend, Qwen3-TTS with Kokoro behind it (D13), both pinned for the tts process.
    assert isinstance(gateway.providers.tts, FallbackTTS)
    tts = config.providers.tts
    assert isinstance(tts, MlxAudioSpeechConfig)
    assert (tts.primary.family, tts.fallback.family) == ("qwen3_tts", "kokoro")
    assert tts.max_in_flight == 1  # the server runs one stream at a time
    backend = config.backends.tts
    assert backend is not None
    assert [pin.repo for pin in backend.fetch] == [tts.primary.model, tts.fallback.model]
    assert gateway.defaults.voice == config.voices.default == "vivian"
    assert config.output.max_lead_ms == 10_000  # D6
    assert tuple(config.lines.verbatim_prefixes) == DEFAULT_VERBATIM_PREFIXES  # D16
    # B2's G5 scope: config.yaml spells out the defaults, and they are the normalizer's own.
    normalizer = config.text.normalizer
    assert normalizer == NormalizerConfig()
    assert (normalizer.phones, normalizer.ids, normalizer.emails, normalizer.cardinals) == (
        True,
        True,
        True,
        True,
    )
    assert (normalizer.numbers, normalizer.money, normalizer.dates, normalizer.units) == (
        False,
        False,
        False,
        False,
    )
    assert (normalizer.min_cardinal_digits, normalizer.en_amount_min) == (5, 1000)
    assert normalizer_options(normalizer) == NormalizerOptions()
    # M4: D17 as G7 decided it.
    aec = config.aec
    assert (aec.enabled, aec.engine, aec.asr_input) == (True, "webrtc_aec3", "processed")
    assert (aec.client_tick_ms, aec.client_prebuffer_ms, aec.client_underrun_ticks) == (20, 60, 2)
    assert (aec.reference_lead_ms, aec.learn_tick_phase) == (60, True)
    assert (aec.min_residual_delay_ms, aec.max_residual_delay_ms) == (5, 450)
    assert gateway.echo is not None
    barge_in = config.barge_in
    assert (barge_in.echo_guard_ms, barge_in.min_speech_ms_while_playing) == (2000, 400)
    assert barge_in.echo_turn_suppression


@needs_models
def test_mock_config_has_mock_providers_and_the_real_vad() -> None:
    config = load_config(MOCK_CONFIG, {})
    gateway = Gateway(config)
    assert isinstance(gateway.providers.llm, EchoLLM)
    assert isinstance(gateway.providers.tts, SineTTS)
    assert config.backends.processes() == {}
    # M4: AICC's barge-in scenarios need a VAD that hears the caller, and the barge-in rules;
    # the canceller is off (AEC3 silences a caller talking over the mock's sine tone).
    assert (config.turn.vad.type, config.turn.smart_turn.enabled) == ("silero", False)
    assert not config.aec.enabled
    assert gateway.echo is None
    assert config.barge_in.active
    assert gateway.defaults.voice == "vivian"


def test_echo_defaults_are_g7s() -> None:
    config = validate_config({})
    assert config.aec.enabled
    assert (config.aec.reference_lead_ms, config.aec.max_residual_delay_ms) == (60, 450)
    assert (config.barge_in.echo_guard_ms, config.barge_in.min_speech_ms_while_playing) == (
        2000,
        400,
    )


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (
            {"aec": {"min_residual_delay_ms": 500, "max_residual_delay_ms": 450}},
            "min_residual_delay_ms must be below max_residual_delay_ms",
        ),
        ({"aec": {"reference_lead_ms": 500}}, "reference_lead_ms must lie within"),
        ({"aec": {"engine": "speex"}}, "aec.engine"),
        ({"aec": {"asr_input": "both"}}, "aec.asr_input"),
        ({"barge_in": {"echo_match_threshold": 0}}, "barge_in.echo_match_threshold"),
        ({"barge_in": {"echo_guard_ms": -1}}, "barge_in.echo_guard_ms"),
    ],
)
def test_echo_settings_are_validated(section: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_config(section)


def test_every_barge_in_rule_can_be_switched_off() -> None:
    config = validate_config(
        {
            "barge_in": {
                "echo_guard_ms": 0,
                "min_speech_ms_while_playing": None,
                "echo_turn_suppression": False,
            }
        }
    )
    assert not config.barge_in.active


def test_empty_file_takes_every_default(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, ""), {})
    assert config.limits.outbound_queue_events == 256
    assert config.text.segmenter.max_chars == 80


def test_env_override_is_parsed_as_yaml(tmp_path: Path) -> None:
    environ = {
        "MLXRT__GATEWAY__PORT": "9001",
        "MLXRT__GATEWAY__API_KEY": "secret",
        "MLXRT__PROVIDERS__ASR__TRANSCRIPTS": "[一, two]",
        "MLXRT__TURN__SMART_TURN__ENABLED": "true",
        "UNRELATED": "1",
    }
    config = load_config(write(tmp_path, "gateway: {port: 8000}\n"), environ)
    assert config.gateway.port == 9001
    assert config.gateway.api_key == "secret"
    assert isinstance(config.providers.asr, MockASRConfig)
    assert config.providers.asr.transcripts == ["一", "two"]
    assert config.turn.smart_turn.enabled is True


@pytest.mark.parametrize(
    ("text", "environ", "path"),
    [
        ("gateway: {bogus: 1}\n", {}, "gateway.bogus"),
        ("gateway: {port: 70000}\n", {}, "gateway.port"),
        ("limits: {outbound_queue_events: many}\n", {}, "limits.outbound_queue_events"),
        ("providers: {llm: {type: nope}}\n", {}, "providers.llm"),
        ("providers: {llm: {type: mlx_lm_chat, top_p: 2}}\n", {}, "providers.llm.top_p"),
        ("providers: {asr: {type: mlx_audio_transcription, url: x}}\n", {}, "providers.asr.url"),
        (
            "providers: {llm: {type: mlx_lm_chat, system_note: x}}\n",
            {},
            "providers.llm.system_note",
        ),
        ("backends: {llm: {project: p, command: [], port: 1}}\n", {}, "backends.llm.command"),
        ("turn: {smart_turn: {threshold: 2}}\n", {}, "turn.smart_turn.threshold"),
        ("", {"MLXRT__TEXT__SEGMENTER__MIN_CHARS": "0"}, "text.segmenter.min_chars"),
        ("", {"MLXRT__NOPE__KEY": "1"}, "nope"),
    ],
)
def test_invalid_values_name_the_dotted_path(
    tmp_path: Path, text: str, environ: dict[str, str], path: str
) -> None:
    with pytest.raises(ConfigError, match=rf"\b{path.replace('.', r'\.')}\b"):
        load_config(write(tmp_path, text), environ)


def test_errors_name_the_env_variable_that_set_the_value(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"gateway\.port: .*\(set by MLXRT__GATEWAY__PORT\)"):
        load_config(write(tmp_path, ""), {"MLXRT__GATEWAY__PORT": "eighty"})


def test_override_below_a_scalar_is_rejected(tmp_path: Path) -> None:
    environ = {"MLXRT__GATEWAY__PORT__X": "1"}
    with pytest.raises(
        ConfigError, match=r"MLXRT__GATEWAY__PORT__X: gateway\.port is not a section"
    ):
        load_config(write(tmp_path, "gateway: {port: 8000}\n"), environ)
    with pytest.raises(ConfigError, match=r"gateway\.port: .*\(set by MLXRT__GATEWAY__PORT__X\)"):
        load_config(write(tmp_path, ""), environ)


def test_unreadable_and_malformed_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml", {})
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "gateway: [\n"), {})
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_config(write(tmp_path, "- 1\n"), {})


@pytest.mark.parametrize(
    ("session_defaults", "path"),
    [
        (
            {"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}},
            "session_defaults.audio.input.turn_detection.type",
        ),
        ({"audio": {"output": {"speed": 1.2}}}, "session_defaults.audio.output.speed"),
        ({"modalities": ["text"]}, "session_defaults.modalities"),
        ({"type": "transcription"}, "session_defaults.type"),
    ],
)
def test_session_defaults_are_validated_like_a_session_update(
    session_defaults: dict[str, object], path: str
) -> None:
    config = validate_config({"session_defaults": session_defaults})
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        Gateway(config, build_providers(config.providers))


def test_provider_sections_without_a_type_are_mocks(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "providers: {llm: {greeting: hi}}\n"), {})
    assert isinstance(config.providers.llm, MockLLMConfig)
    assert config.providers.llm.greeting == "hi"


def test_the_llm_model_name_is_omitted_when_the_backend_serves_a_snapshot(tmp_path: Path) -> None:
    text = (
        "providers: {llm: {type: mlx_lm_chat, model: some/repo}}\n"
        "backends: {llm: {project: backends/llm, command: [mlx_lm.server], port: 8101,"
        " model: {repo: some/repo, revision: main}}}\n"
    )
    with pytest.raises(ConfigError, match=r"providers\.llm\.model must be omitted"):
        load_config(write(tmp_path, text), {})


@pytest.mark.parametrize(
    ("raw", "path"),
    [
        ({"output": {"max_lead_ms": 50}}, "output"),  # below one delta
        ({"output": {"max_delta_ms": 2000}}, "output.max_delta_ms"),  # a 2 s gap at the cap
        ({"lines": {"verbatim_prefixes": [""]}}, "lines.verbatim_prefixes"),
        (
            {"providers": {"tts": {"type": "mlx_audio_speech", "max_in_flight": 0}}},
            "providers.tts.max_in_flight",
        ),
        (
            {
                "providers": {
                    "tts": {"type": "mlx_audio_speech", "primary": {"model": "m", "family": "x"}}
                }
            },
            "providers.tts.primary.family",
        ),
        (
            {"providers": {"tts": {"type": "mlx_audio_speech", "warmup_texts": {"zh": "你好"}}}},
            "warmup_texts",
        ),
        ({"text": {"normalizer": {"currency": True}}}, "text.normalizer.currency"),
    ],
)
def test_m3_settings_are_validated(raw: dict[str, object], path: str) -> None:
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        validate_config(raw)


def test_session_defaults_must_name_a_configured_voice() -> None:
    config = validate_config({"session_defaults": {"audio": {"output": {"voice": "marin"}}}})
    with pytest.raises(ConfigError, match=r"session_defaults\.audio\.output\.voice: 'marin'"):
        Gateway(config, build_providers(config.providers))


def test_m6_defaults_and_the_repository_values() -> None:
    defaults = validate_config({})
    assert defaults.output.reserve == AudioReserveConfig(
        enabled=True, hold_ms=1500, keepalive_ms=1000, min_lead_ms=400
    )
    observability = defaults.observability
    assert (observability.quantile_window_turns, observability.debug_endpoints) == (200, False)
    speech = validate_config({"providers": {"tts": {"type": "mlx_audio_speech"}}}).providers.tts
    assert isinstance(speech, MlxAudioSpeechConfig)
    assert (speech.scheduling, speech.first_segment_grace_ms) == ("edf", 0)
    repository = load_config(REPO_CONFIG, {})
    assert repository.output.reserve == defaults.output.reserve
    assert repository.observability == defaults.observability
    assert isinstance(repository.providers.tts, MlxAudioSpeechConfig)
    assert repository.providers.tts.scheduling == "edf"
    assert load_config(MOCK_CONFIG, {}).observability.debug_endpoints  # for bench/


@pytest.mark.parametrize(
    ("raw", "path"),
    [
        ({"output": {"reserve": {"keepalive_ms": 2000}}}, "output.reserve.keepalive_ms"),
        ({"output": {"reserve": {"hold_ms": -1}}}, "output.reserve.hold_ms"),
        ({"observability": {"quantile_window_turns": 0}}, "observability.quantile_window_turns"),
        ({"observability": {"debug_sessions": 0}}, "observability.debug_sessions"),
        (
            {"providers": {"tts": {"type": "mlx_audio_speech", "scheduling": "lifo"}}},
            "providers.tts.scheduling",
        ),
    ],
)
def test_m6_settings_are_validated(raw: dict[str, object], path: str) -> None:
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        validate_config(raw)

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_realtime.config import (
    ConfigError,
    MlxAudioAsrConfig,
    MlxLmChatConfig,
    MockASRConfig,
    MockLLMConfig,
    load_config,
    validate_config,
)
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.mock import EchoLLM
from mlx_realtime.providers.registry import build_providers
from mlx_realtime.server.gateway import Gateway
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


def test_mock_config_is_the_m1_setup() -> None:
    config = load_config(MOCK_CONFIG, {})
    gateway = Gateway(config)
    assert isinstance(gateway.providers.llm, EchoLLM)
    assert config.backends.processes() == {}
    assert config.turn.vad.type == "mock"


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

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_realtime.config import ConfigError, load_config, validate_config
from mlx_realtime.providers.registry import build_providers
from mlx_realtime.server.gateway import Gateway

REPO_CONFIG = Path(__file__).parents[2] / "config.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repository_config_loads_and_builds_a_gateway() -> None:
    config = load_config(REPO_CONFIG, {})
    gateway = Gateway(config, build_providers(config.providers))
    assert config.gateway.port == 8000
    assert config.providers.llm.type == "mock"
    assert gateway.defaults.turn_detection is not None
    assert gateway.defaults.turn_detection.silence_duration_ms == 200
    assert gateway.defaults.transcription is not None


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
    assert config.providers.asr.transcripts == ["一", "two"]
    assert config.turn.smart_turn.enabled is True


@pytest.mark.parametrize(
    ("text", "environ", "path"),
    [
        ("gateway: {bogus: 1}\n", {}, "gateway.bogus"),
        ("gateway: {port: 70000}\n", {}, "gateway.port"),
        ("limits: {outbound_queue_events: many}\n", {}, "limits.outbound_queue_events"),
        ("providers: {llm: {type: mlx_lm_chat}}\n", {}, "providers.llm.type"),
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

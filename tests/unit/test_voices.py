"""The voice map (D19): configured names, the default for every other name, and the checks the
configuration makes."""

from __future__ import annotations

import pytest

from mlx_realtime.config import ConfigError, VoicesConfig, validate_config
from mlx_realtime.providers.voices import SEGMENT_LANG, VoiceBook


def test_the_builtin_voices_are_qwen3_speakers_without_the_dialect_ones() -> None:
    book = VoiceBook(VoicesConfig())
    assert book.default == "vivian"
    assert set(book.names()) == {"vivian", "serena", "uncle_fu", "aiden"}
    assert not {"dylan", "eric"} & set(book.names())  # dialects with chinese/auto
    assert "ryan" not in book  # S3: the runaway defect follows it
    vivian = book.voice("vivian")
    assert (vivian.speaker_for("zh"), vivian.speaker_for("en")) == ("vivian", "aiden")  # S3
    for voice in book.voices():
        assert voice.speaker == voice.name
        assert voice.lang == SEGMENT_LANG
        assert voice.fallback["zh"].startswith("z")  # a Mandarin Kokoro voice
        assert voice.fallback["en"].startswith("a")  # an American English one


def test_unknown_names_resolve_to_the_default() -> None:
    book = VoiceBook(
        VoicesConfig.model_validate(
            {
                "default": "agent",
                "map": {"agent": {"speaker": "serena"}, "boss": {"speaker": "ryan"}},
            }
        )
    )
    assert book.effective("boss") == "boss"
    assert book.effective("flow-voice") == "agent"
    assert book.effective("") == "agent"
    assert book.voice("marin").speaker == "serena"
    assert "boss" in book
    assert "marin" not in book


@pytest.mark.parametrize(
    ("raw", "path", "message"),
    [
        ({"voices": {"default": "nope"}}, "voices", "not in voices.map"),
        ({"voices": {"default": "x", "map": {}}}, "voices", "at least one voice"),
        ({"voices": {"map": {"vivian": {"speaker": ""}}}}, "voices.map.vivian.speaker", ""),
        (
            {"voices": {"map": {"vivian": {"speaker": "v", "fallback": {"fr": "x"}}}}},
            "fallback",
            "",
        ),
    ],
)
def test_invalid_voice_maps_name_the_problem(
    raw: dict[str, object], path: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")) as failed:
        validate_config(raw)
    assert message in str(failed.value)


def test_kokoro_fallback_voices_are_required_while_the_fallback_is_on() -> None:
    voices = {"default": "a", "map": {"a": {"speaker": "vivian", "fallback": {"zh": "zf_xiaobei"}}}}
    with pytest.raises(ConfigError, match=r"voices\.map\.a\.fallback needs a Kokoro voice for en"):
        validate_config({"voices": voices, "providers": {"tts": {"type": "mlx_audio_speech"}}})
    off = {
        "type": "mlx_audio_speech",
        "fallback": {"model": "k", "family": "kokoro", "enabled": False},
    }
    validate_config({"voices": voices, "providers": {"tts": off}})
    validate_config({"voices": voices})  # the mock TTS needs none

"""The benchmark end to end against the mock stack (``config.mock.yaml``: real Silero VAD, mock
providers), run in-process with ``--serve``: two utterances, one of them 8 kHz telephone audio,
in two parallel sessions, replayed at real-time speed."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from bench.bench import Options, benchmark, main, write
from bench.dataset import load
from tests.speech import EN, SILERO, SPEECH, ZH, needs_models

REPO = Path(__file__).parents[2]
MOCK_TRANSCRIPT = "你好，我想查询一下账户余额。"  # config.mock.yaml's scripted ASR


def telephone_copy(source: Path, target: Path) -> None:
    """The 16 kHz fixture decimated to 8 kHz (every other sample, after a crude low-pass)."""
    with wave.open(str(source)) as file:
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2").astype(float)
    smoothed = np.convolve(samples, np.ones(4) / 4, mode="same")[::2]
    with wave.open(str(target), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(8000)
        file.writeframes(smoothed.astype("<i2").tobytes())


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "zh.wav").write_bytes((SPEECH / f"{ZH}.wav").read_bytes())
    (tmp_path / "zh.txt").write_text("但鸟类还是有很多地方跟恐龙相似。", encoding="utf-8")
    telephone_copy(SPEECH / f"{EN}.wav", tmp_path / "en_tel.wav")
    (tmp_path / "en_tel.txt").write_text(
        "Many people don't think about them as dinosaurs because they have feathers and can fly.",
        encoding="utf-8",
    )
    return tmp_path


@needs_models
async def test_the_bench_measures_the_mock_stack(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLXRT__TURN__VAD__MODEL", str(SILERO))
    utterances = load(dataset)
    assert {u.source_rate for u in utterances} == {8000, 16000}
    options = Options(concurrency=2, min_silence_ms=600)
    report = await benchmark(utterances, options, serve_config=REPO / "config.mock.yaml")
    assert report["counts"] == {
        "utterances": 2,
        "answered": 2,
        "no_audio": 0,
        "no_turn": 0,
        "timeout": 0,
        "split_turns": 0,
    }
    turns = report["turns"]
    assert isinstance(turns, list)
    for turn in turns:
        assert turn["transcript"] == MOCK_TRANSCRIPT
        assert turn["response_status"] == "completed"
        assert set(turn["server_stages_ms"]) >= {"speech_stopped", "asr_done", "first_audio_sent"}
        assert 0 < turn["server_e2e_ms"] < 3000
        assert turn["client_e2e_ms"] > 0
    assert len({turn["session_id"] for turn in turns}) == 2  # two sessions in parallel
    latency = report["latency"]
    assert isinstance(latency, dict)
    assert latency["e2e_server"]["n"] == 2
    asr = report["asr"]
    assert isinstance(asr, dict)
    assert asr["zh"]["all"]["rate"] > 0.5  # the mock ASR says something else entirely
    json_path, md_path = write(report, tmp_path / "out")
    assert json_path.is_file()
    assert "## Latency (ms)" in md_path.read_text(encoding="utf-8")


def test_the_cli_refuses_an_empty_selection(dataset: Path) -> None:
    with pytest.raises(SystemExit, match="no utterances"):
        main(["--input", str(dataset), "--sets", "mix"])

"""The benchmark's scoring, input and report (``bench/``): M0's text_norm rule and error rates,
G.711 against ``audioop``, the input layouts, and the report's aggregation."""

from __future__ import annotations

import json
import warnings
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from bench.dataset import (
    Law,
    g711_decode,
    g711_encode,
    guess_set,
    load,
    speech_end_ms,
)
from bench.report import TurnResult, markdown, stat, summarize
from bench.scoring import align, error, norm_text, pooled, tokens

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # 3.12 still has it (removed in 3.13): the G.711 reference


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("1990 年，由于…", "1990年由于"),  # the README's examples
        ("one-hundred percent", "one hundred percent"),
        ("（Recep Tayyip Erdoğan）", "recep tayyip erdoğan"),
        ("What’s up? It's 'fine'.", "what's up it's fine"),
        ("我用 iPhone 打电话。", "我用iphone打电话"),
        ("ＡＢＣ　１２３", "abc 123"),
    ],
)
def test_text_norm_follows_the_manifest_rule(raw: str, normalized: str) -> None:
    assert norm_text(raw) == normalized


def test_the_manifest_text_norm_is_reproduced_on_every_row() -> None:
    manifest = Path(__file__).parents[2] / "spikes" / "datasets" / "manifest.jsonl"
    if not manifest.is_file():
        pytest.skip("the M0 manifest (spikes/datasets/manifest.jsonl) is not here")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 126
    assert all(norm_text(row["text"]) == row["text_norm"] for row in rows)


def test_tokens_per_metric() -> None:
    text = norm_text("我用 iPhone 打了 3 次电话")
    assert tokens(text, "cer") == list("我用iphone打了3次电话")
    assert tokens(text, "mer") == ["我", "用", "iphone", "打", "了", "3", "次", "电", "话"]
    assert tokens("many people don't", "wer") == ["many", "people", "don't"]


def test_alignment_counts_substitutions_deletions_and_insertions() -> None:
    edits = align(list("abcdef"), list("azcdxfg"))
    assert (edits.ref, edits.sub, edits.dels, edits.ins, edits.edits) == (6, 2, 0, 1, 3)
    assert align([], list("ab")).rate is None
    assert align(list("abc"), []).dels == 3
    zh = error("但鸟类还是有很多地方跟恐龙相似", "但鸟类还是有很多地方跟恐龙想似。", "cer")
    assert (zh.sub, zh.rate) == (1, pytest.approx(1 / 15))
    en = error("many people don't think", "Many people do not think.", "wer")
    assert (en.sub, en.ins, en.edits) == (1, 1, 2)
    total = pooled([zh, en])
    assert (total.ref, total.edits) == (19, 3)


def test_g711_matches_audioop_exhaustively() -> None:
    codes = bytes(range(256))
    every = np.arange(-32768, 32768, dtype=np.int16)
    cases: list[tuple[Law, Callable[[bytes, int], bytes], Callable[[bytes, int], bytes]]] = [
        ("ulaw", audioop.ulaw2lin, audioop.lin2ulaw),
        ("alaw", audioop.alaw2lin, audioop.lin2alaw),
    ]
    for law, decode, encode in cases:
        assert g711_decode(codes, law).tobytes() == decode(codes, 2)
        assert g711_encode(every, law) == encode(every.tobytes(), 2)


def test_the_speech_end_excludes_trailing_silence() -> None:
    rate = 24_000
    speech = (8000 * np.sin(np.arange(rate) / rate * 2 * np.pi * 220)).astype("<i2")
    pcm = speech.tobytes() + bytes(2 * rate // 2)  # 1 s of tone, 0.5 s of silence
    assert speech_end_ms(pcm) == 1000
    assert speech_end_ms(bytes(2 * rate)) == 0


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.astype("<i2").tobytes())


def tone(rate: int, seconds: float) -> np.ndarray:
    times = np.arange(int(rate * seconds)) / rate
    return 6000 * np.sin(2 * np.pi * 300 * times)


def test_a_directory_of_wavs_and_transcripts(tmp_path: Path) -> None:
    write_wav(tmp_path / "a.wav", tone(8000, 1.0), 8000)  # telephone audio
    (tmp_path / "a.txt").write_text("你好，我想查询 3 个订单。\n", encoding="utf-8")
    write_wav(tmp_path / "b.wav", tone(16000, 0.5), 16000)
    (tmp_path / "b.txt").write_text("Hello there.", encoding="utf-8")
    write_wav(tmp_path / "c.wav", tone(16000, 0.5), 16000)  # no transcript: skipped
    first, second = load(tmp_path)
    assert (first.id, first.set, first.has_number, first.source_rate) == ("a", "zh", True, 8000)
    assert first.text_norm == "你好我想查询3个订单"
    assert first.duration_ms == 1000  # resampled to 24 kHz
    assert abs(len(first.pcm) - 48_000) <= 4
    assert (second.set, second.has_number) == ("en", False)
    assert [u.id for u in load(tmp_path, numbers="exclude")] == ["b"]
    assert [u.id for u in load(tmp_path, ids="^a$")] == ["a"]
    assert len(load(tmp_path, limit=1)) == 1


def test_an_m0_manifest_with_every_copy(tmp_path: Path) -> None:
    (tmp_path / "asr").mkdir()
    samples = tone(8000, 0.4).astype(np.int16)
    (tmp_path / "asr" / "x.ulaw").write_bytes(g711_encode(samples, "ulaw"))
    write_wav(tmp_path / "asr" / "x16.wav", tone(16000, 0.4), 16000)
    row = {
        "id": "mix_001",
        "set": "mix",
        "path_16k": "asr/x16.wav",
        "path_tel_16k": "asr/x16.wav",
        "path_8k_ulaw": "asr/x.ulaw",
        "codec_8k": "ulaw",
        "text": "我用 iPhone",
        "text_norm": "我用iphone",
        "has_number": False,
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tel,) = load(tmp_path / "manifest.jsonl")
    assert tel.g711 is not None
    assert tel.g711[0] == "ulaw"
    assert (tel.source_rate, tel.duration_ms) == (8000, 400)
    (wide,) = load(tmp_path / "manifest.jsonl", variant="wide")
    assert (wide.source_rate, wide.g711) == (16000, None)
    assert load(tmp_path / "manifest.jsonl", sets=["zh"]) == []
    assert guess_set("hello 你好") == "mix"


def turn(name: str, outcome: str = "answered", **fields: object) -> TurnResult:
    defaults: dict[str, object] = {"set": "zh", "has_number": False, "text_norm": "你好"}
    return TurnResult(utterance_id=name, outcome=outcome, **{**defaults, **fields})  # type: ignore[arg-type]


def test_the_report_aggregates_latency_and_scores_by_set_and_numbers() -> None:
    turns = [
        turn("1", transcript="你好", client_e2e_ms=900.0, server_e2e_ms=850.0,
             server_stages_ms={"asr_done": 100.0, "llm_first_token": 300.0}, items=["i1"]),
        turn("2", transcript="你", client_e2e_ms=1100.0, server_e2e_ms=1000.0,
             server_stages_ms={"asr_done": 200.0}, items=["i2", "i3"]),
        turn("3", set="en", text_norm="call 911", has_number=True, transcript="call nine one one",
             client_e2e_ms=1300.0, items=["i4"]),
        turn("4", outcome="no_turn"),
        turn("5", outcome="no_audio", transcript="你好", items=["i5"]),
    ]  # fmt: skip
    report = summarize(turns, meta={"target": "mock"})
    latency = report["latency"]
    assert isinstance(latency, dict)
    assert latency["asr_done"] == {"n": 2, "p50_ms": 150.0, "p95_ms": 195.0}
    assert latency["llm_first_token"]["n"] == 1
    assert latency["e2e_client"] == {"n": 3, "p50_ms": 1100.0, "p95_ms": 1280.0}
    assert latency["e2e_server"]["n"] == 2
    assert latency["first_audio_sent"] == {"n": 0, "p50_ms": None, "p95_ms": None}
    counts = report["counts"]
    assert counts == {
        "utterances": 5,
        "answered": 3,
        "no_audio": 1,
        "no_turn": 1,
        "timeout": 0,
        "split_turns": 1,
    }
    asr = report["asr"]
    assert isinstance(asr, dict)
    zh = asr["zh"]["without_numbers"]  # rows 1, 2 and 5: 1 deletion in 6 characters
    assert (zh["kind"], zh["utterances"], zh["ref_tokens"], zh["edits"]) == ("cer", 3, 6, 1)
    assert asr["zh"]["with_numbers"]["rate"] is None
    en = asr["en"]["with_numbers"]
    assert (en["kind"], en["edits"], en["ref_tokens"]) == ("wer", 3, 2)
    assert asr["all"]["all"]["kind"] == "mer"
    assert asr["all"]["all"]["utterances"] == 4  # no_turn is not scored
    text = markdown(report)
    assert "| asr_done | turn_confirmed / committed | 2 | 150 | 195 |" in text
    assert "| zh | CER | 16.67% (1/6, n=3) | – | 16.67% (1/6, n=3) |" in text
    assert "- target: mock" in text


def test_stat_of_nothing() -> None:
    assert stat([]) == stat(())
    assert stat([5.0]).p95_ms == 5.0

"""Edge-silence trimming of synthesized segments (M0 S3; ``audio/trim.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlx_realtime.audio.trim import SilenceTrimmer

RATE = 24_000
MS = RATE * 2 // 1000  # bytes per millisecond


def tone(ms: int, amplitude: float = 0.2) -> bytes:
    n = np.arange(RATE * ms // 1000)
    return (amplitude * 32767 * np.sin(2 * math.pi * 220 * n / RATE)).astype("<i2").tobytes()


def quiet(ms: int, amplitude: int = 0) -> bytes:
    rng = np.random.default_rng(1)
    noise = rng.integers(-amplitude, amplitude + 1, RATE * ms // 1000) if amplitude else None
    if noise is None:
        return bytes(RATE * ms // 1000 * 2)
    return noise.astype("<i2").tobytes()


def trimmer(**overrides: float) -> SilenceTrimmer:
    options: dict[str, float] = {"threshold_dbfs": -50.0, "max_lead_ms": 60, "max_tail_ms": 100}
    options.update(overrides)
    return SilenceTrimmer(RATE, **options)  # pyright: ignore[reportArgumentType]


def run(trim: SilenceTrimmer, audio: bytes, chunk_ms: int) -> list[bytes]:
    step = chunk_ms * MS
    outputs = [trim.push(audio[start : start + step]) for start in range(0, len(audio), step)]
    return [*outputs, trim.flush()]


@pytest.mark.parametrize("chunk_ms", [7, 20, 80, 160, 1000])
def test_edges_are_trimmed_and_inner_pauses_kept(chunk_ms: int) -> None:
    speech = tone(300) + quiet(240) + tone(200)
    audio = quiet(460, amplitude=20) + speech + quiet(500, amplitude=20)
    out = b"".join(run(trimmer(), audio, chunk_ms))
    # 60 ms of the lead (three 20 ms frames) and 100 ms of the tail are kept
    assert len(out) == 60 * MS + len(speech) + 100 * MS
    assert out[60 * MS : 60 * MS + len(speech)] == speech


def test_speech_off_the_frame_grid_keeps_at_most_a_frame_more() -> None:
    speech = tone(300)
    audio = quiet(450, amplitude=20) + speech + quiet(510, amplitude=20)
    out = b"".join(run(trimmer(), audio, 80))
    lead = out.index(speech)
    assert 60 * MS <= lead <= 80 * MS
    assert 100 * MS <= len(out) - lead - len(speech) <= 120 * MS


def test_the_first_voiced_audio_goes_out_at_once() -> None:
    """The lead is held only while it is silent: the chunk that brings the first voiced frame
    releases it with the kept lead."""
    trim = trimmer()
    chunks = [quiet(160), quiet(160), quiet(130) + tone(30), tone(160)]
    outputs = [trim.push(chunk) for chunk in chunks]
    assert outputs[:2] == [b"", b""]
    # 450 ms of silence: the frame at 440-460 ms holds 10 ms of tone, the next is all tone;
    # both go out with the 60 ms before them, while the stream is still at 480 ms
    assert len(outputs[2]) == (60 + 40) * MS
    assert outputs[2].endswith(chunks[2][120 * MS :])


def test_silence_only_yields_nothing() -> None:
    trim = trimmer()
    assert b"".join(run(trim, quiet(900, amplitude=10), 80)) == b""
    assert trim.trimmed_bytes == 900 * MS


def test_zero_lead_and_tail() -> None:
    speech = tone(200)
    out = b"".join(run(trimmer(max_lead_ms=0, max_tail_ms=0), quiet(100) + speech + quiet(100), 40))
    assert out == speech


def test_the_threshold_decides() -> None:
    soft = tone(200, amplitude=0.002)  # about -57 dBFS RMS
    assert b"".join(run(trimmer(), soft, 40)) == b""
    assert len(b"".join(run(trimmer(threshold_dbfs=-60), soft, 40))) == len(soft)


def test_a_partial_last_frame_counts() -> None:
    speech = tone(205)  # 10 frames and 5 ms
    out = b"".join(run(trimmer(), speech, 50))
    assert out == speech

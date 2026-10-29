"""The opt-in echo capture (session/echo_capture.py, aec.capture_dir): off by default, sample-
aligned tracks and events on the input timeline, the duration cap, and an I/O failure that never
reaches the session."""

from __future__ import annotations

import json
import logging
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from mlx_realtime.audio.aec import FRAME_SAMPLES, Pcm
from mlx_realtime.config import AecConfig, BargeInConfig
from mlx_realtime.session.echo import EchoParams, EchoReference, EchoStage
from mlx_realtime.session.echo_capture import TRACKS, EchoCapture

RATE = 16_000


class FakeClock:
    def now(self) -> float:
        return 100.0

    async def sleep(self, seconds: float) -> None:
        return None


class HalvingEngine:
    def process(self, capture: Pcm, reference: Pcm) -> Pcm:
        return (capture // 2).astype(np.int16)


def tone(count: int, level: int = 1000) -> npt.NDArray[np.int16]:
    return (np.arange(count) % 50 * level // 50).astype(np.int16)


def read(directory: Path, name: str) -> npt.NDArray[np.int16]:
    with wave.open(str(directory / f"{name}.wav"), "rb") as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, RATE)
        return np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")


def capture_of(tmp_path: Path, max_s: int = 900) -> EchoCapture:
    return EchoCapture(tmp_path, "sess_1", max_s=max_s, meta={"input_format": "audio/pcm"})


def test_it_is_off_by_default(tmp_path: Path) -> None:
    assert AecConfig().capture_dir is None
    echo = EchoReference(EchoParams.of(AecConfig(), BargeInConfig()), FakeClock())
    assert echo.capture is None
    echo.note("flush", generation=1)  # no capture: a no-op
    assert not list(tmp_path.iterdir())


def test_the_tracks_are_aligned_on_the_input_timeline(tmp_path: Path) -> None:
    capture = capture_of(tmp_path)
    first, second = tone(1600), tone(800, 500)
    capture.audio(0, first, first // 4, first // 2)
    capture.event(100.04, "gcc", lag_ms=100.0)
    capture.audio(2400, second, second, second, second, second)  # 800 samples skipped
    capture.close({"erle_db": 12.5})
    capture.wait()
    (directory,) = tmp_path.iterdir()
    assert directory.name.endswith("_sess_1")
    tracks = {name: read(directory, name) for name in TRACKS}
    assert {len(track) for track in tracks.values()} == {3200}
    assert np.array_equal(tracks["input"][:1600], first)
    assert np.array_equal(tracks["ref_aec3"][:1600], first // 4)
    assert np.array_equal(tracks["asr"][:1600], first)  # no ASR path: the input as received
    assert not tracks["ref_speex"][:1600].any()
    assert not tracks["input"][1600:2400].any()  # the gap is silence
    assert np.array_equal(tracks["aec3"][2400:], second)
    (event,) = (
        json.loads(line) for line in (directory / "events.jsonl").read_text().split("\n") if line
    )
    assert event == {"t": 100.0, "kind": "gcc", "lag_ms": 100.0}
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["session_id"] == "sess_1"
    assert meta["samples"] == 3200
    assert meta["echo_summary"] == {"erle_db": 12.5}
    assert meta["complete"]
    assert meta["started_utc"].endswith("+00:00")


def test_the_duration_cap_ends_the_capture(tmp_path: Path) -> None:
    capture = capture_of(tmp_path, max_s=1)
    for start in range(0, 3 * RATE, RATE // 2):
        capture.audio(start, tone(RATE // 2), tone(RATE // 2), tone(RATE // 2))
    capture.event(5.0, "late")
    capture.close({})
    capture.wait()
    (directory,) = tmp_path.iterdir()
    assert len(read(directory, "input")) == RATE
    meta = json.loads((directory / "meta.json").read_text())
    assert meta["capped"]
    assert "late" not in (directory / "events.jsonl").read_text()


def test_an_io_error_warns_once_and_stops_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="mlx_realtime.session.echo_capture")
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    capture = EchoCapture(blocker / "captures", "sess_1", max_s=10, meta={})
    capture.audio(0, tone(1600), tone(1600), tone(1600))
    capture._thread.join(5.0)  # pyright: ignore[reportPrivateUsage]
    assert not capture.active
    capture.audio(1600, tone(1600), tone(1600), tone(1600))  # later calls are no-ops
    capture.event(1.0, "gcc")
    capture.close({})
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


async def test_the_stage_writes_what_it_processed(tmp_path: Path) -> None:
    clock = FakeClock()
    echo = EchoReference(
        EchoParams.of(AecConfig(delay_guard=False, asr_input="raw"), BargeInConfig()), clock
    )
    echo.capture = capture_of(tmp_path)
    with ThreadPoolExecutor(1) as executor:
        stage = EchoStage(HalvingEngine(), echo, executor)
        pcm = tone(5 * FRAME_SAMPLES + 40)
        await stage.feed(0, pcm.tobytes())
        await stage.release(force=True)  # the partial frame goes through as it is
    echo.close_capture({})
    echo.capture.wait()
    (directory,) = list(tmp_path.glob("*"))  # noqa: ASYNC240
    assert np.array_equal(read(directory, "input"), pcm)
    aec3 = read(directory, "aec3")
    assert len(aec3) == len(pcm)
    assert np.array_equal(aec3[: 5 * FRAME_SAMPLES], pcm[: 5 * FRAME_SAMPLES] // 2)
    assert np.array_equal(aec3[5 * FRAME_SAMPLES :], pcm[5 * FRAME_SAMPLES :])


@pytest.mark.parametrize(
    ("aec", "message"),
    [
        ({"capture_max_s": 0}, "capture_max_s"),
        ({"capture_dir": ""}, "capture_dir"),
        ({"capture_dir": "/tmp/x", "enabled": False}, "capture_dir needs the canceller"),
    ],
)
def test_the_capture_options_are_validated(aec: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AecConfig.model_validate(aec)


def test_the_capture_options_have_defaults() -> None:
    assert AecConfig.model_validate({"capture_dir": "/tmp/x"}).capture_max_s == 900

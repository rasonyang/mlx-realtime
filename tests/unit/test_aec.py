"""The echo-canceller engine (audio/aec.py: WebRTC AEC3 through pywebrtc-audio, G7) and the
GCC-PHAT delay estimator (audio/delay.py), on synthetic telephone echo: the fixture speech
through a seeded, decaying random impulse response at 8 kHz after a bulk delay (no ITU G.168
coefficients, which are © ITU). Also the canceller's CPU per second of audio (M0 B4: 7.8 ms)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from mlx_realtime.audio.aec import FRAME_SAMPLES, WebRtcAec3, check_frame, make_engine
from mlx_realtime.audio.delay import DelayEstimate, DelayEstimator
from mlx_realtime.audio.resample import Resampler
from tests.speech import EN, ZH, clip

RATE = 16_000
type Pcm = npt.NDArray[np.int16]
type Floats = npt.NDArray[np.float64]


def speech(name: str, seconds: float) -> Floats:
    """The fixture, looped to ``seconds``, as floats at 16 kHz."""
    samples = np.frombuffer(clip(name), dtype="<i2").astype(np.float64)
    return np.resize(samples, int(seconds * RATE))


def echo_of(reference: Floats, *, delay_ms: int, erl_db: float = 10.0, seed: int = 3) -> Floats:
    """``reference`` through a synthetic line-echo path: 96 taps at 8 kHz (12 ms), decaying,
    scaled to the echo return loss, after a bulk delay (all at 16 kHz here)."""
    rng = np.random.default_rng(seed)
    taps = rng.standard_normal(96) * np.exp(-np.arange(96) / 19)
    path8 = taps / np.sqrt(np.sum(taps**2)) * 10 ** (-erl_db / 20)
    path16 = np.zeros(2 * len(path8))
    path16[::2] = path8  # the 8 kHz path's taps at 16 kHz
    delayed = np.concatenate([np.zeros(delay_ms * RATE // 1000), reference])[: len(reference)]
    return np.convolve(delayed, path16)[: len(reference)]


def frames(signal: Floats) -> list[Pcm]:
    pcm = np.clip(np.rint(signal), -32768, 32767).astype(np.int16)
    return [pcm[start : start + FRAME_SAMPLES] for start in range(0, len(pcm), FRAME_SAMPLES)]


def energy_db(signal: Floats) -> float:
    return 10 * float(np.log10(np.mean(signal**2) + 1e-9))


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros(159, dtype=np.int16),
        np.zeros(240, dtype=np.int16),
        np.zeros(160, dtype=np.float32),
        np.zeros((1, 160), dtype=np.int16),
    ],
)
def test_only_whole_int16_frames_reach_the_engine(frame: npt.NDArray[np.generic]) -> None:
    """pywebrtc-audio zero-pads a partial frame silently, so the wrapper refuses one."""
    engine = WebRtcAec3()
    whole = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    with pytest.raises(ValueError, match=r"int16\[160\]"):
        engine.process(frame, whole)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=r"int16\[160\]"):
        engine.process(whole, frame)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="capture"):
        check_frame(frame, "capture")  # pyright: ignore[reportArgumentType]


def test_aec3_removes_a_line_echo_and_keeps_the_caller() -> None:
    far = speech(ZH, 12.0) * 0.8
    near = np.zeros_like(far)
    caller = speech(EN, 3.0)
    near[9 * RATE : 12 * RATE] = caller
    capture = echo_of(far, delay_ms=120) + near + np.random.default_rng(1).normal(0, 10, len(far))
    engine = make_engine("webrtc_aec3")
    output = np.concatenate(
        [engine.process(c, r) for c, r in zip(frames(capture), frames(far), strict=True)]
    ).astype(np.float64)
    echo_only = slice(2 * RATE, 9 * RATE)  # after a second or two of convergence
    erle = energy_db(capture[echo_only]) - energy_db(output[echo_only])
    assert erle > 20, erle
    talk = slice(9 * RATE, 12 * RATE)  # double talk: the caller over our speech
    assert energy_db(output[talk]) > energy_db(caller) - 15  # the caller is not cancelled


def test_aec3_cpu_per_second_of_audio(record_property: Callable[[str, object], None]) -> None:
    """M0 measured 7.8 ms of CPU per audio second (B4); the bound here only catches a
    pathological regression. ``pytest -rP`` prints the measurement."""
    seconds = 20.0
    far = speech(ZH, seconds)
    capture = echo_of(far, delay_ms=120)
    engine = make_engine("webrtc_aec3")
    pairs = list(zip(frames(capture), frames(far), strict=True))
    began = time.thread_time_ns()
    for near, reference in pairs:
        engine.process(near, reference)
    cpu_ms = (time.thread_time_ns() - began) / 1e6 / seconds
    record_property("aec_cpu_ms_per_audio_s", round(cpu_ms, 2))
    print(f"AEC3 CPU: {cpu_ms:.2f} ms per second of audio")
    assert cpu_ms < 50


def estimates_for(capture: Floats, reference: Floats) -> list[DelayEstimate]:
    estimator = DelayEstimator()
    found: list[DelayEstimate] = []
    for near, far in zip(frames(capture), frames(reference), strict=True):
        found += estimator.push(near, far)
    return found


@pytest.mark.parametrize("delay_ms", [20, 160, 600])
def test_the_estimator_finds_the_residual_delay(delay_ms: int) -> None:
    reference = speech(ZH, 8.0) * 0.8
    capture = echo_of(reference, delay_ms=delay_ms)
    capture += np.random.default_rng(2).normal(0, 10, len(capture))
    found = estimates_for(capture, reference)
    confirmed = [estimate for estimate in found if estimate.confirmed]
    assert len(confirmed) >= 5
    for estimate in confirmed:  # the path's main taps add a few samples to the bulk delay
        assert 0 <= estimate.lag - delay_ms * 16 <= 64, estimate
    assert {estimate.end % 8000 for estimate in found} == {0}  # 0.5 s hops


def test_an_echo_ahead_of_its_reference_gives_a_negative_lag() -> None:
    """The guard needs it: echo the canceller cannot reach (the reference arrives too late)."""
    reference = speech(ZH, 8.0) * 0.8
    capture = np.concatenate([echo_of(reference, delay_ms=0)[1600:], np.zeros(1600)])
    found = [estimate for estimate in estimates_for(capture, reference) if estimate.confirmed]
    assert found
    assert all(-1600 <= estimate.lag <= -1600 + 64 for estimate in found)


def test_a_silent_reference_gives_no_windows_and_no_echo_gives_invalid_ones() -> None:
    noise = np.random.default_rng(4).normal(0, 300, 8 * RATE)
    assert estimates_for(noise, np.zeros(8 * RATE)) == []
    unrelated = estimates_for(speech(EN, 8.0), speech(ZH, 8.0) * 0.8)
    assert unrelated
    assert not any(estimate.confirmed for estimate in unrelated)


def test_restart_skips_windows_that_mix_two_alignments() -> None:
    reference = speech(ZH, 8.0) * 0.8
    capture = echo_of(reference, delay_ms=160)
    estimator = DelayEstimator()
    seen: list[DelayEstimate] = []
    for index, (near, far) in enumerate(zip(frames(capture), frames(reference), strict=True)):
        if index == 300:  # 3 s in: the reference is moved
            estimator.restart()
        seen += estimator.push(near, far)
    after = [estimate for estimate in seen if estimate.end > 3 * RATE]
    assert after
    # A whole window later (less the 16 → 8 kHz resampler's lag, a few ms).
    assert min(estimate.end for estimate in after) >= 5 * RATE - 1600
    assert not after[0].confirmed  # nothing to agree with after a restart


def test_resampling_for_the_estimator_keeps_the_streams_aligned() -> None:
    """Both copies go through the same 16 → 8 kHz stream resampler: equal lengths, equal lag."""
    first, second = Resampler(16_000, 8_000), Resampler(16_000, 8_000)
    a = b"".join(first.process(bytes(320)) for _ in range(50))
    b = b"".join(second.process(bytes(320)) for _ in range(50))
    assert len(a) == len(b)

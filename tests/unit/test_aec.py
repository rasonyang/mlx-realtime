"""The echo-canceller engines (audio/aec.py: WebRTC AEC3 through pywebrtc-audio, and the
vendored SpeexDSP linear canceller of the G7 dual path, with its pre-filter) and the GCC-PHAT
delay estimator (audio/delay.py), on synthetic telephone echo: the fixture speech through a
seeded, decaying random impulse response at 8 kHz after a bulk delay (no ITU G.168
coefficients, which are © ITU). Also the cancellers' CPU per second of audio (M0 B4: 7.8 ms for
AEC3), the SpeexDSP binding's own checks, and the vendored sources against their manifest."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import speexdsp_aec

from mlx_realtime.audio.aec import (
    FRAME_SAMPLES,
    LowPass,
    SpeexLinear,
    WebRtcAec3,
    check_frame,
    make_engine,
)
from mlx_realtime.audio.delay import DelayEstimate, DelayEstimator, span_psr
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
@pytest.mark.parametrize("engine_type", [WebRtcAec3, SpeexLinear])
def test_only_whole_int16_frames_reach_the_engine(
    frame: npt.NDArray[np.generic], engine_type: type[WebRtcAec3 | SpeexLinear]
) -> None:
    """pywebrtc-audio zero-pads a partial frame silently, so the wrappers refuse one."""
    engine = engine_type()
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


def band(signal: Floats) -> Floats:
    """The 200-3600 Hz telephone band of ``signal``."""
    spectrum = np.fft.rfft(signal)
    hz = np.fft.rfftfreq(len(signal), 1 / RATE)
    spectrum[(hz < 200) | (hz > 3600)] = 0
    return np.fft.irfft(spectrum, len(signal))


def cancel(
    engine: WebRtcAec3 | SpeexLinear, capture: Floats, reference: Floats
) -> npt.NDArray[np.float64]:
    return np.concatenate(
        [engine.process(c, r) for c, r in zip(frames(capture), frames(reference), strict=True)]
    ).astype(np.float64)


@pytest.mark.parametrize("preprocess", [False, True])
def test_speex_removes_an_aligned_line_echo_and_leaves_the_caller_whole(preprocess: bool) -> None:
    """The ASR path's canceller: with the echo 20 ms behind its reference (as GCC-PHAT keeps
    it), a 64 ms linear filter cancels it, and in double talk the caller comes through at full
    level: a linear filter has no suppressor to take them for echo (AEC3's does, M0 S11)."""
    far = speech(ZH, 16.0) * 0.8
    caller = speech(EN, 3.0) * 0.5
    near = np.zeros_like(far)
    near[12 * RATE : 15 * RATE] = caller
    noise = np.random.default_rng(1).normal(0, 10, len(far))
    echo = echo_of(far, delay_ms=20, erl_db=6.0)
    engine = SpeexLinear(64, preprocess=preprocess)
    output = cancel(engine, echo + near + noise, far)
    converged = slice(6 * RATE, 12 * RATE)  # echo only, after a few seconds of adaptation
    assert energy_db(echo[converged]) - energy_db(output[converged]) > 20
    talk = slice(12 * RATE, 15 * RATE)
    assert energy_db(output[talk]) > energy_db(caller) - 1.5  # the caller at full level
    if not preprocess:  # the linear output is the caller plus a little residual echo
        # The caller as the same canceller passes them with no echo on the line (its DC notch
        # shifts the lowest frequencies, so the input itself is not the comparison).
        alone = cancel(SpeexLinear(64), near + noise, far)
        residual = band(output[talk] - alone[talk])
        assert energy_db(residual) < energy_db(band(echo[talk])) - 6  # (20+ dB without talk)
    assert engine.resets == 0


def test_speex_cannot_cancel_echo_outside_its_filter() -> None:
    """Why the session aligns Speex's reference: echo 300 ms behind it is left as it is."""
    far = speech(ZH, 10.0) * 0.8
    echo = echo_of(far, delay_ms=300, erl_db=6.0)
    output = cancel(SpeexLinear(64), echo, far)
    late = slice(5 * RATE, 10 * RATE)
    assert energy_db(echo[late]) - energy_db(output[late]) < 3


def test_speex_cpu_per_second_of_audio(record_property: Callable[[str, object], None]) -> None:
    seconds = 20.0
    far = speech(ZH, seconds)
    pairs = list(zip(frames(echo_of(far, delay_ms=20)), frames(far), strict=True))
    engine = SpeexLinear(64)
    began = time.thread_time_ns()
    for near, reference in pairs:
        engine.process(near, reference)
    cpu_ms = (time.thread_time_ns() - began) / 1e6 / seconds
    record_property("speex_cpu_ms_per_audio_s", round(cpu_ms, 2))
    print(f"Speex CPU: {cpu_ms:.2f} ms per second of audio")
    assert cpu_ms < 50


@pytest.mark.parametrize(
    ("capture", "reference", "message"),
    [
        (bytes(318), bytes(320), "capture frame must be 320 bytes"),
        (bytes(320), bytes(640), "reference frame must be 320 bytes"),
        (np.zeros(80, dtype=np.float32), bytes(320), "int16 samples or bytes"),
        (np.zeros(160, dtype=np.int32)[:80], bytes(320), "int16 samples or bytes"),
        (bytes(320), np.zeros(320, dtype=np.uint8)[::2], "contiguous"),
    ],
)
def test_the_speex_binding_checks_every_buffer_itself(
    capture: object, reference: object, message: str
) -> None:
    """The second line of checks, in C, before the native call: length, item type and layout.
    Whatever the caller hands over, SpeexDSP only ever reads the binding's own frame copy."""
    canceller = speexdsp_aec.EchoCanceller(160, 1024, 16000)
    with pytest.raises((ValueError, BufferError), match=message):
        canceller.process(capture, reference)  # pyright: ignore[reportArgumentType]
    whole = canceller.process(bytes(320), np.zeros(160, dtype="<i2"))
    assert whole == bytes(320)
    assert canceller.frames == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((8, 1024, 16000), "frame_size must be"),
        ((160, 100, 16000), "filter_length must be"),
        ((160, 1 << 20, 16000), "filter_length must be"),
        ((160, 1024, 4000), "sample_rate must be"),
    ],
)
def test_the_speex_binding_bounds_its_sizes(arguments: tuple[int, int, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        speexdsp_aec.EchoCanceller(*arguments)


def test_a_closed_or_busy_speex_canceller_refuses_calls_instead_of_racing() -> None:
    canceller = speexdsp_aec.EchoCanceller(160, 1024, 16000)
    inner: list[BaseException] = []

    class Reentrant:
        """A buffer whose export runs Python code: the moment another thread could get in."""

        def __buffer__(self, flags: int) -> memoryview:
            try:
                canceller.process(bytes(320), bytes(320))
            except RuntimeError as error:
                inner.append(error)
            return memoryview(bytes(320))

    assert canceller.process(Reentrant(), bytes(320)) == bytes(320)
    assert len(inner) == 1
    assert "processing a frame" in str(inner[0])
    canceller.close()
    assert canceller.closed
    with pytest.raises(ValueError, match="closed"):
        canceller.process(bytes(320), bytes(320))
    with pytest.raises(ValueError, match="closed"):
        canceller.reset()


def test_speex_runs_without_the_gil() -> None:
    """Sessions share the canceller executor: one session's frame must not block another's."""
    cancellers = [speexdsp_aec.EchoCanceller(160, 1024, 16000) for _ in range(4)]
    frame = np.random.default_rng(5).normal(0, 3000, 160).astype("<i2").tobytes()

    def work(canceller: speexdsp_aec.EchoCanceller) -> None:
        for _ in range(2000):
            canceller.process(frame, frame)

    threads = [threading.Thread(target=work, args=(c,)) for c in cancellers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(c.frames == 2000 for c in cancellers)


def test_the_pre_filter_passes_the_telephone_band_and_stops_the_images() -> None:
    """3.8 kHz low-pass (M0 S11): AICC's linear upsampler leaves images at 4-8 kHz."""
    times = np.arange(RATE) / RATE

    def gain_db(hz: float) -> float:
        tone = np.rint(8000 * np.sin(2 * np.pi * hz * times)).astype(np.int16)
        low = LowPass(3800)
        out = np.concatenate([low(f) for f in frames(tone.astype(np.float64))])
        steady = slice(RATE // 10, None)
        return energy_db(out[steady].astype(np.float64)) - energy_db(
            tone[steady].astype(np.float64)
        )

    assert abs(gain_db(300)) < 0.5
    assert abs(gain_db(3000)) < 1.0
    assert gain_db(5000) < -40
    assert gain_db(7000) < -40
    signal = speech(ZH, 1.0)
    low = LowPass(3800)
    streamed = np.concatenate([low(f) for f in frames(signal)])
    whole = np.convolve(signal, low._taps)[: len(signal)]  # pyright: ignore[reportPrivateUsage]
    assert np.max(np.abs(streamed - np.clip(np.rint(whole), -32768, 32767))) <= 1
    assert low.delay == 31


VENDORED = Path(__file__).parents[2] / "vendor" / "speexdsp-aec" / "speexdsp"


def test_the_vendored_speexdsp_sources_are_the_release_tarballs() -> None:
    """Every vendored file is listed with the sha256 of its copy in speexdsp-1.2.1.tar.gz
    (itself recorded with Xiph's published sha256), and none was changed."""
    manifest = json.loads((VENDORED / "SOURCE.json").read_text(encoding="utf-8"))
    assert (manifest["version"], speexdsp_aec.SPEEXDSP_VERSION) == ("1.2.1", "1.2.1")
    assert manifest["url"] == "https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz"
    assert len(manifest["sha256"]) == 64
    present = {
        path.relative_to(VENDORED).as_posix()
        for path in VENDORED.rglob("*")
        if path.is_file() and path.name != "SOURCE.json"
    }
    assert present == set(manifest["files"])
    assert "COPYING" in present  # the BSD license notice travels with the sources
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == digest, name


def _reference(seconds: float, seed: int = 3) -> npt.NDArray[np.int16]:
    return np.random.default_rng(seed).normal(0.0, 3000.0, int(16_000 * seconds)).astype(np.int16)


def test_a_delayed_filtered_clipped_copy_has_a_high_psr_at_its_lag() -> None:
    lag, span = 6400, 11_200  # 400 ms late, 700 ms of input
    reference = _reference(3.0)
    start = 20_000
    echo = np.zeros(len(reference))
    echo[lag:] = reference[:-lag]
    echo = np.convolve(echo, np.array([0.5, 0.3, -0.2, 0.1]), mode="same")  # a short filter
    echo = np.clip(echo * 0.5, -1500, 1500)  # and a nonlinear clip
    capture = echo[start : start + span].astype(np.int16)
    # the reference holds the second before the capture, then the capture's span
    window = reference[start - 16_000 : start + span]
    found = span_psr(capture, window, 16_000)
    assert found is not None
    psr, found_lag = found
    assert psr > 20
    assert abs(found_lag - lag) <= 8


def test_independent_noise_and_speech_have_a_low_psr() -> None:
    reference = _reference(3.0)
    start, span = 20_000, 11_200
    window = reference[start - 16_000 : start + span]
    noise = _reference(3.0, seed=4)[start : start + span]
    found = span_psr(noise, window, 16_000)
    assert found is not None
    assert found[0] < 10
    speech = np.frombuffer(clip(EN), dtype="<i2")[16_000 : 16_000 + span]
    found = span_psr(speech, window, 16_000)
    assert found is not None
    assert found[0] < 10


def test_a_silent_reference_has_no_psr() -> None:
    assert span_psr(_reference(0.7), np.zeros(16_000 + 11_200, dtype=np.int16), 16_000) is None
    with pytest.raises(ValueError, match="lag_max samples longer"):
        span_psr(_reference(0.7), _reference(0.7), 16_000)


def test_the_correlation_costs_a_few_milliseconds() -> None:
    reference = _reference(3.0)
    capture = _reference(3.0, seed=5)[16_000 : 16_000 + 28_800]  # the 1.8 s of a long check
    window = reference[:44_800]
    began = time.perf_counter()
    for _ in range(5):
        span_psr(capture, window, 16_000)
    assert (time.perf_counter() - began) / 5 < 0.1

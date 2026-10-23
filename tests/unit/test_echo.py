"""The session's echo reference and canceller stage (session/echo.py, D17) on a fake clock: the
capture clock, the reference placed on the AICC playout model (lead, flush, truncate re-anchoring),
the hold-back while a response may still add audio, the playback state the barge-in rules read,
the GCC-PHAT guard's re-alignment and the bypass on a line without echo; and the overlap measure
of echo-turn suppression."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import pytest

from mlx_realtime.audio.aec import FRAME_SAMPLES
from mlx_realtime.audio.delay import DelayEstimate
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.config import AecConfig, BargeInConfig
from mlx_realtime.session.echo import (
    CaptureClock,
    EchoParams,
    EchoReference,
    EchoStage,
    PlayedSpan,
)
from mlx_realtime.text.similarity import lcs_length, overlap, units

RATE = 16_000
TICK = 320
LEAD = 960  # 60 ms


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


def reference(clock: FakeClock, **aec: object) -> EchoReference:
    params = EchoParams.of(AecConfig.model_validate(aec), BargeInConfig())
    return EchoReference(params, clock)


def stream(echo: EchoReference, clock: FakeClock, seconds: float, *, start: int = 0) -> int:
    """The client streams 20 ms of input every 20 ms; returns the input sample reached."""
    end = start
    for _ in range(round(seconds / 0.02)):
        clock.t += 0.02
        end += 320
        echo.capture_arrived(end)
    return end


def offset_of(clock: FakeClock, end: int) -> int:
    return round(clock.t * RATE) - end


def wire(samples: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 3000, samples)).astype("<i2").tobytes()


def at_16k(pcm24: bytes) -> npt.NDArray[np.int16]:
    resampler = Resampler(24_000, 16_000)
    return np.frombuffer(resampler.process(pcm24) + resampler.flush(), dtype="<i2")


def test_the_capture_clock_takes_the_least_delayed_arrival() -> None:
    capture = CaptureClock(window=RATE)
    assert capture.offset is None
    for end, jitter in [(0, 0), (320, 30), (640, 10), (960, 30)]:
        capture.observe(end, 1000 + end + jitter)
    assert capture.offset == 1000
    end = 960 + 20 * 320  # 0.4 s later: the earlier minimum is still in the window
    capture.observe(end, 1000 + end + 25)
    assert capture.offset == 1000
    end = 960 + 60 * 320  # over a second later: it expired
    capture.observe(end, 1000 + end + 25)
    assert capture.offset == 1025


def test_the_reference_is_our_audio_where_the_client_plays_it_less_the_lead() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    offset = offset_of(clock, end)
    echo.response_started()
    echo.audio_queued(1)
    pcm = wire(2400 * 2)  # 200 ms at 24 kHz
    sent_at = round(clock.t * RATE)
    echo.audio_sent("item_1", 1, pcm)
    echo.response_finished(1)  # all sent: the item is complete, its tail resampled
    first_play = -(-sent_at // TICK) * TICK  # the first tick at or after the arrival (phase 0)
    stream(echo, clock, 0.6, start=end)
    start = first_play - offset - LEAD
    expected = at_16k(pcm)
    rendered = echo.render(start - 160, 160 + len(expected))
    assert not rendered[:160].any()  # nothing before the playout
    assert np.array_equal(rendered[160:], expected[: len(rendered) - 160])


def test_the_reference_is_held_back_while_a_response_may_still_add_audio() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    offset = offset_of(clock, end)
    assert echo.final_until() is None  # idle: nothing can change
    echo.response_started()
    final = echo.final_until()
    assert final is not None
    assert final == round(clock.t * RATE) - offset - LEAD
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 10))  # 1 s queued at the client from now on
    clock.t += 0.03  # it has started to play
    playing = echo.final_until()
    assert playing is not None
    assert playing > final + RATE * 9 // 10  # the client's queue determines what plays
    echo.response_finished(1)
    assert echo.final_until() is None


def test_a_flush_drops_what_the_client_never_played() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    offset = offset_of(clock, end)
    echo.response_started()
    echo.audio_queued(1)
    sent_at = round(clock.t * RATE)
    echo.audio_sent("item_1", 1, wire(2400 * 10))
    end = stream(echo, clock, 0.3, start=end)
    echo.flush(1)  # a barge-in 300 ms in
    echo.response_finished(1)
    stream(echo, clock, 1.0, start=end)
    first_play = -(-sent_at // TICK) * TICK
    played = echo.render(first_play - offset - LEAD, RATE)  # one second from the first play
    flushed_at = round(clock.t * RATE) - RATE  # when flush() ran
    cut = flushed_at - first_play  # the frame playing at the flush still plays out
    assert played[: cut - TICK].any()
    assert not played[cut + TICK :].any()


def test_a_truncate_drops_the_rest_and_re_anchors_the_model() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.response_started()
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 20))  # 2 s
    echo.response_finished(1)
    stream(echo, clock, 0.5, start=end)
    (span,) = echo.played_spans(0, 10**9, 0)
    played = span.end_ms // 20  # about 25 frames
    echo.truncated("item_1", 800)  # the client played 40: the model is late
    assert echo.anchor_ms == -(40 - played) * 20
    assert echo.stats.anchors == 1
    assert echo.played_spans(0, 10**9, 0) == [PlayedSpan("item_1", 0, played * 20)]
    echo.truncated("item_1", (played + 1) * 20)  # within a frame of the model: no re-anchoring
    assert echo.stats.anchors == 1


def test_playback_spans_the_response_its_echo_tail_and_its_guard() -> None:
    clock = FakeClock()
    echo = reference(clock)  # barge-in defaults: a 2 s guard; echo tail 450 ms
    end = stream(echo, clock, 1.0)
    offset = offset_of(clock, end)
    assert not echo.playback(end).playing
    echo.response_started()
    echo.audio_queued(1)
    clock.t += 0.005  # between two of the client's ticks
    sent_at = round(clock.t * RATE)
    echo.audio_sent("item_1", 1, wire(2400 * 30))  # 3 s
    echo.response_finished(1)
    first_play = -(-sent_at // TICK) * TICK
    stream(echo, clock, 4.0, start=end)
    inside = echo.playback(first_play - offset + RATE)
    assert inside.playing
    assert inside.guard_end == first_play - offset + 2 * RATE
    assert echo.playback(first_play - offset + 3 * RATE + 400 * 16).playing  # the echo tail
    assert not echo.playback(first_play - offset + 3 * RATE + 500 * 16).playing
    assert not echo.playback(first_play - offset - 100).playing


def test_the_guard_re_aligns_only_outside_the_keep_range() -> None:
    clock = FakeClock()
    echo = reference(clock)
    late = DelayEstimate(0, valid=True, lag=600 * 16, psr=20.0, confirmed=True)
    fine = DelayEstimate(0, valid=True, lag=200 * 16, psr=20.0, confirmed=True)
    early = DelayEstimate(0, valid=True, lag=2 * 16, psr=20.0, confirmed=True)
    assert not echo.delay_estimate(0, fine)
    assert echo.shift_ms == 0
    assert not echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=600 * 16, psr=20.0))
    assert echo.shift_ms == 0  # one window is not enough
    assert echo.delay_estimate(0, late)
    assert echo.shift_ms == 540  # the residual becomes the 60 ms lead
    assert echo.delay_estimate(0, early)
    assert echo.shift_ms == 540 + 2 - 60
    assert echo.stats.realignments == 2


def test_no_echo_bypasses_the_canceller_until_echo_is_confirmed() -> None:
    clock = FakeClock()
    echo = reference(clock, bypass_without_echo_ms=1500)
    silent = DelayEstimate(0, valid=False, psr=4.0)
    for _ in range(2):
        echo.delay_estimate(0, silent)
    assert not echo.echo_absent
    echo.delay_estimate(0, silent)  # 1.5 s of playing reference without any echo of it
    assert echo.echo_absent
    echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=3200, psr=12.0))  # a chance peak
    assert echo.echo_absent
    echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=3200, psr=12.0, confirmed=True))
    assert not echo.echo_absent
    never = reference(clock, bypass_without_echo_ms=None)
    for _ in range(10):
        never.delay_estimate(0, silent)
    assert not never.echo_absent


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]] = []

    def process(
        self, capture: npt.NDArray[np.int16], reference: npt.NDArray[np.int16]
    ) -> npt.NDArray[np.int16]:
        assert capture.shape == reference.shape == (FRAME_SAMPLES,)
        self.calls.append((capture.copy(), reference.copy()))
        return (capture // 2).astype(np.int16)


@pytest.fixture
def executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(1)


async def test_the_stage_feeds_frames_in_lock_step_with_their_reference(
    executor: ThreadPoolExecutor,
) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False)
    stream(echo, clock, 1.0)
    engine = RecordingEngine()
    stage = EchoStage(engine, echo, executor)
    pcm = np.arange(960, dtype=np.int16)
    out = await stage.feed(16_000, pcm[:500].tobytes())  # three whole frames, 20 samples left
    out += await stage.feed(16_500, pcm[500:].tobytes())  # three more
    assert [start for start, _ in out] == [16_000, 16_480]
    processed = np.frombuffer(b"".join(chunk for _, chunk in out), dtype=np.int16)
    assert np.array_equal(processed, pcm // 2)
    assert len(engine.calls) == 6
    for index, (capture, ref) in enumerate(engine.calls):
        start = 16_000 + index * FRAME_SAMPLES
        assert np.array_equal(capture, pcm[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES])
        assert np.array_equal(ref, echo.render(start, FRAME_SAMPLES))


async def test_the_stage_holds_frames_until_their_reference_is_final(
    executor: ThreadPoolExecutor,
) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False)
    end = stream(echo, clock, 1.0)
    echo.response_started()  # audio may come any moment: the reference ahead is unknown
    stage = EchoStage(RecordingEngine(), echo, executor)
    out = await stage.feed(end, bytes(640))  # 20 ms arriving now
    assert out == []  # its reference is 60 ms ahead: not final yet
    due = stage.wake_at()
    assert due is not None
    assert due == pytest.approx(clock.t + 0.07, abs=0.001)  # the lead and one frame
    clock.t += 0.08
    released = await stage.release()
    assert sum(len(chunk) for _, chunk in released) == 640
    assert stage.wake_at() is None


async def test_a_gap_releases_what_is_held_and_passes_the_partial_frame(
    executor: ThreadPoolExecutor,
) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False)
    stream(echo, clock, 1.0)
    engine = RecordingEngine()
    stage = EchoStage(engine, echo, executor)
    first = await stage.feed(0, bytes(400))  # one whole frame, 40 samples held
    out = await stage.feed(5_000, bytes(320))  # a commit or clear moved the timeline
    assert [(start, len(chunk)) for start, chunk in first + out] == [
        (0, 320),
        (160, 80),
        (5_000, 320),
    ]
    assert len(engine.calls) == 2  # the partial frame never reached the engine


async def test_a_bypassed_canceller_gets_a_silent_reference(executor: ThreadPoolExecutor) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False)
    stream(echo, clock, 1.0)
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 10))
    echo.response_finished(1)
    engine = RecordingEngine()
    stage = EchoStage(engine, echo, executor)
    echo.echo_absent = True
    for start in range(0, 32_000, 3200):
        await stage.feed(16_000 + start, bytes(6400))
    assert all(not ref.any() for _, ref in engine.calls)


def test_units_are_cjk_characters_and_latin_words() -> None:
    assert units("Hello, World! 你好。iPhone15，ＡＢＣ") == [
        "hello", "world", "你", "好", "iphone15", "abc",
    ]  # fmt: skip


def brute_lcs(first: list[str], second: list[str]) -> int:
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for i, j in itertools.product(range(len(first)), range(len(second))):
        table[i + 1][j + 1] = (
            table[i][j] + 1 if first[i] == second[j] else max(table[i][j + 1], table[i + 1][j])
        )
    return table[-1][-1]


def test_lcs_matches_dynamic_programming() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        first = [str(value) for value in rng.integers(0, 6, size=int(rng.integers(0, 40)))]
        second = [str(value) for value in rng.integers(0, 6, size=int(rng.integers(0, 60)))]
        assert lcs_length(first, second) == brute_lcs(first, second)


@pytest.mark.parametrize(
    ("transcript", "played", "low", "high"),
    [
        (
            "但鸟类还是有很多地方跟恐龙相似",
            "但鸟类还是有很多地方跟恐龙相似。它们的骨骼也很像。",
            1.0,
            1.0,
        ),
        ("鸟类还是有很多地方跟", "但鸟类还是有很多地方跟恐龙相似。", 1.0, 1.0),
        ("等一下，我想问问价格。", "但鸟类还是有很多地方跟恐龙相似。", 0.0, 0.2),
        (
            "Many people don't think about them",
            "Good morning, NovaNet support. How can I help you today with your account?",
            0.0,
            0.2,
        ),
        (
            "NovaNet support, how can I help",
            "Good morning, NovaNet support. How can I help?",
            1.0,
            1.0,
        ),
    ],
)
def test_overlap(transcript: str, played: str, low: float, high: float) -> None:
    assert low <= overlap(transcript, played) <= high

"""The session's echo reference and canceller stage (session/echo.py, D17) on a fake clock: the
capture clock, the reference placed on the AICC playout model (lead, flush, truncate re-anchoring),
the hold-back while a response may still add audio, the playback state the barge-in rules read,
the GCC-PHAT guard's re-alignment and the bypass on a line without echo; the G7 dual path (the ASR
path's Speex reference aligned from the guard's estimates, kept within its keep range, passed
through until aligned); and the overlap measure of echo-turn suppression."""

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
    AsrPath,
    CaptureClock,
    EchoParams,
    EchoReference,
    EchoStage,
    PlayedSpan,
    log_echo_stats,
)
from mlx_realtime.text.similarity import contiguity, lcs_length, overlap, units, unmatched

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


def send(echo: EchoReference, item: str, generation: int, frames: int, *, seed: int = 0) -> None:
    echo.audio_queued(generation)
    echo.audio_sent(item, generation, wire(480 * frames, seed))  # 480 samples: 20 ms at 24 kHz


def test_a_clear_drops_the_whole_queue_a_previous_responses_tail_included() -> None:
    """AICC's ClearTx empties its TX queue: B, queued behind A's unplayed tail, goes with it."""
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.response_started()
    send(echo, "item_a", 1, 100)  # 2 s
    echo.response_finished(1)
    end = stream(echo, clock, 1.0, start=end)  # half of A has played
    echo.response_started()
    send(echo, "item_b", 2, 50)
    assert echo._model.queued > 50
    echo.flush(2, speech=True)  # past the guard: AICC acts
    assert echo._model.queued == 0
    assert all(item.queued == 0 and item.complete for item in echo._items.values())
    stream(echo, clock, 0.5, start=end)
    (span_a,) = (s for s in echo.played_spans(0, 10**9, 0) if s.item_id == "item_a")
    assert span_a.end_ms < 1200  # A's tail never played


def test_a_truncate_of_a_queued_item_clears_the_tail_behind_which_it_waited() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.response_started()
    send(echo, "item_a", 1, 100)
    echo.response_finished(1)
    end = stream(echo, clock, 1.0, start=end)
    echo.response_started()
    send(echo, "item_b", 2, 50)
    echo.truncated("item_b", 0)  # AICC played none of B: it was behind A's tail
    assert echo._model.queued == 0
    assert all(item.queued == 0 and item.complete for item in echo._items.values())
    stream(echo, clock, 0.5, start=end)
    assert not [sp for sp in echo.played_spans(0, 10**9, 0) if sp.item_id == "item_b"]


def resumed_after_an_underrun(clock: FakeClock, echo: EchoReference) -> int:
    """5 frames play out and the client stops (an underrun), then 100 more arrive: returns the
    input position."""
    end = stream(echo, clock, 1.0)
    echo.response_started()
    send(echo, "item_1", 1, 5)
    end = stream(echo, clock, 0.5, start=end)
    echo._advance(round(clock.t * RATE))
    assert not echo._model.playing
    assert echo._model.queued == 0
    send(echo, "item_1", 1, 100)
    return end


def test_a_speech_barge_in_within_the_guard_of_a_resume_leaves_the_queue_playing() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = resumed_after_an_underrun(clock, echo)
    end = stream(echo, clock, 0.4, start=end)  # 400 ms after the resume: inside 800 ms
    echo.flush(1, speech=True)  # AICC ignores the detection: no ClearTx
    assert echo._model.queued > 50
    stream(echo, clock, 0.2, start=end)
    assert echo._model.queued > 30  # and the audio goes on playing
    echo.truncated("item_1", 200)  # a truncate that comes anyway still applies
    assert echo._model.queued == 0


def test_a_speech_barge_in_past_the_guard_of_a_resume_clears_the_queue() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = resumed_after_an_underrun(clock, echo)
    stream(echo, clock, 0.9, start=end)
    echo.flush(1, speech=True)
    assert echo._model.queued == 0


def test_a_cancel_aicc_initiated_clears_whatever_the_guard() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = resumed_after_an_underrun(clock, echo)
    stream(echo, clock, 0.1, start=end)
    echo.flush(1)  # AICC's interrupt (a keypress, its own detection): ClearTx, then the cancel
    assert echo._model.queued == 0


def test_the_audible_clock_runs_on_through_a_fed_queue_and_restarts_after_a_gap() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 0.5)
    echo.response_started()
    send(echo, "item_1", 1, 10)
    first = echo._audible_since
    assert first == round(clock.t * RATE)
    for _ in range(4):  # fed in real time, 200 ms, a frame at a time: never audibly idle
        end = stream(echo, clock, 0.02, start=end)
        send(echo, "item_1", 1, 1)
    assert echo._audible_since == first
    end = stream(echo, clock, 1.0, start=end)  # a gap well past the slack
    send(echo, "item_1", 1, 1)
    assert echo._audible_since == round(clock.t * RATE)


def test_keepalive_silence_to_a_stopped_queue_starts_the_audible_clock() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 0.5)
    echo.response_started()
    echo.audio_queued(1)
    echo.audio_sent("keepalive", 1, bytes(480 * 2))  # one 20 ms frame of silence
    assert echo._audible_since == round(clock.t * RATE)
    end = stream(echo, clock, 0.3, start=end)  # played out; AICC stopped again
    send(echo, "item_1", 1, 50)  # speech 300 ms later: a new start, 800 ms guard again
    assert echo._audible_since == round(clock.t * RATE)
    stream(echo, clock, 0.4, start=end)
    echo.flush(1, speech=True)  # ignored: the silence's start did not extend the guard past it
    assert echo._model.queued > 0


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
    # The estimator's own confirmation (the next, overlapping window agrees) is not enough
    # while bypassed: a caller talking over us makes such pairs about every 10 s.
    echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=3200, psr=12.0, confirmed=True))
    assert echo.echo_absent
    assert echo.stats.realignments == 0  # nor does the chance pair move the reference
    echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=3205, psr=12.0, confirmed=True))
    assert not echo.echo_absent  # a third agreeing window: echo
    never = reference(clock, bypass_without_echo_ms=None)
    for _ in range(10):
        never.delay_estimate(0, silent)
    assert not never.echo_absent


def test_leaving_the_bypass_needs_three_agreeing_windows_of_a_clear_peak() -> None:
    clock = FakeClock()
    echo = reference(clock, bypass_without_echo_ms=1500)
    silent = DelayEstimate(0, valid=False, psr=4.0)
    for _ in range(3):
        echo.delay_estimate(0, silent)
    assert echo.echo_absent

    def window(lag_ms: float, psr: float = 12.0) -> DelayEstimate:
        return DelayEstimate(0, valid=True, lag=round(lag_ms * 16), psr=psr, confirmed=True)

    for run in (
        [window(200), window(200)],  # the estimator's own confirmation: two windows
        [window(200), window(200), window(260)],  # a third that disagrees starts a new run
        [window(200, psr=9.0), window(200), window(200)],  # too weak a peak
    ):
        echo.delay_estimate(0, silent)  # a window without a peak ends any run
        for estimate in run:
            echo.delay_estimate(0, estimate)
        assert echo.echo_absent, run
    echo.delay_estimate(0, silent)
    for lag_ms in (260, 260, 260.5):
        echo.delay_estimate(0, window(lag_ms))
    assert not echo.echo_absent
    assert echo.stats.bypasses == 1
    echo.delay_estimate(0, window(260))  # back on: the estimator's own confirmation applies
    assert not echo.echo_absent


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
    echo = reference(clock, delay_guard=False, asr_input="aec3")
    stream(echo, clock, 1.0)
    engine = RecordingEngine()
    stage = EchoStage(engine, echo, executor)
    pcm = np.arange(960, dtype=np.int16)
    out = await stage.feed(16_000, pcm[:500].tobytes())  # three whole frames, 20 samples left
    out += await stage.feed(16_500, pcm[500:].tobytes())  # three more
    assert [chunk.start for chunk in out] == [16_000, 16_480]
    assert all(chunk.asr is None for chunk in out)  # no ASR path: the ASR keeps the input
    processed = np.frombuffer(b"".join(chunk.pcm for chunk in out), dtype=np.int16)
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
    echo = reference(clock, delay_guard=False, asr_input="aec3")
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
    assert sum(len(chunk.pcm) for chunk in released) == 640
    assert stage.wake_at() is None


async def test_a_gap_releases_what_is_held_and_passes_the_partial_frame(
    executor: ThreadPoolExecutor,
) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False, asr_input="aec3")
    stream(echo, clock, 1.0)
    engine = RecordingEngine()
    stage = EchoStage(engine, echo, executor)
    first = await stage.feed(0, bytes(400))  # one whole frame, 40 samples held
    out = await stage.feed(5_000, bytes(320))  # a commit or clear moved the timeline
    assert [(chunk.start, len(chunk.pcm)) for chunk in first + out] == [
        (0, 320),
        (160, 80),
        (5_000, 320),
    ]
    assert len(engine.calls) == 2  # the partial frame never reached the engine


async def test_a_bypassed_canceller_gets_a_silent_reference(executor: ThreadPoolExecutor) -> None:
    clock = FakeClock()
    echo = reference(clock, delay_guard=False, asr_input="aec3")
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


def confirmed(lag_ms: float) -> DelayEstimate:
    return DelayEstimate(0, valid=True, lag=round(lag_ms * 16), psr=20.0, confirmed=True)


def test_the_asr_paths_reference_is_aligned_from_confirmed_estimates() -> None:
    """Speex's reference is placed so the echo trails it by 20 ms, and moved again only when a
    confirmed estimate leaves [0, 44] ms (M0 S11: re-shifting on every estimate kept the
    filter from converging). AEC3's guard is independent: its keep range is [5, 450] ms."""
    clock = FakeClock()
    echo = reference(clock)
    assert not echo.speex_aligned
    assert echo.speex_shift_ms is None
    echo.delay_estimate(0, DelayEstimate(0, valid=True, lag=183 * 16, psr=20.0))
    assert not echo.speex_aligned  # one window can peak by chance
    echo.delay_estimate(16_000, confirmed(183))
    assert echo.speex_shift_ms == 163  # the echo now trails Speex's reference by 20 ms
    assert echo.stats.speex_aligned_at == 16_000
    assert echo.shift_ms == 0  # 183 ms is inside AEC3's own keep range
    for drift in (203, 206, 163):  # within [0, 44] ms of Speex's reference: left alone
        echo.delay_estimate(0, confirmed(drift))
        assert echo.speex_shift_ms == 163
    echo.delay_estimate(0, confirmed(160))  # the echo came ahead of the reference
    assert echo.speex_shift_ms == 140
    echo.delay_estimate(0, confirmed(185.5))  # 45.5 ms behind: out of the keep range
    assert echo.speex_shift_ms == 165
    assert echo.stats.speex_realignments == 2


def test_the_asr_path_follows_the_guards_re_alignment_of_aec3() -> None:
    """The estimate is measured against AEC3's (shifted) reference: Speex's shift adds it."""
    clock = FakeClock()
    echo = reference(clock)
    assert echo.delay_estimate(0, confirmed(600))  # AEC3's guard moves its reference
    assert (echo.shift_ms, echo.speex_shift_ms) == (540, 580)
    echo.delay_estimate(0, confirmed(60))  # the same echo, against the moved reference
    assert (echo.shift_ms, echo.speex_shift_ms) == (540, 580)


@pytest.mark.parametrize("asr_input", ["aec3", "raw"])
def test_without_the_speex_input_nothing_is_aligned_for_it(asr_input: str) -> None:
    clock = FakeClock()
    echo = reference(clock, asr_input=asr_input)
    echo.delay_estimate(0, confirmed(183))
    assert not echo.speex_aligned
    assert echo.render_speex(0, 160) is None


def test_the_asr_paths_reference_is_ours_placed_by_its_own_shift() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.response_started()
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 10))
    echo.response_finished(1)
    end = stream(echo, clock, 1.5, start=end)
    assert echo.render_speex(end - 16_000, 16_000) is None  # not aligned yet
    echo.delay_estimate(0, confirmed(183))
    start = end - 20_000
    speex = echo.render_speex(start, 8000)
    assert speex is not None
    shift = 163 * 16
    assert np.array_equal(speex, echo.render(start - shift, 8000))  # AEC3's shift is 0
    assert speex.any()


def test_re_anchoring_the_model_keeps_the_asr_paths_alignment() -> None:
    """A client truncate corrects the model's timing for later audio; Speex's reference stays
    where GCC-PHAT last measured the echo (else it would leave its 0-44 ms keep range)."""
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.delay_estimate(0, confirmed(183))
    echo.response_started()
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 20))
    echo.response_finished(1)
    stream(echo, clock, 0.5, start=end)
    (span,) = echo.played_spans(0, 10**9, 0)
    played = span.end_ms // 20
    echo.truncated("item_1", (played + 3) * 20)  # the client played 3 frames more
    assert echo.anchor_ms == -60  # later audio is placed 60 ms earlier ...
    assert echo.speex_shift_ms == 163 + 60  # ... and read 60 ms earlier for Speex
    assert echo.shift_ms == 0  # AEC3 follows the model (its own estimator absorbs the rest)
    now = round(clock.t * RATE) - offset_of(clock, end + 8000)  # the input arriving now
    echo.delay_estimate(now + 16_000, confirmed(183))  # a window still over the old placement
    assert echo.speex_shift_ms == 223
    # Audio placed after the correction: the same echo is 60 ms later against the model.
    echo.delay_estimate(now + 2 * 16_000, confirmed(243))
    assert echo.speex_shift_ms == 223
    echo.delay_estimate(now + 2 * 16_000, confirmed(183))  # the echo really moved: re-aligned
    assert echo.speex_shift_ms == 163


def test_the_hold_back_waits_for_the_reference_that_reads_furthest_ahead() -> None:
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.response_started()
    before = echo.final_until()
    assert before is not None
    echo.delay_estimate(0, confirmed(600))  # AEC3 shifted 540 ms back, Speex 580 ms
    assert echo.final_until() == before + 540 * 16  # the smaller shift binds
    assert echo.final_at(end) == pytest.approx(
        (end + offset_of(clock, end) + LEAD - 540 * 16) / RATE
    )


class TrackingEngine(RecordingEngine):
    """Records like RecordingEngine; returns the capture minus a quarter (the ASR path)."""

    def process(
        self, capture: npt.NDArray[np.int16], reference: npt.NDArray[np.int16]
    ) -> npt.NDArray[np.int16]:
        super().process(capture, reference)
        return (capture - capture // 4).astype(np.int16)


async def test_the_stage_runs_the_asr_path_once_aligned(executor: ThreadPoolExecutor) -> None:
    """G7 dual path: until Speex's reference is aligned the ASR keeps the (pre-filtered) input;
    then its engine gets each frame with Speex's own reference, AEC3's bypass or not."""
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    echo.audio_queued(1)
    echo.audio_sent("item_1", 1, wire(2400 * 30))
    echo.response_finished(1)
    end = stream(echo, clock, 1.0, start=end)
    aec3, speex = RecordingEngine(), TrackingEngine()
    stage = EchoStage(aec3, echo, executor, AsrPath(speex, prefilter_hz=None))
    pcm = (np.arange(1600) % 300 + 100).astype(np.int16)
    (first,) = await stage.feed(end - 3200, pcm.tobytes())
    assert first.asr == pcm.tobytes()  # not aligned: passed through as received
    assert speex.calls == []
    echo.delay_estimate(0, confirmed(183))
    (second,) = await stage.feed(end - 1600, pcm.tobytes())
    assert second.pcm == (pcm // 2).astype(np.int16).tobytes()  # AEC3's output, for detection
    assert second.asr == (pcm - pcm // 4).astype(np.int16).tobytes()
    assert len(speex.calls) == 10
    for index, (_, ref) in enumerate(speex.calls):
        frame_start = end - 1600 + index * FRAME_SAMPLES
        rendered = echo.render_speex(frame_start, FRAME_SAMPLES)
        assert rendered is not None
        assert np.array_equal(ref, rendered)
    echo.echo_absent = True  # AEC3's bypass: a linear filter needs none (it would only lose
    await stage.feed(end, pcm.tobytes())  # the echo when the bypass misfires in double talk)
    assert all(not ref.any() for _, ref in aec3.calls[20:])
    assert any(ref.any() for _, ref in speex.calls[10:])
    assert echo.stats.speex_ns > 0


async def test_the_asr_paths_pre_filter_applies_once_it_cancels(
    executor: ThreadPoolExecutor,
) -> None:
    """The 3.8 kHz pre-filter feeds Speex once its reference is aligned (AICC's upsampler
    leaves images at 4-8 kHz that no linear filter can cancel); before that the input passes
    through whole (a line without echo, or a wideband client, keeps its band)."""
    clock = FakeClock()
    echo = reference(clock)
    end = stream(echo, clock, 1.0)
    speex = TrackingEngine()
    stage = EchoStage(RecordingEngine(), echo, executor, AsrPath(speex, 3800))
    times = np.arange(3200) / RATE
    tone = (8000 * np.sin(2 * np.pi * 6000 * times)).astype(np.int16)  # an image's frequency
    (before,) = await stage.feed(end - 6400, tone[:1600].tobytes())
    assert before.asr == tone[:1600].tobytes()
    echo.delay_estimate(0, confirmed(183))
    await stage.feed(end - 4800, tone[1600:].tobytes())
    fed = np.concatenate([capture for capture, _ in speex.calls]).astype(np.float64)[200:]
    assert np.sqrt(np.mean(fed**2)) < 8000 / np.sqrt(2) / 100  # below -40 dB


def test_the_summary_reports_both_paths(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    echo = reference(clock)
    echo.delay_estimate(32_000, confirmed(183))
    with caplog.at_level("INFO", logger="mlx_realtime.session.echo"):
        log_echo_stats(echo, {"barge_ins": 0})
    (record,) = [r for r in caplog.records if r.getMessage() == "echo summary"]
    fields = record.__dict__
    assert (fields["speex_aligned_at_ms"], fields["speex_shift_ms"]) == (2000, 163)
    assert (fields["speex_realignments"], fields["speex_resets"]) == (0, 0)


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
    ("transcript", "played", "new"),
    [
        ("鸟类还是有很多地方", "但鸟类还是有很多地方跟恐龙相似。", 0),  # all ours
        ("鸟类 还是 我要", "但鸟类还是有很多地方跟恐龙相似。", 2),  # two words that are not
        ("我想问价格", "但鸟类还是有很多地方跟恐龙相似。", 5),
        (
            "地方 鸟类",
            "但鸟类还是有很多地方跟恐龙相似。",
            2,
        ),  # out of order: only one of them counts
        ("hello there", "", 2),  # nothing played: all of it is new
        ("", "但鸟类", 0),
        ("Many people really think", "Many people think about dinosaurs", 1),
    ],
)
def test_unmatched_units_are_those_outside_the_longest_in_order_match(
    transcript: str, played: str, new: int
) -> None:
    assert unmatched(transcript, played) == new


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


GREETING = (
    "您好，欢迎来电。我是一位科普讲解员，您可以随便问我一个简短的问题，"
    "比如天为什么是蓝色的，或者咖啡是怎么被发现的。"
)


@pytest.mark.parametrize(
    ("transcript", "played", "low", "high"),
    [
        ("但鸟类还是有很多地方跟恐龙相似", "但鸟类还是有很多地方跟恐龙相似。它们很像。", 1.0, 1.0),
        # Live line, call 2: the caller's question reuses only "为什么" in a row (3 of 7 units),
        # though the in-order overlap counts 6 of 7.
        ("为什么天是蓝的？", GREETING, 3 / 7, 3 / 7),
        # Long echo with ASR errors (two substituted characters, one dropped): runs of 8, 6, 5.
        (
            "您好欢迎来电我是一位科普讲解员您可以随便问我一个简短的问题比如天为什么是兰色的或者咖啡是怎被发现的",
            GREETING,
            0.8,
            1.0,
        ),
        ("Many people don't think about them", "Many people think about dinosaurs", 0.0, 0.0),
        ("many PEOPLE think about", "Many people think about dinosaurs", 1.0, 1.0),
        ("people think", "Many people think about dinosaurs", 0.0, 0.0),  # shorter than a run
        ("地方 鸟类 相似", "但鸟类还是有很多地方跟恐龙相似。", 0.0, 0.0),  # scattered, out of order
        ("hello there", "", 0.0, 0.0),
        ("", GREETING, 0.0, 0.0),
    ],
)
def test_contiguity(transcript: str, played: str, low: float, high: float) -> None:
    assert low <= contiguity(transcript, played) <= high

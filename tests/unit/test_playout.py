"""The model of AICC's playout (audio/playout.py; AICC voice/rtp.go ``Run``) and the tick-phase
learner (M0 S11: the gateway must learn AICC's 20 ms grid)."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_realtime.audio.playout import (
    PlayoutModel,
    PlayoutParams,
    RunStart,
    TickPhaseLearner,
    ceil_tick,
)

TICK = 320  # 20 ms at 16 kHz
AICC = PlayoutParams(tick=TICK, prebuffer_frames=3, prebuffer=3 * TICK, underrun_ticks=2)


def model(phase: int = 0) -> tuple[PlayoutModel[int], list[RunStart]]:
    starts: list[RunStart] = []
    playout = PlayoutModel[int](AICC, on_run_start=starts.append)
    playout.set_phase(phase)
    return playout, starts


def plays(playout: PlayoutModel[int], until: int) -> list[tuple[int, int]]:
    return [(entry.handle, entry.start) for entry in playout.advance(until)]


def test_ceil_tick() -> None:
    assert ceil_tick(0, 0, TICK) == 0
    assert ceil_tick(1, 0, TICK) == TICK
    assert ceil_tick(100, 50, TICK) == 370
    assert ceil_tick(50, 50, TICK) == 50
    assert ceil_tick(-300, 50, TICK) == 50 - TICK


def test_three_frames_start_playback_on_the_next_tick() -> None:
    playout, starts = model()
    playout.advance(10_000)  # idle ticks
    playout.arrive([0, 1, 2, 3, 4], at=10_100)
    assert plays(playout, 12_000) == [
        (0, 10_240),
        (1, 10_560),
        (2, 10_880),
        (3, 11_200),
        (4, 11_520),
    ]
    assert starts == [RunStart(10_240, 10_100, by_count=True)]


def test_a_frame_exactly_at_a_tick_counts_for_that_tick() -> None:
    playout, _ = model()
    playout.advance(10_000)
    playout.arrive([0, 1, 2], at=10_240)
    assert plays(playout, 10_240) == [(0, 10_240)]


def test_fewer_frames_start_once_the_prebuffer_deadline_passes() -> None:
    playout, starts = model()
    playout.advance(10_000)  # the last empty tick, 9920, sets the deadline to 10 880
    playout.arrive([0], at=10_100)
    assert plays(playout, 10_879) == []
    assert plays(playout, 11_000) == [(0, 10_880)]
    assert starts == [RunStart(10_880, 10_100, by_count=False)]


def test_short_gaps_are_bridged_and_longer_ones_rebuffer() -> None:
    playout, starts = model()
    playout.advance(0)
    playout.arrive([0, 1, 2], at=1)
    assert [frame for frame, _ in plays(playout, 1_000)] == [0, 1, 2]  # ticks 320, 640, 960
    playout.arrive([3], at=1_700)  # two empty ticks (1280, 1600) before it: still playing
    assert plays(playout, 1_920) == [(3, 1_920)]
    assert playout.playing
    playout.advance(2_900)  # three empty ticks (2240, 2560, 2880): playback stops
    assert not playout.playing
    # As in AICC, the prebuffer deadline is only renewed by a tick that finds the queue empty
    # while stopped (3200 here): one frame then waits for it (3200 + 960).
    playout.arrive([4], at=3_300)
    assert plays(playout, 4_159) == []
    assert plays(playout, 4_160) == [(4, 4_160)]
    assert len(starts) == 2


def test_queue_end_and_drop() -> None:
    playout, _ = model()
    playout.advance(0)
    playout.arrive([0, 1, 2, 3, 4], at=1)
    playout.advance(320)
    assert playout.queue_end() == 320 + 5 * TICK  # four more frames, the last ends then
    assert playout.drop(keep=lambda frame: frame < 3) == [3, 4]
    assert playout.queued == 2
    assert playout.queue_end() == 320 + 3 * TICK


def test_the_grid_follows_the_phase_and_changes_only_when_idle() -> None:
    playout, _ = model(phase=100)
    playout.advance(5_000)
    playout.arrive([0, 1, 2], at=5_000)
    playout.set_phase(200)  # mid-run: deferred
    assert [start for _, start in plays(playout, 6_000)] == [5_220, 5_540, 5_860]
    assert playout.phase == 100
    playout.advance(8_000)  # the run ends, the queue empties: the new grid applies
    assert playout.phase == 200
    playout.arrive([3, 4, 5], at=8_100)
    assert plays(playout, 8_400)[0] == (3, 8_200)


def test_a_late_arrival_never_lands_before_ticks_already_run() -> None:
    playout, _ = model()
    playout.advance(10_000)
    playout.arrive([0, 1, 2], at=9_000)  # the model already ran to 10 000: from the next tick
    assert plays(playout, 10_300) == [(0, 10_240)]


def learner_runs(
    true_phase: int, transit: int, count: int, seed: int, noise: int = 16
) -> tuple[TickPhaseLearner, np.random.Generator]:
    rng = np.random.default_rng(seed)
    learner = TickPhaseLearner(TICK, max_runs=count)
    constant = 4321  # lead, echo path, transit: unknown to the learner
    for _ in range(count):
        trigger = int(rng.integers(0, 10_000_000))
        truth = ceil_tick(trigger + transit, true_phase, TICK)
        learner.observe(trigger, truth + constant + int(rng.integers(-noise, noise + 1)))
    return learner, rng


@pytest.mark.parametrize(("true_phase", "transit"), [(0, 0), (137, 30), (319, 5), (60, 300)])
def test_the_learned_phase_predicts_the_starts(true_phase: int, transit: int) -> None:
    """With 24 runs, each located to within 1 ms, the learned grid puts at least 90 % of the
    playback starts on the client's tick (a blind grid can miss up to every other one)."""
    learner, rng = learner_runs(true_phase, transit, count=24, seed=true_phase)
    phase = learner.phase()
    assert phase is not None
    triggers = [int(t) for t in rng.integers(0, 10_000_000, size=2000)]

    def hits(grid: int) -> float:
        offsets = [
            ceil_tick(t + transit, true_phase, TICK) - ceil_tick(t, grid, TICK) for t in triggers
        ]
        return max(offsets.count(value) for value in set(offsets)) / len(offsets)

    assert hits(phase) >= 0.9


def test_the_learner_waits_for_three_runs() -> None:
    learner, _ = learner_runs(137, 30, count=2, seed=1)
    assert learner.phase() is None


def test_an_outlier_does_not_move_the_phase() -> None:
    learner, rng = learner_runs(137, 30, count=10, seed=3)
    clean = learner.phase()
    learner.observe(int(rng.integers(0, 10_000_000)), int(rng.integers(0, 10_000_000)))
    assert learner.phase() == clean

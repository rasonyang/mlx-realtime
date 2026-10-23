"""The barge-in rules of D17's second line of defence, in the speech gate (vad/gate.py): while
the client is modelled as playing our audio, speech needs ``min_speech_ms_while_playing``, and
an onset inside a response's echo guard counts only from the guard's end; a segment that ends
before then is discarded."""

from __future__ import annotations

from collections.abc import Sequence

from mlx_realtime.vad.base import FRAME_MS
from mlx_realtime.vad.gate import GateEvent, GateParams, SpeechEnd, SpeechGate, SpeechStart


def run(
    speech: Sequence[tuple[int, int]],
    until_ms: int,
    *,
    playing: tuple[int, int] | None = (0, 100_000),
    guard_end_ms: int | None = None,
    while_playing: int | None = 400,
) -> list[tuple[int, GateEvent]]:
    """Frames through the gate, each tagged with the playback state at its end."""
    gate = SpeechGate(
        GateParams(
            threshold=0.5,
            min_speech_ms=150,
            silence_duration_ms=200,
            max_silence_ms=None,
            min_speech_ms_while_playing=while_playing,
        )
    )
    events: list[tuple[int, GateEvent]] = []
    for start in range(0, until_ms, FRAME_MS):
        end = start + FRAME_MS
        probability = 1.0 if any(a <= start + FRAME_MS // 2 < b for a, b in speech) else 0.0
        now_playing = playing is not None and playing[0] <= end < playing[1]
        events += [
            (end, event)
            for event in gate.on_frame(
                start,
                end,
                probability,
                playing=now_playing,
                guard_end_ms=guard_end_ms if now_playing else None,
            )
        ]
    return events


def starts(events: list[tuple[int, GateEvent]]) -> list[tuple[int, int]]:
    """(frame end, onset) of each speech start."""
    return [(at, event.onset_ms) for at, event in events if isinstance(event, SpeechStart)]


def test_without_playback_150_ms_of_speech_starts_a_turn() -> None:
    assert starts(run([(320, 1280)], 2000, playing=None)) == [(480, 320)]


def test_while_playing_speech_needs_400_ms() -> None:
    assert starts(run([(3200, 4000)], 5000)) == [(3616, 3200)]
    assert starts(run([(3200, 3500)], 5000)) == []  # 300 ms of it is not enough


def test_an_onset_inside_the_guard_counts_from_the_guard_end() -> None:
    events = run([(1024, 2624)], 4000, guard_end_ms=2000)
    assert starts(events) == [(2400, 1024)]  # 400 ms after the guard; the onset is the speech's
    assert any(isinstance(event, SpeechEnd) for _, event in events)


def test_a_segment_inside_the_guard_is_discarded() -> None:
    events = run([(1024, 2304)], 4000, guard_end_ms=2000)
    assert events == []  # never a turn: no start, no end, nothing to commit


def test_when_playback_ends_the_normal_rule_applies_from_the_onset() -> None:
    events = run([(1024, 1760)], 3000, playing=(0, 1500), guard_end_ms=2000)
    assert starts(events) == [(1504, 1024)]


def test_speech_that_began_before_playback_needs_400_ms_once_it_plays() -> None:
    assert starts(run([(960, 2000)], 3000, playing=(1000, 100_000))) == [(1376, 960)]


def test_each_rule_can_be_switched_off() -> None:
    assert starts(run([(3200, 4000)], 5000, while_playing=None)) == [(3360, 3200)]
    guard_only = run([(1024, 2624)], 4000, guard_end_ms=2000, while_playing=None)
    assert starts(guard_only) == [(2176, 1024)]  # 150 ms after the guard

from __future__ import annotations

from collections.abc import Sequence

from mlx_realtime.session.turn import (
    EvaluateTurn,
    Framer,
    SpeechStart,
    TurnAction,
    TurnEnd,
    TurnManager,
    TurnParams,
    TurnState,
)
from mlx_realtime.vad.base import FRAME_MS, FRAME_SAMPLES
from mlx_realtime.vad.scripted import ScriptedSpeech, ScriptedTurnEnd


def params(**overrides: object) -> TurnParams:
    values: dict[str, object] = {
        "threshold": 0.5,
        "min_speech_ms": 96,
        "silence_duration_ms": 200,
        "smart_turn": False,
        "smart_turn_threshold": 0.5,
        "max_silence_ms": 1000,
        "window_ms": 8000,
    }
    values.update(overrides)
    return TurnParams(**values)  # pyright: ignore[reportArgumentType]


def run(
    manager: TurnManager,
    speech: Sequence[tuple[int, int]],
    until_ms: int,
    verdicts: ScriptedTurnEnd | None = None,
) -> list[tuple[int, TurnAction]]:
    """Feed scripted frames up to ``until_ms``; answer evaluations from ``verdicts``."""
    detector = ScriptedSpeech(speech)
    actions: list[tuple[int, TurnAction]] = []
    for start in range(0, until_ms, FRAME_MS):
        probability = detector.speech_probability(b"", start_ms=start)
        pending = manager.on_frame(start, start + FRAME_MS, probability)
        while pending:
            action = pending.pop(0)
            actions.append((start + FRAME_MS, action))
            if isinstance(action, EvaluateTurn) and verdicts is not None:
                pending += manager.on_verdict(
                    action.evaluation, verdicts.end_of_turn_probability(b"")
                )
    return actions


def test_one_turn_without_smart_turn() -> None:
    manager = TurnManager(params())
    actions = run(manager, [(320, 1280)], until_ms=2000)
    # Speech confirmed after min_speech_ms (3 frames), ended after 200 ms of silence (7 frames).
    assert actions == [(416, SpeechStart(320)), (1504, TurnEnd(1280))]
    assert manager.state is TurnState.IDLE


def test_short_noise_never_starts_a_turn() -> None:
    manager = TurnManager(params())
    assert run(manager, [(320, 384)], until_ms=1000) == []  # 64 ms < min_speech_ms


def test_pauses_shorter_than_the_silence_window_stay_in_one_turn() -> None:
    manager = TurnManager(params())
    actions = run(manager, [(320, 800), (928, 1600)], until_ms=2500)
    assert [action for _, action in actions] == [SpeechStart(320), TurnEnd(1600)]


def test_smart_turn_complete_commits_after_the_silence_window() -> None:
    manager = TurnManager(params(smart_turn=True))
    actions = run(manager, [(320, 1280)], until_ms=2000, verdicts=ScriptedTurnEnd([0.9]))
    assert [action for _, action in actions] == [
        SpeechStart(320),
        EvaluateTurn(1, 320, 1280),
        TurnEnd(1280),
    ]


def test_smart_turn_incomplete_holds_until_speech_resumes() -> None:
    manager = TurnManager(params(smart_turn=True))
    verdicts = ScriptedTurnEnd([0.1, 0.9])
    actions = run(manager, [(320, 1280), (1600, 2240)], until_ms=3000, verdicts=verdicts)
    kinds = [type(action).__name__ for _, action in actions]
    # One speech start for the whole turn: the resumed speech continues it.
    assert kinds == ["SpeechStart", "EvaluateTurn", "EvaluateTurn", "TurnEnd"]
    assert actions[-1][1] == TurnEnd(2240)


def test_holding_commits_after_max_silence() -> None:
    manager = TurnManager(params(smart_turn=True, max_silence_ms=640))
    actions = run(manager, [(320, 1280)], until_ms=3000, verdicts=ScriptedTurnEnd([0.1]))
    assert actions[-1] == (1920, TurnEnd(1280))  # 640 ms after the last speech frame


def test_a_late_verdict_after_speech_resumed_is_ignored() -> None:
    manager = TurnManager(params(smart_turn=True))
    evaluation = run(manager, [(0, 640)], until_ms=864)
    assert isinstance(evaluation[-1][1], EvaluateTurn)
    stale = evaluation[-1][1].evaluation
    assert manager.on_frame(864, 896, 1.0) == []  # speech resumes while evaluating
    assert manager.state is TurnState.SPEAKING
    assert manager.on_verdict(stale, 1.0) == []


def test_hysteresis_keeps_speech_above_the_exit_threshold() -> None:
    manager = TurnManager(params())
    actions: list[TurnAction] = []
    for index, probability in enumerate([0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]):
        actions += manager.on_frame(index * 32, index * 32 + 32, probability)
    assert actions == [SpeechStart(0)]  # 0.4 is above 0.5 - 0.15, so the speech continues


def test_keep_from_tracks_padding_candidates_and_turns() -> None:
    manager = TurnManager(params())
    manager.on_frame(0, 32, 0.0)
    assert manager.keep_from_ms(300) == 32 - 300 - FRAME_MS
    manager.on_frame(32, 64, 1.0)  # a candidate onset
    assert manager.keep_from_ms(300) == 32 - 300
    manager.on_frame(64, 96, 1.0)
    manager.on_frame(96, 128, 1.0)  # confirmed
    assert manager.state is TurnState.SPEAKING
    assert manager.keep_from_ms(300) == 32 - 300


def test_reset_drops_the_turn_in_progress() -> None:
    manager = TurnManager(params())
    run(manager, [(0, 2000)], until_ms=500)
    assert manager.state is TurnState.SPEAKING
    manager.reset()
    assert manager.state is TurnState.IDLE


def test_framer_cuts_fixed_frames_and_restarts_after_a_gap() -> None:
    framer = Framer()
    frame_bytes = FRAME_SAMPLES * 2
    assert framer.push(0, b"\x00" * (frame_bytes + 10)) == [(0, b"\x00" * frame_bytes)]
    frames = framer.push(FRAME_SAMPLES + 5, b"\x01" * (frame_bytes - 10))
    assert [start for start, _ in frames] == [FRAME_SAMPLES]
    after_gap = framer.push(10_000, b"\x02" * frame_bytes)
    assert after_gap == [(10_000, b"\x02" * frame_bytes)]

"""Turn detection (ARCHITECTURE §6.2, D4): the speech gate's hysteresis and timing, and the
TurnManager's states, alone and chained as the VAD feeder and the actor chain them."""

from __future__ import annotations

from collections.abc import Sequence

from mlx_realtime.session.turn import (
    EvaluateTurn,
    Framer,
    TurnAction,
    TurnEnd,
    TurnManager,
    TurnParams,
    TurnStart,
    TurnState,
)
from mlx_realtime.vad.base import FRAME_MS, FRAME_SAMPLES
from mlx_realtime.vad.gate import (
    GateEvent,
    GateParams,
    LongSilence,
    SpeechEnd,
    SpeechGate,
    SpeechStart,
)
from mlx_realtime.vad.scripted import ScriptedSpeech


def gate(**overrides: int | float | None) -> SpeechGate:
    values: dict[str, int | float | None] = {
        "threshold": 0.5,
        "min_speech_ms": 96,
        "silence_duration_ms": 200,
        "max_silence_ms": 1000,
    }
    values.update(overrides)
    return SpeechGate(GateParams(**values))  # pyright: ignore[reportArgumentType]


def manager(*, smart_turn: bool = False, window_ms: int = 8000, lead_ms: int = 300) -> TurnManager:
    return TurnManager(TurnParams(smart_turn, 0.5, window_ms, lead_ms))


def frames(
    speech_gate: SpeechGate, speech: Sequence[tuple[int, int]], until_ms: int, from_ms: int = 0
) -> list[tuple[int, GateEvent]]:
    """Scripted frames through the gate; each event with the frame end that produced it."""
    detector = ScriptedSpeech(speech)
    events: list[tuple[int, GateEvent]] = []
    for start in range(from_ms, until_ms, FRAME_MS):
        probability = detector.probability_at(start)
        events += [
            (start + FRAME_MS, event)
            for event in speech_gate.on_frame(start, start + FRAME_MS, probability)
        ]
    return events


def run(
    speech: Sequence[tuple[int, int]],
    until_ms: int,
    *,
    verdicts: Sequence[float] = (),
    smart_turn: bool = False,
    max_silence_ms: int = 1000,
) -> list[tuple[int, TurnAction]]:
    """Gate and manager chained, evaluations answered at once from ``verdicts``."""
    speech_gate = gate(max_silence_ms=max_silence_ms if smart_turn else None)
    turns = manager(smart_turn=smart_turn)
    answers = list(verdicts)
    actions: list[tuple[int, TurnAction]] = []
    for at, event in frames(speech_gate, speech, until_ms):
        match event:
            case SpeechStart(onset_ms=onset):
                pending = turns.on_speech_start(onset)
            case SpeechEnd(end_ms=end, detected_ms=detected):
                pending = turns.on_speech_end(end, detected)
            case LongSilence():
                pending = turns.on_long_silence()
        while pending:
            action = pending.pop(0)
            actions.append((at, action))
            if isinstance(action, EvaluateTurn):
                pending += turns.on_verdict(action.evaluation, answers.pop(0) if answers else 1.0)
    return actions


def test_one_turn_without_smart_turn() -> None:
    actions = run([(320, 1280)], until_ms=2000)
    # Speech confirmed after min_speech_ms (3 frames), ended after 200 ms of silence (7 frames).
    assert actions == [(416, TurnStart(320)), (1504, TurnEnd(1280))]


def test_short_noise_never_starts_a_turn() -> None:
    assert run([(320, 384)], until_ms=1000) == []  # 64 ms < min_speech_ms


def test_pauses_shorter_than_the_silence_window_stay_in_one_turn() -> None:
    actions = run([(320, 800), (928, 1600)], until_ms=2500)
    assert [action for _, action in actions] == [TurnStart(320), TurnEnd(1600)]


def test_smart_turn_complete_commits_after_the_silence_window() -> None:
    actions = run([(320, 1280)], until_ms=2000, verdicts=[0.9], smart_turn=True)
    # The window starts prefix_padding_ms before the onset and ends where silence was detected.
    assert [action for _, action in actions] == [
        TurnStart(320),
        EvaluateTurn(1, 20, 1504),
        TurnEnd(1280),
    ]


def test_smart_turn_incomplete_holds_until_speech_resumes() -> None:
    actions = run([(320, 1280), (1600, 2240)], until_ms=3000, verdicts=[0.1, 0.9], smart_turn=True)
    kinds = [type(action).__name__ for _, action in actions]
    # One turn start for the whole turn: the resumed speech continues it.
    assert kinds == ["TurnStart", "EvaluateTurn", "EvaluateTurn", "TurnEnd"]
    assert actions[-1][1] == TurnEnd(2240)
    second = actions[2][1]
    assert isinstance(second, EvaluateTurn)
    assert (second.start_ms, second.end_ms) == (20, 2464)  # the whole turn, pause included


def test_holding_commits_after_max_silence() -> None:
    actions = run([(320, 1280)], until_ms=3000, verdicts=[0.1], smart_turn=True, max_silence_ms=640)
    assert actions[-1] == (1920, TurnEnd(1280))  # 640 ms after the last speech frame


def test_the_evaluation_window_is_at_most_window_ms() -> None:
    turns = manager(smart_turn=True, window_ms=8000)
    turns.on_speech_start(1000)
    (evaluate,) = turns.on_speech_end(20_000, 20_224)
    assert evaluate == EvaluateTurn(1, 12_224, 20_224)


def test_a_late_verdict_after_speech_resumed_is_ignored() -> None:
    turns = manager(smart_turn=True)
    turns.on_speech_start(0)
    (evaluate,) = turns.on_speech_end(640, 864)
    assert isinstance(evaluate, EvaluateTurn)
    assert turns.on_speech_start(896) == []  # speech resumes while evaluating: same turn
    assert turns.state is TurnState.SPEAKING
    assert turns.on_verdict(evaluate.evaluation, 1.0) == []


def test_long_silence_ends_an_evaluation_that_never_answered() -> None:
    turns = manager(smart_turn=True)
    turns.on_speech_start(0)
    turns.on_speech_end(640, 864)
    assert turns.on_long_silence() == [TurnEnd(640)]
    assert turns.state is TurnState.IDLE
    assert turns.on_long_silence() == []


def test_hysteresis_keeps_speech_above_the_exit_threshold() -> None:
    speech_gate = gate()
    events: list[GateEvent] = []
    for index, probability in enumerate([0.9, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]):
        events += speech_gate.on_frame(index * 32, index * 32 + 32, probability)
    assert events == [SpeechStart(0)]  # 0.4 is above 0.5 - 0.15, so the speech continues


def test_long_silence_waits_for_a_pending_onset() -> None:
    speech_gate = gate(max_silence_ms=500)
    events = frames(speech_gate, [(0, 640), (1100, 1150)], until_ms=1400)
    # 500 ms after the end (640), at 1152, a candidate onset (1088) is pending: it defers the
    # long silence, then fails (64 ms < min_speech_ms), after which the silence is reported.
    assert [event for _, event in events] == [SpeechStart(0), SpeechEnd(640, 864), LongSilence(640)]
    assert events[-1][0] == 1184


def test_keep_from_tracks_candidates_and_speech() -> None:
    speech_gate = gate()
    speech_gate.on_frame(0, 32, 0.0)
    assert speech_gate.keep_from_ms() == 32 - FRAME_MS
    speech_gate.on_frame(32, 64, 1.0)  # a candidate onset
    assert (speech_gate.keep_from_ms(), speech_gate.in_speech) == (32, True)
    speech_gate.on_frame(64, 96, 1.0)
    speech_gate.on_frame(96, 128, 1.0)  # confirmed
    assert speech_gate.keep_from_ms() == 32
    assert frames(speech_gate, [], until_ms=500, from_ms=128)[0][1] == SpeechEnd(128, 352)
    # The last frame processed ended at 512: keep one frame before it.
    assert (speech_gate.keep_from_ms(), speech_gate.in_speech) == (512 - FRAME_MS, False)


def test_reset_drops_the_turn_in_progress() -> None:
    turns = manager()
    turns.on_speech_start(0)
    assert turns.state is TurnState.SPEAKING
    turns.reset()
    assert turns.state is TurnState.IDLE
    assert turns.on_speech_end(640, 864) == []


def test_framer_cuts_fixed_frames_and_restarts_after_a_gap() -> None:
    framer = Framer()
    frame_bytes = FRAME_SAMPLES * 2
    assert framer.push(0, b"\x00" * (frame_bytes + 10)) == [(0, b"\x00" * frame_bytes)]
    pushed = framer.push(FRAME_SAMPLES + 5, b"\x01" * (frame_bytes - 10))
    assert [start for start, _ in pushed] == [FRAME_SAMPLES]
    after_gap = framer.push(10_000, b"\x02" * frame_bytes)
    assert after_gap == [(10_000, b"\x02" * frame_bytes)]

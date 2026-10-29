"""Metrics and the debug records (ARCHITECTURE §9): the stage histograms, the windowed P50/P95
computed in-process, the counters and gauges with bounded labels, and the bounded turn records of
``/debug/turns``."""

from __future__ import annotations

import numpy as np
import pytest
from prometheus_client.parser import text_string_to_metric_families

from mlx_realtime.config import ObservabilityConfig
from mlx_realtime.server.metrics import (
    DebugTurns,
    Metrics,
    WindowedQuantiles,
    quantile,
    turn_record,
)
from mlx_realtime.session.observer import TurnReport
from mlx_realtime.session.timeline import TurnTimeline


def samples(metrics: Metrics) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Every sample of the exposition, by name and labels."""
    found: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for family in text_string_to_metric_families(metrics.exposition().decode()):
        for sample in family.samples:
            found[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return found


def timeline(
    item_id: str | None = "item_1", *, offset: float = 0.0, e2e: float = 1.0
) -> TurnTimeline:
    line = TurnTimeline("turn_1", item_id)
    line.mark("speech_end", 10.0)
    line.mark("speech_stopped", 10.2)
    line.mark("turn_confirmed", 10.3)
    line.mark("asr_done", 10.5 + offset)
    line.mark("first_audio_sent", 10.0 + e2e)
    line.note("tts_queue_ms", 250)
    return line


@pytest.mark.parametrize("q", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_quantiles_interpolate_linearly_as_numpy_does(q: float) -> None:
    values = [0.3, 0.1, 0.7, 0.2, 1.5, 0.9, 0.4]
    assert quantile(values, q) == pytest.approx(float(np.quantile(values, q)))
    assert quantile([2.0], q) == 2.0
    with pytest.raises(ValueError, match="no values"):
        quantile([], q)


def test_the_window_keeps_the_last_n_turns() -> None:
    window = WindowedQuantiles(3)
    for value in (10.0, 1.0, 2.0, 3.0):
        window.observe("asr_done", value)
    assert window.quantiles("asr_done") == (2.0, pytest.approx(2.9))
    assert window.quantiles("tts_first_chunk") is None


def test_a_turn_feeds_the_stage_histograms_and_the_windowed_gauges() -> None:
    metrics = Metrics(ObservabilityConfig(quantile_window_turns=2))
    for offset, e2e in ((0.0, 1.0), (0.1, 2.0), (0.2, 3.0)):
        metrics.turn_finished(TurnReport("sess_1", timeline(offset=offset, e2e=e2e), "resp_1"))
    found = samples(metrics)
    assert found[("mlxrt_stage_seconds_count", (("stage", "asr_done"),))] == 3
    assert found[("mlxrt_stage_seconds_sum", (("stage", "speech_stopped"),))] == pytest.approx(0.6)
    assert found[("mlxrt_e2e_seconds_count", ())] == 3
    assert found[("mlxrt_e2e_seconds_bucket", (("le", "2.0"),))] == 2
    assert found[("mlxrt_tts_queue_wait_seconds_count", ())] == 3
    # The window holds the last two turns: asr_done 0.3 and 0.4 s, e2e 2 and 3 s.
    assert found[("mlxrt_stage_p50_seconds", (("stage", "asr_done"),))] == pytest.approx(0.35)
    assert found[("mlxrt_stage_p95_seconds", (("stage", "asr_done"),))] == pytest.approx(0.395)
    assert found[("mlxrt_e2e_p50_seconds", ())] == pytest.approx(2.5)
    assert found[("mlxrt_e2e_p95_seconds", ())] == pytest.approx(2.95)
    assert ("mlxrt_stage_p50_seconds", (("stage", "llm_first_token"),)) not in found


def test_a_timeline_that_answers_no_audio_turn_is_not_measured() -> None:
    metrics = Metrics(ObservabilityConfig())
    metrics.turn_finished(TurnReport("sess_1", timeline(None), "resp_1", "completed"))
    found = samples(metrics)
    assert found.get(("mlxrt_e2e_seconds_count", ()), 0) == 0
    assert metrics.windowed() == {}


def test_counters_and_gauges_carry_bounded_labels() -> None:
    depth = [3]
    metrics = Metrics(
        ObservabilityConfig(), active_sessions=lambda: 2, tts_queue_depth=lambda: depth[0]
    )
    metrics.session_started()
    metrics.session_refused("session_limit_reached")
    metrics.response_finished("completed")
    metrics.response_finished("completed")
    metrics.response_finished("cancelled")
    metrics.barge_in()
    metrics.barge_in_empty_or_echo()
    metrics.echo_turn()
    metrics.hold_released()
    metrics.hold_dropped("echo")
    metrics.hold_dropped("noise")
    metrics.hold_dropped("noise")
    metrics.provider_error("tts", "timeout")
    metrics.tts_fallback()
    metrics.set_readiness(
        {"llm": {"ready": True}, "tts": {"ready": False}, "asr": {"ready": True, "warm": False}}
    )
    found = samples(metrics)
    assert found[("mlxrt_sessions_total", ())] == 1
    assert found[("mlxrt_sessions_refused_total", (("reason", "session_limit_reached"),))] == 1
    assert found[("mlxrt_responses_total", (("status", "completed"),))] == 2
    assert found[("mlxrt_responses_total", (("status", "cancelled"),))] == 1
    assert found[("mlxrt_barge_ins_total", ())] == 1
    assert found[("mlxrt_barge_ins_empty_or_echo_total", ())] == 1
    assert found[("mlxrt_echo_turns_total", ())] == 1
    assert found[("mlxrt_held_turns_released_total", ())] == 1
    assert found[("mlxrt_held_turns_dropped_total", (("reason", "echo"),))] == 1
    assert found[("mlxrt_held_turns_dropped_total", (("reason", "noise"),))] == 2
    assert found[("mlxrt_provider_errors_total", (("backend", "tts"), ("kind", "timeout")))] == 1
    assert found[("mlxrt_tts_fallback_total", ())] == 1
    assert found[("mlxrt_backend_ready", (("backend", "llm"),))] == 1
    assert found[("mlxrt_backend_ready", (("backend", "tts"),))] == 0
    assert found[("mlxrt_backend_ready", (("backend", "asr"),))] == 0  # being warmed again
    assert found[("mlxrt_active_sessions", ())] == 2
    assert found[("mlxrt_tts_queue_depth", ())] == 3
    depth[0] = 0
    assert samples(metrics)[("mlxrt_tts_queue_depth", ())] == 0
    labels = {label for name, pairs in found for label, _ in pairs}
    assert labels <= {"stage", "status", "reason", "backend", "kind", "le"}


def test_gateways_do_not_share_counters() -> None:
    first, second = Metrics(ObservabilityConfig()), Metrics(ObservabilityConfig())
    first.session_started()
    assert samples(second)[("mlxrt_sessions_total", ())] == 0


def test_debug_records_are_kept_only_when_enabled_and_bounded() -> None:
    off = Metrics(ObservabilityConfig())
    off.turn_finished(TurnReport("sess_1", timeline()))
    assert off.debug.sessions() == []
    on = Metrics(
        ObservabilityConfig(debug_endpoints=True, debug_sessions=2, debug_turns_per_session=2)
    )
    for session in ("sess_1", "sess_2", "sess_3"):
        for _ in range(3):
            on.turn_finished(TurnReport(session, timeline(), "resp_1", "completed"))
    assert on.debug.sessions() == ["sess_2", "sess_3"]
    turns = on.debug.session("sess_3")
    assert turns is not None
    assert len(turns) == 2
    assert on.debug.session("sess_1") is None


def test_a_turn_record_lists_marks_stages_and_end_to_end_in_ms() -> None:
    record = turn_record(TurnReport("sess_1", timeline(), "resp_1", "completed"))
    assert record == {
        "turn_id": "turn_1",
        "item_id": "item_1",
        "response_id": "resp_1",
        "status": "completed",
        "marks_ms": {
            "speech_end": 0,
            "speech_stopped": 200,
            "turn_confirmed": 300,
            "asr_done": 500,
            "first_audio_sent": 1000,
        },
        "stages_ms": {"speech_stopped": 200.0, "turn_confirmed": 100.0, "asr_done": 200.0},
        "e2e_ms": 1000.0,
        "values": {"tts_queue_ms": 250},
    }
    store = DebugTurns(1, 1)
    store.add(TurnReport("sess_1", timeline()))
    assert store.session("sess_1") == [turn_record(TurnReport("sess_1", timeline()))]

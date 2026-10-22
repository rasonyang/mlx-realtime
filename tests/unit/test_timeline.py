"""The turn timeline: marks in stage order, offsets from the earliest mark, and the arrival
time of a position on the input timeline."""

from __future__ import annotations

import logging

import pytest

from mlx_realtime.session.timeline import AudioArrivals, TurnTimeline


def test_marks_are_reported_in_stage_order_from_the_earliest(
    caplog: pytest.LogCaptureFixture,
) -> None:
    timeline = TurnTimeline("turn_1", "item_1")
    timeline.mark("asr_done", 10.420)
    timeline.mark("speech_end", 10.000)
    timeline.mark("turn_confirmed", 10.290)
    timeline.mark("speech_stopped", 10.210)
    timeline.mark("speech_stopped", 11.0)  # the first mark of a name wins
    assert timeline.offsets_ms() == {
        "speech_end": 0,
        "speech_stopped": 210,
        "turn_confirmed": 290,
        "asr_done": 420,
    }
    with caplog.at_level(logging.INFO, logger="mlx_realtime.session.timeline"):
        timeline.log()
    (record,) = caplog.records
    assert record.getMessage() == "turn timeline"
    assert record.__dict__["marks_ms"]["asr_done"] == 420
    assert TurnTimeline("t", "i").offsets_ms() == {}


def test_arrivals_map_a_timeline_position_to_when_its_audio_came() -> None:
    arrivals = AudioArrivals()
    for index in range(10):
        arrivals.record(end_ms=(index + 1) * 20, at_s=100 + index * 0.02)
    assert arrivals.time_of(0) == 100
    assert arrivals.time_of(45) == pytest.approx(100.04)  # in the chunk ending at 60 ms
    assert arrivals.time_of(200) == pytest.approx(100.18)
    assert arrivals.time_of(201) is None  # not arrived yet
    arrivals.record(end_ms=300_000, at_s=500)
    assert arrivals.time_of(100) is None  # forgotten: more than two minutes back

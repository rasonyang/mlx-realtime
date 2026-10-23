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


def test_stages_run_from_the_mark_before_and_end_to_end_from_speech_end() -> None:
    timeline = TurnTimeline("turn_1", "item_1")
    for name, at in {
        "speech_end": 10.0,
        "speech_stopped": 10.2,
        "turn_confirmed": 10.3,
        "response_created": 10.3,
        "asr_done": 10.5,
        "llm_first_token": 10.8,
        "first_sentence_ready": 11.0,
        "tts_first_chunk": 11.1,
        "first_audio_sent": 11.15,
    }.items():
        timeline.mark(name, at)
    assert timeline.stages_s() == pytest.approx(
        {
            "speech_stopped": 0.2,
            "turn_confirmed": 0.1,
            "asr_done": 0.2,
            "llm_first_token": 0.3,
            "first_sentence_ready": 0.2,
            "tts_first_chunk": 0.1,
            "first_audio_sent": 0.05,
        }
    )
    assert timeline.e2e_s() == pytest.approx(1.15)


def test_the_llm_starts_at_the_later_of_the_transcript_and_the_response() -> None:
    timeline = TurnTimeline("turn_1", "item_1")
    timeline.mark("committed", 5.0)  # manual commit: no VAD marks
    timeline.mark("asr_done", 5.2)
    timeline.mark("response_created", 6.0)  # the client asked later
    timeline.mark("llm_first_token", 6.4)
    assert timeline.stages_s() == pytest.approx({"asr_done": 0.2, "llm_first_token": 0.4})
    assert timeline.e2e_s() is None  # no speech_end, no first audio


def test_a_stage_whose_marks_are_out_of_order_is_left_out() -> None:
    timeline = TurnTimeline("turn_1", None)
    timeline.mark("tts_first_chunk", 2.0)
    timeline.mark("first_audio_sent", 1.0)
    assert timeline.stages_s() == {}
    assert timeline.values == {}
    timeline.note("tts_queue_ms", 12)
    assert timeline.values == {"tts_queue_ms": 12}

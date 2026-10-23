"""Multi-session TTS fairness on the wire (M6, DECISIONS 2026-10-23), with the fake TTS at M0 S3's
capped pace (``tests/fairness.py``; ``uv run --locked python -m tests.fairness`` prints the full
before/after table for 2–8 sessions).

- Two sessions in a burst: without the mitigations a session's next segment queues behind the
  other session's and the delta gap reaches AICC's 2 s watchdog; with the audio reserve and EDF it
  stays well under, and the first audio stays within 3 s.
- Eight sessions overload one TTS stream (RTF 0.56): the gateway degrades cleanly. Every response
  ends, the ones the queue gives up on fail with ``backend_busy``, and no answered response's
  first audio misses 3 s.
"""

from __future__ import annotations

from tests.fairness import AFTER, BEFORE, FIRST_AUDIO_RULE_S, GAP_RULE_S, run_scenario


async def test_two_sessions_keep_aiccs_rules_with_the_reserve_and_edf() -> None:
    before = await run_scenario(2, BEFORE)
    after = await run_scenario(2, AFTER)
    assert (before.failed, after.failed) == (0, 0)
    assert before.max_gap_s > 1.8  # the risk DECISIONS names: close to (or past) the watchdog
    assert after.max_gap_s < GAP_RULE_S - 0.3
    assert after.max_gap_s < before.max_gap_s - 0.4
    assert after.first_audio_max_s < FIRST_AUDIO_RULE_S


async def test_eight_sessions_on_one_tts_stream_degrade_cleanly() -> None:
    result = await run_scenario(8, AFTER, turns=1, stagger_s=0.3)
    assert result.responses == 8
    assert set(result.errors) <= {"backend_busy"}
    assert result.errors.get("backend_busy", 0) == result.failed
    assert result.first_audio_over_rule == 0

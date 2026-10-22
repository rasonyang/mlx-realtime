"""AICC's client deadlines (ARCHITECTURE §2), checked on a session the test client recorded.

- ``response.created`` within 3 s of the ``response.create`` it answers;
- the first ``response.output_audio.delta`` within 3 s of ``response.created``;
- gaps between a response's audio deltas under 2 s;
- a tool-only response reaches ``response.done`` within 3 s of ``response.created``.

A missed deadline makes AICC abandon the turn silently, so every one is an assertion here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from tests.support import Event

CREATED_S = 3.0
FIRST_AUDIO_S = 3.0
GAP_S = 2.0
TOOL_ONLY_S = 3.0


@dataclass
class Deadlines:
    """The worst case of each deadline over the session, in seconds."""

    responses: int = 0
    created_s: list[float] = field(default_factory=list[float])
    first_audio_s: list[float] = field(default_factory=list[float])
    max_gap_s: float = 0.0
    audio_responses: int = 0


def check_deadlines(timeline: Sequence[tuple[float, str, Event]]) -> Deadlines:
    report = Deadlines()
    asked: deque[float] = deque()
    created: dict[str, float] = {}
    last_audio: dict[str, float] = {}
    for at, direction, event in timeline:
        kind = event.get("type")
        if direction == "client":
            if kind == "response.create":
                asked.append(at)
            continue
        if kind == "error" and event["error"]["code"] == "conversation_already_has_active_response":
            asked.popleft()  # answered, with the refusal AICC expects
        elif kind == "response.created":
            response_id = event["response"]["id"]
            created[response_id] = at
            report.responses += 1
            if asked:
                waited = at - asked.popleft()
                report.created_s.append(waited)
                assert waited <= CREATED_S, f"response.created after {waited:.3f} s"
        elif kind == "response.output_audio.delta":
            response_id = event["response_id"]
            previous = last_audio.get(response_id)
            if previous is None:
                first = at - created[response_id]
                report.first_audio_s.append(first)
                report.audio_responses += 1
                assert first <= FIRST_AUDIO_S, f"first audio {first:.3f} s after response.created"
            else:
                gap = at - previous
                report.max_gap_s = max(report.max_gap_s, gap)
                assert gap < GAP_S, f"{gap:.3f} s between audio deltas of {response_id}"
            last_audio[response_id] = at
        elif kind == "response.done":
            response = event["response"]
            output = response["output"]
            calls_only = output and all(item["type"] == "function_call" for item in output)
            if calls_only and response["status"] == "completed":
                took = at - created[response["id"]]
                assert took <= TOOL_ONLY_S, f"tool-only response.done after {took:.3f} s"
    return report

"""What a session reports for metrics and the debug endpoint (ARCHITECTURE §9).

The session layer only calls this interface; the server layer implements it (``server/metrics.py``)
and keeps every counter, histogram and debug record. Calls are synchronous and cheap: they come
from the actor's handlers, which never await.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mlx_realtime.session.timeline import TurnTimeline


@dataclass(frozen=True, slots=True)
class TurnReport:
    """One finished turn timeline: when its response ended, or with its transcript if no response
    answered it."""

    session_id: str
    timeline: TurnTimeline
    response_id: str | None = None
    status: str | None = None  # the response's final status, if a response answered the turn


class SessionObserver(Protocol):
    def turn_finished(self, report: TurnReport) -> None: ...
    def response_finished(self, status: str) -> None: ...
    def barge_in(self) -> None: ...
    def barge_in_empty_or_echo(self) -> None: ...
    def echo_turn(self) -> None: ...
    def transcription_skipped(self) -> None: ...
    def provider_error(self, backend: str, kind: str) -> None: ...
    def tts_fallback(self) -> None: ...


class NullObserver:
    """Reports nowhere (unit tests that build a session or a pipeline by hand)."""

    def turn_finished(self, report: TurnReport) -> None:
        return None

    def response_finished(self, status: str) -> None:
        return None

    def barge_in(self) -> None:
        return None

    def barge_in_empty_or_echo(self) -> None:
        return None

    def echo_turn(self) -> None:
        return None

    def transcription_skipped(self) -> None:
        return None

    def provider_error(self, backend: str, kind: str) -> None:
        return None

    def tts_fallback(self) -> None:
        return None


NULL_OBSERVER: SessionObserver = NullObserver()

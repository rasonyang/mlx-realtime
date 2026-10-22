"""The turn timeline (ARCHITECTURE §9): monotonic marks per user turn, logged in one line.

    speech_end (acoustic, from the audio timeline) · speech_stopped (the VAD fired)
    → turn_confirmed → asr_done → llm_first_token

Under manual commit, ``committed`` stands in for speech_stopped and turn_confirmed. M6 adds the
TTS marks and turns the stages into metrics.
"""

from __future__ import annotations

import logging
from collections import deque

from mlx_realtime import logs

logger = logging.getLogger(__name__)

MARKS = (
    "speech_end",
    "speech_stopped",
    "turn_confirmed",
    "committed",
    "asr_done",
    "llm_first_token",
)

_ARRIVALS_KEPT_MS = 120_000


class TurnTimeline:
    def __init__(self, turn_id: str, item_id: str) -> None:
        self.turn_id = turn_id
        self.item_id = item_id
        self._marks: dict[str, float] = {}

    def mark(self, name: str, at_s: float) -> None:
        """Record ``name`` at loop time ``at_s``; the first mark of a name wins."""
        self._marks.setdefault(name, at_s)

    def offsets_ms(self) -> dict[str, int]:
        """Every mark taken, in milliseconds after the earliest one."""
        if not self._marks:
            return {}
        origin = min(self._marks.values())
        return {
            name: round((self._marks[name] - origin) * 1000)
            for name in MARKS
            if name in self._marks
        }

    def log(self) -> None:
        with logs.bound(turn=self.turn_id, item=self.item_id):
            logger.info("turn timeline", extra={"marks_ms": self.offsets_ms()})


class AudioArrivals:
    """When each stretch of the input timeline reached the gateway (for ``speech_end``)."""

    def __init__(self) -> None:
        self._entries: deque[tuple[int, float]] = deque()  # (timeline end ms, loop time)
        self._forgotten_ms = 0  # positions before this are no longer remembered

    def record(self, end_ms: int, at_s: float) -> None:
        self._entries.append((end_ms, at_s))
        while self._entries and self._entries[0][0] < end_ms - _ARRIVALS_KEPT_MS:
            self._forgotten_ms = self._entries.popleft()[0]

    def time_of(self, position_ms: int) -> float | None:
        """The time the audio at ``position_ms`` arrived, if it is still remembered."""
        if position_ms < self._forgotten_ms:
            return None
        return next((at for end, at in self._entries if end >= position_ms), None)

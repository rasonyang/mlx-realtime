"""The turn timeline (ARCHITECTURE §9): monotonic marks per user turn, logged in one line.

    speech_end (acoustic, from the audio timeline) · speech_stopped (the VAD fired)
    → turn_confirmed → response_created → asr_done → llm_first_token
    → first_sentence_ready → tts_first_chunk → first_audio_sent

Under manual commit, ``committed`` stands in for speech_stopped and turn_confirmed. A response
that answers no user audio turn (a client ``response.create`` after text items, a verbatim
line) gets a timeline of its own, from ``response_created``. Values logged next to the marks:
``tts_queue_ms`` (the first segment's wait for a TTS slot; none when its audio was replayed),
``tts_model``, ``tts_fallback``, ``tts_cache`` (the first segment's audio was replayed from the
line cache), ``verbatim``.

Stages (M6): each stage is named after the mark that ends it and measured from the mark before
it in the chain below (``STAGES``); a stage whose marks were not both taken is left out.
``llm_first_token`` starts at the later of ``asr_done`` and ``response_created``, when the LLM
request could start. End to end is ``first_audio_sent − speech_end``.
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
    "response_created",
    "asr_done",
    "llm_first_token",
    "first_sentence_ready",
    "tts_first_chunk",
    "first_audio_sent",
)

_ARRIVALS_KEPT_MS = 120_000

STAGES: dict[str, tuple[str, ...]] = {
    # stage (its end mark): the marks it may start from; the latest one taken counts
    "speech_stopped": ("speech_end",),
    "turn_confirmed": ("speech_stopped",),
    "asr_done": ("turn_confirmed", "committed"),
    "llm_first_token": ("asr_done", "response_created"),
    "first_sentence_ready": ("llm_first_token",),
    "tts_first_chunk": ("first_sentence_ready",),
    "first_audio_sent": ("tts_first_chunk",),
}
"""The stages of a turn, in order (ARCHITECTURE §9)."""


class TurnTimeline:
    def __init__(self, turn_id: str, item_id: str | None) -> None:
        self.turn_id = turn_id
        self.item_id = item_id  # the user audio item, or None for a response's own timeline
        self._marks: dict[str, float] = {}
        self._values: dict[str, object] = {}

    def mark(self, name: str, at_s: float) -> None:
        """Record ``name`` at loop time ``at_s``; the first mark of a name wins."""
        self._marks.setdefault(name, at_s)

    def note(self, name: str, value: object) -> None:
        """A value logged next to the marks; the first one of a name wins."""
        self._values.setdefault(name, value)

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

    @property
    def marks(self) -> dict[str, float]:
        return dict(self._marks)

    @property
    def values(self) -> dict[str, object]:
        return dict(self._values)

    def stages_s(self) -> dict[str, float]:
        """Every stage whose marks were taken, in seconds (a negative one is left out)."""
        stages: dict[str, float] = {}
        for stage, starts in STAGES.items():
            end = self._marks.get(stage)
            begun = [self._marks[name] for name in starts if name in self._marks]
            if end is not None and begun and end >= max(begun):
                stages[stage] = end - max(begun)
        return stages

    def e2e_s(self) -> float | None:
        """``first_audio_sent − speech_end``: what the caller waits, when both were taken."""
        start, end = self._marks.get("speech_end"), self._marks.get("first_audio_sent")
        if start is None or end is None or end < start:
            return None
        return end - start

    def log(self, *, response_id: str | None = None) -> None:
        with logs.bound(turn=self.turn_id, item=self.item_id, response=response_id):
            logger.info("turn timeline", extra={"marks_ms": self.offsets_ms(), **self._values})


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

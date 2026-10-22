"""Deterministic detectors driven by a script instead of a model.

They make turn detection reproducible in tests, and they stand in for Silero and Smart Turn in
M1, where an empty script means no speech is ever detected.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from mlx_realtime.vad.base import FRAME_MS


class ScriptedSpeech:
    """Reports speech in the frames whose midpoint falls inside a scripted span.

    ``spans`` are ``(start_ms, end_ms)`` pairs on the session's input timeline: milliseconds of
    audio since the session's first append.
    """

    def __init__(self, spans: Sequence[tuple[int, int]] = ()) -> None:
        self._spans = tuple(spans)

    def speech_probability(self, frame: bytes, *, start_ms: int) -> float:
        midpoint = start_ms + FRAME_MS // 2
        return 1.0 if any(start <= midpoint < end for start, end in self._spans) else 0.0


class ScriptedTurnEnd:
    """Returns scripted end-of-turn probabilities in order, then ``default``."""

    def __init__(self, probabilities: Iterable[float] = (), *, default: float = 1.0) -> None:
        self._script = deque(probabilities)
        self._default = default

    def end_of_turn_probability(self, pcm: bytes) -> float:
        return self._script.popleft() if self._script else self._default

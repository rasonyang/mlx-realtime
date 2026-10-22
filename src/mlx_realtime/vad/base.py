"""Interfaces of the per-session speech detector and end-of-turn detector (ARCHITECTURE §6.2).

Both work on canonical audio (16 kHz mono PCM16). M1 calls them synchronously from the actor
with scripted implementations; M2 moves Silero and Smart Turn onto CPU executors, whose
results reach the actor as signals.
"""

from __future__ import annotations

from typing import Protocol

FRAME_SAMPLES = 512
"""Samples per speech-detector frame at 16 kHz (Silero's frame)."""

FRAME_MS = 32


class SpeechDetector(Protocol):
    """Speech probability per frame, on the session's input timeline."""

    def speech_probability(self, frame: bytes, *, start_ms: int) -> float:
        """Probability that the ``FRAME_SAMPLES`` frame starting at ``start_ms`` holds speech."""
        ...


class TurnEndDetector(Protocol):
    """Probability that the speaker has finished their turn."""

    def end_of_turn_probability(self, pcm: bytes) -> float:
        """``pcm`` is the last part (at most the configured window) of the turn's audio."""
        ...

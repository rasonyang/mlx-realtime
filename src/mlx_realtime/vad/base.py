"""Interfaces of the per-session speech detector and end-of-turn detector (ARCHITECTURE §6.2).

Both work on canonical audio (16 kHz mono PCM16) and may run their model off the event loop,
on the CPU executors of ``vad/executors.py``; scripted implementations make tests exact.
"""

from __future__ import annotations

from typing import Protocol

FRAME_SAMPLES = 512
"""Samples per speech-detector frame at 16 kHz (Silero's frame)."""

FRAME_MS = 32


class SpeechDetector(Protocol):
    """Speech probability per frame, on the session's input timeline. One call at a time."""

    async def speech_probability(self, frame: bytes, *, start_ms: int, in_speech: bool) -> float:
        """Probability that the ``FRAME_SAMPLES`` frame starting at ``start_ms`` holds speech.

        ``in_speech`` is the speech gate's state before this frame; Silero's periodic state
        reset waits until it is false.
        """
        ...

    def reset(self) -> None:
        """Forget the audio heard so far (a new, unrelated stretch of audio follows)."""
        ...


class TurnEndDetector(Protocol):
    """Probability that the speaker has finished their turn."""

    async def end_of_turn_probability(self, pcm: bytes) -> float:
        """``pcm`` is the last part (at most the configured window) of the turn's audio."""
        ...

"""Turn detection for ``server_vad``: the state machine of ARCHITECTURE §6.2 and D4.

The speech gate (``vad/gate.py``, run by the VAD feeder) turns frame probabilities into speech
starts and ends; ``TurnManager`` turns those, and the end-of-turn verdicts of Smart Turn, into
turns. It reads no clock: every position is on the input audio timeline.

    IDLE --SpeechStart--> SPEAKING                                       (TurnStart)
    SPEAKING --SpeechEnd--> EVALUATING                                   (EvaluateTurn)
    EVALUATING --SpeechStart--> SPEAKING (the pending verdict is discarded)
    EVALUATING --verdict incomplete--> HOLDING
    EVALUATING --verdict complete--> IDLE                                (TurnEnd)
    HOLDING --SpeechStart--> SPEAKING (same turn, no new TurnStart)
    EVALUATING | HOLDING --LongSilence (max_silence_ms)--> IDLE          (TurnEnd)

Without Smart Turn, SpeechEnd ends the turn at once: plain silence-based VAD.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from mlx_realtime.audio.pcm import SAMPLE_WIDTH
from mlx_realtime.vad.base import FRAME_SAMPLES


class TurnState(enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    EVALUATING = "evaluating"
    HOLDING = "holding"


@dataclass(frozen=True, slots=True)
class TurnStart:
    onset_ms: int  # start of the first speech frame


@dataclass(frozen=True, slots=True)
class EvaluateTurn:
    evaluation: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class TurnEnd:
    audio_end_ms: int  # acoustic end of speech


type TurnAction = TurnStart | EvaluateTurn | TurnEnd


@dataclass(frozen=True, slots=True)
class TurnParams:
    smart_turn: bool
    smart_turn_threshold: float
    window_ms: int  # Smart Turn sees at most this much of the turn, ending where it is evaluated
    lead_ms: int  # audio kept before the onset (prefix_padding_ms), also shown to Smart Turn


class TurnManager:
    def __init__(self, params: TurnParams) -> None:
        self._params = params
        self.state = TurnState.IDLE
        self._onset_ms = 0
        self._speech_end_ms = 0
        self._evaluation = 0

    def reset(self) -> None:
        """Forget any turn in progress; a pending verdict becomes stale."""
        self.state = TurnState.IDLE
        self._evaluation += 1

    def on_speech_start(self, onset_ms: int) -> list[TurnAction]:
        match self.state:
            case TurnState.IDLE:
                self.state = TurnState.SPEAKING
                self._onset_ms = onset_ms
                return [TurnStart(onset_ms)]
            case TurnState.EVALUATING | TurnState.HOLDING:
                self.state = TurnState.SPEAKING
                self._evaluation += 1
            case TurnState.SPEAKING:
                pass
        return []

    def on_speech_end(self, end_ms: int, detected_ms: int) -> list[TurnAction]:
        if self.state is not TurnState.SPEAKING:
            return []
        self._speech_end_ms = end_ms
        if not self._params.smart_turn:
            return self._end_turn()
        self.state = TurnState.EVALUATING
        self._evaluation += 1
        start_ms = max(self._onset_ms - self._params.lead_ms, detected_ms - self._params.window_ms)
        return [EvaluateTurn(self._evaluation, start_ms, detected_ms)]

    def on_long_silence(self) -> list[TurnAction]:
        if self.state in (TurnState.EVALUATING, TurnState.HOLDING):
            return self._end_turn()
        return []

    def on_verdict(self, evaluation: int, probability: float) -> list[TurnAction]:
        """An end-of-turn verdict; ignored unless it answers the evaluation in progress."""
        if self.state is not TurnState.EVALUATING or evaluation != self._evaluation:
            return []
        if probability > self._params.smart_turn_threshold:
            return self._end_turn()
        self.state = TurnState.HOLDING
        return []

    def _end_turn(self) -> list[TurnAction]:
        self.state = TurnState.IDLE
        self._evaluation += 1
        return [TurnEnd(self._speech_end_ms)]


class Framer:
    """Cuts the canonical stream into speech-detector frames, each with its timeline sample."""

    _FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH

    def __init__(self) -> None:
        self._pending = bytearray()
        self._start = 0

    def push(self, start_sample: int, pcm: bytes) -> list[tuple[int, bytes]]:
        if start_sample != self._start + len(self._pending) // SAMPLE_WIDTH:
            # A gap: a commit or clear ended the resampler stream. Restart framing after it.
            self._pending.clear()
            self._start = start_sample
        self._pending += pcm
        frames: list[tuple[int, bytes]] = []
        while len(self._pending) >= self._FRAME_BYTES:
            frames.append((self._start, bytes(self._pending[: self._FRAME_BYTES])))
            del self._pending[: self._FRAME_BYTES]
            self._start += FRAME_SAMPLES
        return frames

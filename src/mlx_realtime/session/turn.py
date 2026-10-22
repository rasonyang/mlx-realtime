"""Turn detection for ``server_vad``: the state machine of ARCHITECTURE §6.2.

``TurnManager`` is fed speech probabilities per frame and end-of-turn verdicts, and answers
with actions. It reads no clock: every duration is measured on the input audio timeline, so
the same audio always yields the same turns.

    IDLE --speech >= min_speech_ms--> SPEAKING                           (SpeechStart)
    SPEAKING --silence >= silence_duration_ms--> EVALUATING              (EvaluateTurn)
    EVALUATING --speech resumes--> SPEAKING (the verdict is discarded)
    EVALUATING --verdict incomplete--> HOLDING
    EVALUATING --verdict complete--> IDLE                                (TurnEnd)
    HOLDING --speech resumes--> SPEAKING (same turn, no new SpeechStart)
    HOLDING --silence >= max_silence_ms--> IDLE                          (TurnEnd)

Without Smart Turn, EVALUATING is skipped: silence ends the turn.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from mlx_realtime.audio.pcm import SAMPLE_WIDTH
from mlx_realtime.vad.base import FRAME_MS, FRAME_SAMPLES

_HYSTERESIS = 0.15  # Silero's own: speech ends below threshold - 0.15


class TurnState(enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    EVALUATING = "evaluating"
    HOLDING = "holding"


@dataclass(frozen=True, slots=True)
class SpeechStart:
    onset_ms: int  # start of the first speech frame


@dataclass(frozen=True, slots=True)
class EvaluateTurn:
    evaluation: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class TurnEnd:
    audio_end_ms: int  # acoustic end of speech


type TurnAction = SpeechStart | EvaluateTurn | TurnEnd


@dataclass(frozen=True, slots=True)
class TurnParams:
    threshold: float
    min_speech_ms: int
    silence_duration_ms: int
    smart_turn: bool
    smart_turn_threshold: float
    max_silence_ms: int
    window_ms: int


class TurnManager:
    def __init__(self, params: TurnParams) -> None:
        self._params = params
        self._exit_threshold = max(params.threshold - _HYSTERESIS, 0.0)
        self.state = TurnState.IDLE
        self._candidate_ms: int | None = None
        self._turn_start_ms = 0
        self._speech_end_ms = 0
        self._frame_end_ms = 0
        self._evaluation = 0

    def reset(self) -> None:
        """Forget any turn in progress; a pending verdict becomes stale."""
        self.state = TurnState.IDLE
        self._candidate_ms = None
        self._evaluation += 1

    def on_frame(self, start_ms: int, end_ms: int, probability: float) -> list[TurnAction]:
        self._frame_end_ms = end_ms
        speech = probability >= self._params.threshold
        match self.state:
            case TurnState.IDLE:
                return self._idle(start_ms, end_ms, probability)
            case TurnState.SPEAKING:
                if probability >= self._exit_threshold:
                    self._speech_end_ms = end_ms
                elif end_ms - self._speech_end_ms >= self._params.silence_duration_ms:
                    return self._evaluate()
            case TurnState.EVALUATING | TurnState.HOLDING:
                if speech:
                    self.state = TurnState.SPEAKING
                    self._speech_end_ms = end_ms
                    self._evaluation += 1
                elif end_ms - self._speech_end_ms >= self._params.max_silence_ms:
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

    def keep_from_ms(self, prefix_padding_ms: int) -> int:
        """The earliest input position the buffer still needs."""
        if self.state is not TurnState.IDLE:
            return self._turn_start_ms - prefix_padding_ms
        if self._candidate_ms is not None:
            return self._candidate_ms - prefix_padding_ms
        return self._frame_end_ms - prefix_padding_ms - FRAME_MS

    def _idle(self, start_ms: int, end_ms: int, probability: float) -> list[TurnAction]:
        if probability >= self._params.threshold or (
            self._candidate_ms is not None and probability >= self._exit_threshold
        ):
            if self._candidate_ms is None:
                self._candidate_ms = start_ms
            if end_ms - self._candidate_ms >= self._params.min_speech_ms:
                self.state = TurnState.SPEAKING
                self._turn_start_ms = self._candidate_ms
                self._speech_end_ms = end_ms
                self._candidate_ms = None
                return [SpeechStart(self._turn_start_ms)]
        else:
            self._candidate_ms = None
        return []

    def _evaluate(self) -> list[TurnAction]:
        if not self._params.smart_turn:
            return self._end_turn()
        self.state = TurnState.EVALUATING
        self._evaluation += 1
        start_ms = max(self._turn_start_ms, self._speech_end_ms - self._params.window_ms)
        return [EvaluateTurn(self._evaluation, start_ms, self._speech_end_ms)]

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

"""The speech gate: hysteresis over per-frame speech probabilities (ARCHITECTURE §6.2).

Speech enters at ``threshold`` and leaves below ``threshold - 0.15`` (Silero's own hysteresis).
It must last ``min_speech_ms`` before it counts, so clicks and short noises never start a turn.
The gate reads no clock: every duration is measured on the input audio timeline, so the same
audio always yields the same events.

    QUIET --speech >= min_speech_ms--> SPEAKING                       SpeechStart(onset)
    SPEAKING --silence >= silence_duration_ms--> QUIET                SpeechEnd(end, detected)
    QUIET --silence since that end >= max_silence_ms--> QUIET          LongSilence(end)
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_realtime.vad.base import FRAME_MS

HYSTERESIS = 0.15


@dataclass(frozen=True, slots=True)
class GateParams:
    threshold: float
    min_speech_ms: int
    silence_duration_ms: int
    max_silence_ms: int | None  # None: no LongSilence events


@dataclass(frozen=True, slots=True)
class SpeechStart:
    onset_ms: int  # start of the first speech frame


@dataclass(frozen=True, slots=True)
class SpeechEnd:
    end_ms: int  # end of the last speech frame: the acoustic end of speech
    detected_ms: int  # where the silence window closed


@dataclass(frozen=True, slots=True)
class LongSilence:
    end_ms: int  # the end of speech the silence is measured from


type GateEvent = SpeechStart | SpeechEnd | LongSilence


class SpeechGate:
    def __init__(self, params: GateParams) -> None:
        self._params = params
        self._exit = max(params.threshold - HYSTERESIS, 0.0)
        self._speaking = False
        self._onset_ms = 0
        self._candidate_ms: int | None = None
        self._speech_end_ms = 0
        self._silence_from_ms: int | None = None  # pending LongSilence
        self._frame_end_ms = 0

    @property
    def in_speech(self) -> bool:
        """Speech confirmed or starting: pipecat's VAD states other than QUIET."""
        return self._speaking or self._candidate_ms is not None

    def keep_from_ms(self) -> int:
        """The earliest input position a turn the gate has not ended yet can start at."""
        if self._speaking:
            return self._onset_ms
        if self._candidate_ms is not None:
            return self._candidate_ms
        return self._frame_end_ms - FRAME_MS

    def on_frame(self, start_ms: int, end_ms: int, probability: float) -> list[GateEvent]:
        self._frame_end_ms = end_ms
        if self._speaking:
            return self._while_speaking(end_ms, probability)
        events: list[GateEvent] = []
        if probability >= self._params.threshold or (
            self._candidate_ms is not None and probability >= self._exit
        ):
            if self._candidate_ms is None:
                self._candidate_ms = start_ms
            if end_ms - self._candidate_ms >= self._params.min_speech_ms:
                self._speaking = True
                self._onset_ms = self._candidate_ms
                self._speech_end_ms = end_ms
                self._candidate_ms = None
                self._silence_from_ms = None
                events.append(SpeechStart(self._onset_ms))
            return events
        self._candidate_ms = None
        return self._check_long_silence(end_ms)

    def _while_speaking(self, end_ms: int, probability: float) -> list[GateEvent]:
        if probability >= self._exit:
            self._speech_end_ms = end_ms
            return []
        if end_ms - self._speech_end_ms < self._params.silence_duration_ms:
            return []
        self._speaking = False
        self._silence_from_ms = self._speech_end_ms
        return [SpeechEnd(self._speech_end_ms, end_ms), *self._check_long_silence(end_ms)]

    def _check_long_silence(self, end_ms: int) -> list[GateEvent]:
        since, limit = self._silence_from_ms, self._params.max_silence_ms
        if since is None or limit is None or end_ms - since < limit:
            return []
        self._silence_from_ms = None
        return [LongSilence(since)]

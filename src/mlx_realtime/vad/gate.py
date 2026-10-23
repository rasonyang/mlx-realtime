"""The speech gate: hysteresis over per-frame speech probabilities (ARCHITECTURE §6.2).

Speech enters at ``threshold`` and leaves below ``threshold - 0.15`` (Silero's own hysteresis).
It must last ``min_speech_ms`` before it counts, so clicks and short noises never start a turn.
The gate reads no clock: every duration is measured on the input audio timeline, so the same
audio always yields the same events.

While the client is modelled as playing our audio (D17, the second line of defence against
echo; the caller passes the state per frame):

- speech must last ``min_speech_ms_while_playing`` instead;
- speech that begins inside a response's echo guard (``guard_end_ms``) is counted only from the
  end of the guard, so an onset there never starts a turn by itself;
- once playback has ended, the normal rule applies from the speech's own onset.

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
    min_speech_ms_while_playing: int | None = None  # None: min_speech_ms throughout


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
        self._candidate_guard_end: int | None = None  # the guard the candidate began inside
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

    def on_frame(
        self,
        start_ms: int,
        end_ms: int,
        probability: float,
        *,
        playing: bool = False,
        guard_end_ms: int | None = None,
    ) -> list[GateEvent]:
        """One frame's probability; ``playing``: the client is modelled as playing our audio
        at the frame's end, ``guard_end_ms``: where that response's echo guard ends."""
        self._frame_end_ms = end_ms
        if self._speaking:
            return self._while_speaking(end_ms, probability)
        events: list[GateEvent] = []
        if probability >= self._params.threshold or (
            self._candidate_ms is not None and probability >= self._exit
        ):
            if self._candidate_ms is None:
                self._candidate_ms = start_ms
                inside = playing and guard_end_ms is not None and guard_end_ms > start_ms
                self._candidate_guard_end = guard_end_ms if inside else None
            if self._lasted(self._candidate_ms, end_ms, playing):
                self._speaking = True
                self._onset_ms = self._candidate_ms
                self._speech_end_ms = end_ms
                self._candidate_ms = None
                self._silence_from_ms = None
                events.append(SpeechStart(self._onset_ms))
            return events
        self._candidate_ms = None
        return self._check_long_silence(end_ms)

    def _lasted(self, onset_ms: int, end_ms: int, playing: bool) -> bool:
        """Whether speech from ``onset_ms`` has lasted long enough to count at ``end_ms``."""
        params = self._params
        if not playing:
            return end_ms - onset_ms >= params.min_speech_ms
        guard_end = self._candidate_guard_end
        if guard_end is not None:
            if end_ms <= guard_end:
                return False
            onset_ms = max(onset_ms, guard_end)
        needed = params.min_speech_ms_while_playing
        return end_ms - onset_ms >= (params.min_speech_ms if needed is None else needed)

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

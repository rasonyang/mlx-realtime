"""Echo cancellation for one session (ARCHITECTURE §4.3, §6.5; D17, G7).

Neither AICC nor FreeSWITCH cancels echo, so the caller's audio carries our own speech back.
Three parts handle it:

- ``CaptureClock`` maps the session's input timeline to loop time, from when the audio arrived:
  the client streams its capture in real time, so the earliest arrivals pin the offset.
- ``EchoReference`` is what the client is modelled to play: every audio delta as the writer
  hands it to the socket, decoded (G.711) and resampled to 16 kHz, framed in 20 ms and played
  on the AICC playout model (``audio/playout.py``). From it come the canceller's reference for
  any stretch of the input (placed ``lead`` early, so it precedes its echo, and moved by the
  GCC-PHAT guard), whether the client was playing at a point of the input (the barge-in rules),
  and which assistant audio played around a user turn (echo-turn suppression). A barge-in or
  cancel flushes the frames the client never played; a client truncate drops the rest of that
  item and re-anchors the model's timing.
- ``EchoStage`` runs the cancellers on the VAD feeder's path: whole 10 ms frames, each fed in
  lock step with the reference frame for the same input positions (zeros when nothing plays, so
  the delay is carried in the content), on the canceller executor, in order. While a response
  may still add audio, input is held back until its reference can no longer change.

G7 dual path: WebRTC AEC3's output goes to speech and end-of-turn detection, and, with
``aec.asr_input: speex``, a SpeexDSP linear canceller runs on the same frames for the ASR. AEC3
absorbs the residual delay itself; the linear filter needs the echo inside its 64 ms, so the
GCC-PHAT estimate that guards AEC3 also places Speex's reference, the echo ``target`` behind it,
and moves it again only when a confirmed estimate leaves [0, ``keep``] (M0 S11). Until the first
confirmed estimate the ASR path passes the input through as received: the first ~2.5 s of a
call's first answer are not cancelled for the ASR.

On a line without echo, AEC3's suppressor takes a caller who talks over our audio for echo and
silences them (measured in M4: up to 45 dB, for seconds). So once the delay estimator has watched
``bypass_after`` of playing reference with no echo of it in the input, the canceller is fed a
silent reference (it then passes the input through); the first echo the estimator finds turns
it back on (AEC3 re-converges within about 0.5 s). Leaving the bypass takes more than the
estimator's own confirmation (two valid windows agreeing): its windows overlap by 1.5 s, so a
chance peak between our speech and a caller talking over it confirms itself about every 10 s of
such double talk, turning the canceller back on mid barge-in. Echo must show in
``_RESUME_WINDOWS`` consecutive valid windows that agree, each with a PSR of at least
``_RESUME_PSR``: a real echo resumes one window (0.5 s) later, and false resumes fell from 29 to
2 in 4.5 minutes of FLEURS double talk (spikes/realstack/echo_bypass.py, 2026-10-23). The ASR
path's linear canceller never takes the bypass: it has no suppressor to silence the caller.

Times and positions are integer sample counts at 16 kHz: "loop samples" for the event loop's
clock, input-timeline samples for the capture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mlx_realtime.audio.aec import (
    FRAME_SAMPLES,
    EchoEngine,
    EngineName,
    LowPass,
    Pcm,
    SpeexLinear,
    make_engine,
)
from mlx_realtime.audio.codec import InputDecoder
from mlx_realtime.audio.delay import HOP, WINDOW, DelayEstimate, DelayEstimator
from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH
from mlx_realtime.audio.playout import (
    Played,
    PlayoutModel,
    PlayoutParams,
    RunStart,
    TickPhaseLearner,
)
from mlx_realtime.audio.resample import Quality
from mlx_realtime.config import AecConfig, BargeInConfig, SpeexConfig
from mlx_realtime.session.clock import Clock
from mlx_realtime.session.echo_capture import EchoCapture

logger = logging.getLogger(__name__)

RATE = CANONICAL_RATE
_FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH
_CAPTURE_WINDOW = 10 * RATE  # the capture clock's sliding window
_MAX_ANCHOR = RATE // 2  # re-anchoring moves the model by at most 500 ms
_HISTORY = 60 * RATE  # how long played audio is remembered for echo-turn matching
# Played frames kept behind the reference already rendered: AEC3's references, and a held
# turn's correlation check (up to its 4 s window and the 1 s lag range behind its end).
_RENDER_MARGIN = 6 * RATE
_ACTIVE_REFERENCE = 10 ** (-50 / 10) * 32768.0**2  # a reference frame above -50 dBFS
_ERLE_WINDOW_FRAMES = 50  # 0.5 s of frames with an active reference
HOP_SAMPLES = HOP * RATE // 8000  # the delay estimator's hop at 16 kHz
WINDOW_SAMPLES = WINDOW * RATE // 8000  # and its window
_RESUME_WINDOWS = 3  # consecutive agreeing valid windows that end the bypass ...
_RESUME_PSR = 10.0  # ... each with at least this peak-to-sidelobe ratio
_RESUME_AGREE = 2 * RATE // 1000  # 2 ms, the estimator's own agreement


def samples(seconds: float) -> int:
    return round(seconds * RATE)


def ms_samples(ms: int) -> int:
    return ms * RATE // 1000


@dataclass(frozen=True, slots=True)
class EchoParams:
    playout: PlayoutParams
    lead: int  # the reference precedes the modelled playout by this much
    tail: int  # how long after the modelled playout its echo can still arrive
    guard: int  # echo guard after a response's playback starts (0: off)
    keep_min: int  # the residual delay the guard leaves alone ...
    keep_max: int  # ... up to here
    learn_tick_phase: bool
    delay_guard: bool
    bypass_after: int | None  # reference played without echo before the canceller is bypassed
    speex: bool = False  # the ASR path's linear canceller is aligned (aec.asr_input: speex)
    speex_target: int = 0  # where its reference puts the echo ...
    speex_keep: int = 0  # ... and how far behind it may drift before it is moved again
    aicc_guard: int = 800 * RATE // 1000  # AICC ignores a speech barge-in this soon after audible
    aicc_gap_slack: int = 40 * RATE // 1000  # a frame this far past the queue end is a new start

    @classmethod
    def of(cls, aec: AecConfig, barge_in: BargeInConfig) -> EchoParams:
        tick = ms_samples(aec.client_tick_ms)
        return cls(
            playout=PlayoutParams(
                tick=tick,
                prebuffer_frames=max(-(-aec.client_prebuffer_ms // aec.client_tick_ms), 1),
                prebuffer=ms_samples(aec.client_prebuffer_ms),
                underrun_ticks=aec.client_underrun_ticks,
            ),
            lead=ms_samples(aec.reference_lead_ms),
            tail=ms_samples(aec.max_residual_delay_ms),
            guard=ms_samples(barge_in.echo_guard_ms),
            keep_min=ms_samples(aec.min_residual_delay_ms),
            keep_max=ms_samples(aec.max_residual_delay_ms),
            learn_tick_phase=aec.learn_tick_phase and aec.delay_guard,
            delay_guard=aec.delay_guard,
            bypass_after=(
                ms_samples(aec.bypass_without_echo_ms)
                if aec.delay_guard and aec.bypass_without_echo_ms is not None
                else None
            ),
            speex=aec.enabled and aec.delay_guard and aec.asr_input == "speex",
            speex_target=ms_samples(aec.speex.target_delay_ms),
            speex_keep=ms_samples(aec.speex.max_delay_ms),
        )


class CaptureClock:
    """Maps the input timeline to loop samples: ``offset`` is the smallest
    ``arrival - position`` seen over the last 10 s (the least delayed arrivals), which tracks
    clock drift and ignores jitter."""

    def __init__(self, window: int = _CAPTURE_WINDOW) -> None:
        self._window = window
        self._candidates: deque[tuple[int, int]] = deque()  # (at, offset), offsets increasing

    @property
    def offset(self) -> int | None:
        return self._candidates[0][1] if self._candidates else None

    def observe(self, end_sample: int, at: int) -> None:
        """Input up to ``end_sample`` had arrived at loop sample ``at``."""
        value = at - end_sample
        while self._candidates and self._candidates[-1][1] >= value:
            self._candidates.pop()
        self._candidates.append((at, value))
        while self._candidates[0][0] < at - self._window:
            self._candidates.popleft()


class _ItemAudio:
    """One assistant item's output audio as sent, and how the client played it."""

    def __init__(
        self, item_id: str, generation: int, audio_format: AudioFormat, g711_quality: Quality
    ) -> None:
        self.item_id = item_id
        self.generation = generation
        self._decoder = InputDecoder(audio_format, g711_quality)  # as the caller's audio comes in
        self._sample_width = audio_format.sample_width
        self._per_frame = audio_format.rate // 50  # wire samples in one 20 ms client frame
        self._reference = bytearray()  # 16 kHz, as resampled so far
        self._wire = 0  # wire samples sent
        self.frames = 0  # 20 ms frames handed to the client model
        self.complete = False
        self.plays: list[int] = []  # play start of frame i (frames play in order)
        self.queued = 0  # frames waiting in the client model's queue
        self.ended = False  # its playback end is in the capture's events

    @property
    def last_end(self) -> int | None:
        return None if not self.plays else self.plays[-1]

    def add(self, audio: bytes) -> int:
        """Audio sent; returns how many whole 20 ms frames it completed."""
        self._reference += self._decoder.decode(audio)
        self._wire += len(audio) // self._sample_width
        whole = self._wire // self._per_frame
        added, self.frames = whole - self.frames, whole
        return added

    def finish(self) -> int:
        """The item's audio is complete: the client pads the last partial frame (its framer
        does at a response's end). Returns the frames added."""
        if self.complete:
            return 0
        self.complete = True
        self._reference += self._decoder.flush()
        if self._wire % self._per_frame:
            self.frames += 1
            return 1
        return 0

    def frame(self, index: int, tick: int) -> npt.NDArray[np.int16]:
        """The 16 kHz reference audio of frame ``index`` (zeros where not resampled yet)."""
        out = np.zeros(tick, dtype=np.int16)
        start = index * tick * SAMPLE_WIDTH
        chunk = self._reference[start : start + tick * SAMPLE_WIDTH]
        out[: len(chunk) // SAMPLE_WIDTH] = np.frombuffer(chunk, dtype="<i2")
        return out


@dataclass(frozen=True, slots=True)
class _Frame:
    item: _ItemAudio
    index: int


@dataclass(slots=True)
class _Run:
    """One playback run of the client model, for tick-phase learning."""

    start: int
    trigger: int
    by_count: bool
    shift: int
    last_end: int
    broken: bool = False  # an underrun gap: the alignment inside is not the onset's
    observed: bool = False


@dataclass(frozen=True, slots=True)
class Playback:
    """The client model at one input position, for the barge-in rules."""

    playing: bool
    guard_end: int | None  # input sample where the guard of the playing response ends


@dataclass(frozen=True, slots=True)
class PlayedSpan:
    item_id: str
    start_ms: int  # on the item's audio timeline
    end_ms: int


@dataclass(slots=True)
class _Erle:
    """ERLE over 0.5 s windows of frames with an active reference: the median, so windows
    where the caller talked over us weigh little."""

    _window: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # frames, in, out
    _values: deque[float] = field(default_factory=lambda: deque[float](maxlen=1200))

    def add(self, frames: int, capture_energy: float, output_energy: float) -> None:
        window = self._window
        window[0] += frames
        window[1] += capture_energy
        window[2] += output_energy
        if window[0] >= _ERLE_WINDOW_FRAMES:
            if window[1] > 0:
                self._values.append(10 * float(np.log10(window[1] / max(window[2], 1.0))))
            self._window = [0.0, 0.0, 0.0]

    def median_db(self) -> float | None:
        if not self._values:
            return None
        return round(float(np.median(np.asarray(self._values))), 1)


@dataclass(slots=True)
class EchoStats:
    """The canceller's work in one session. ERLE is the median over 0.5 s windows in which the
    reference was active, so windows where the caller talked over us weigh little."""

    frames: int = 0
    engine_ns: int = 0
    realignments: int = 0
    estimates: int = 0
    anchors: int = 0
    bypasses: int = 0
    erle: _Erle = field(default_factory=lambda: _Erle())
    # The ASR path's linear canceller (Speex): its CPU (with the pre-filter), the input sample
    # where its reference was first aligned, its re-alignments, ERLE, and filter resets.
    speex_ns: int = 0
    speex_aligned_at: int | None = None
    speex_realignments: int = 0
    speex_erle: _Erle = field(default_factory=lambda: _Erle())
    speex_resets: int = 0

    def add(self, frames: int, capture_energy: float, output_energy: float) -> None:
        """``frames`` processed with an active reference, and their energy in and out."""
        self.erle.add(frames, capture_energy, output_energy)

    def erle_db(self) -> float | None:
        return self.erle.median_db()

    def cpu_ms_per_audio_s(self) -> float | None:
        return self._per_audio_s(self.engine_ns)

    def speex_cpu_ms_per_audio_s(self) -> float | None:
        return self._per_audio_s(self.speex_ns)

    def _per_audio_s(self, ns: int) -> float | None:
        audio_s = self.frames * FRAME_SAMPLES / RATE
        return None if audio_s == 0 else round(ns / 1e6 / audio_s, 2)


class EchoReference:
    """The client's modelled playout of this session's output (see the module notes).

    Fed on the event loop by the actor (responses, cancels, truncates, input arrivals) and by
    the writer through the output ledger (audio queued, sent, dropped); read by the VAD feeder.
    """

    def __init__(self, params: EchoParams, clock: Clock, g711_quality: Quality = "HQ") -> None:
        """``g711_quality``: the 8 → 16 kHz resampling of G.711 output, as its echo comes in."""
        self.params = params
        self._clock = clock
        self._g711_quality: Quality = g711_quality
        self._model = PlayoutModel[_Frame](params.playout, on_run_start=self._run_started)
        self._capture = CaptureClock()
        self._items: dict[str, _ItemAudio] = {}
        self._played: deque[Played[_Frame]] = deque()  # for rendering, by start
        self._runs: deque[_Run] = deque(maxlen=64)
        self._learner = TickPhaseLearner(params.playout.tick) if params.learn_tick_phase else None
        self._shift = 0  # the reference is delayed by this much (the guard's re-alignment)
        self._speex_shift: int | None = None  # the ASR path's reference: None until aligned
        self._speex_hold = 0  # input sample before which estimates may predate a re-anchoring
        self._anchor = 0  # added to send times (re-anchored by client truncates)
        # AICC's audibleSince/audibleUntil (session.go queueFrame), on the send timeline: when the
        # client started hearing us after the last silence, and when the audio sent so far ends.
        self._audible_since: int | None = None
        self._audible_until = 0
        self._pending: dict[int, int] = {}  # generation -> deltas queued, not sent or dropped
        self._ended: set[int] = set()  # generations whose response is over
        self._active = False  # a response is in progress
        self._no_echo = 0  # reference played with no echo found since the last echo
        self.echo_absent = False  # the canceller is bypassed (see the module notes)
        self._resume: list[int] = []  # while bypassed: the lags of the agreeing run of windows
        self.stats = EchoStats()
        self.on_change: Callable[[], None] | None = None  # the feeder's wakeup
        self.capture: EchoCapture | None = None  # aec.capture_dir: the events go there too

    # Capture events (aec.capture_dir)

    def note(self, kind: str, at: int | None = None, **fields: object) -> None:
        """An event for the capture, at loop sample ``at`` (now): its time is the input position
        the client's audio had reached then."""
        capture = self.capture
        if capture is None or not capture.active:
            return
        offset = self._capture.offset
        when = self._now() if at is None else at
        capture.event(None if offset is None else (when - offset) * 1000 / RATE, kind, **fields)

    def note_input(self, kind: str, end: int, **fields: object) -> None:
        """The same at input sample ``end``."""
        if self.capture is not None and self.capture.active:
            self.capture.event(end * 1000 / RATE, kind, **fields)

    def close_capture(self, summary: dict[str, object]) -> None:
        if self.capture is not None:
            self.capture.close(summary)

    def _note_play(self, item: _ItemAudio, entry: Played[_Frame]) -> None:
        """The first frame of an item played (the model's time), where its reference sits."""
        if self.capture is None or not self.capture.active:
            return
        tick = self.params.playout.tick
        shift = self._shift
        self.note(
            "play_start",
            entry.start,
            item=item.item_id,
            generation=item.generation,
            ref_t=None
            if self._capture.offset is None
            else round(
                (entry.start - self._capture.offset - self.params.lead + shift) * 1000 / RATE, 1
            ),
            shift_ms=shift * 1000 / RATE,
            tick_ms=tick * 1000 / RATE,
        )

    def _note_end(self, item: _ItemAudio) -> None:
        """An item that is complete and has played its queue out ended where its last frame did."""
        if item.ended or not item.plays or not item.complete or item.queued > 0:
            return
        item.ended = True
        tick = self.params.playout.tick
        self.note(
            "play_end",
            item.plays[-1] + tick,
            item=item.item_id,
            generation=item.generation,
            frames=len(item.plays),
        )

    # Inputs

    def capture_arrived(self, end_sample: int) -> None:
        self._capture.observe(end_sample, self._now())

    def response_started(self) -> None:
        self._active = True

    def response_finished(self, generation: int) -> None:
        self._active = False
        self._ended.add(generation)
        self._finish_if_sent(generation)
        self._changed()

    def audio_queued(self, generation: int) -> None:
        self._pending[generation] = self._pending.get(generation, 0) + 1

    def audio_sent(
        self, item_id: str, generation: int, audio: bytes, audio_format: AudioFormat = PCM
    ) -> None:
        """The writer handed an audio delta (in ``audio_format``) to the socket."""
        item = self._items.get(item_id)
        if item is None:
            item = _ItemAudio(item_id, generation, audio_format, self._g711_quality)
            self._items[item_id] = item
        first = item.frames
        added = item.add(audio)
        self._arrive(item, first, added)
        self._settle(generation)

    def audio_dropped(self, generation: int) -> None:
        """The writer discarded a delta of a cancelled response."""
        self._settle(generation)

    def flush(self, generation: int, *, speech: bool = False) -> None:
        """A barge-in or cancel of ``generation``: the client's ClearTx empties its whole queue,
        a previous response's unplayed tail included (§6.5 step 4); nothing more of this
        generation will come.

        ``speech``: our own speech barge-in. AICC ignores one within its guard of the audio
        becoming audible (session.go ``bargeIn``) and then clears nothing, so the queued frames
        play on in the model. A cancel AICC initiated (its interrupt, a keypress) always clears.
        """
        self._advance(self._now())
        kept = speech and self._guarded()
        if not kept:
            self._clear()
        for item in self._items.values():
            if item.generation == generation:
                item.complete = True
        self._ended.add(generation)
        self._pending.pop(generation, None)
        self.note("flush", generation=generation, speech=speech, kept=kept)
        for item in self._items.values():
            if item.generation == generation:
                self._note_end(item)
        self._changed()

    def _guarded(self) -> bool:
        """AICC would ignore a speech barge-in now: the client started hearing us too recently."""
        since = self._audible_since
        return since is not None and self._now() - since < self.params.aicc_guard

    def _clear(self) -> None:
        """AICC's ClearTx: every queued frame goes, and its items are complete."""
        for frame in self._model.drop():
            frame.item.queued -= 1
            if not frame.item.complete:
                frame.item.complete = True
                self._note_end(frame.item)
        self._audible_since = None

    def truncated(self, item_id: str, audio_end_ms: int) -> None:
        """The client reports ``audio_end_ms`` of the item played (its played frames): it has
        stopped playing the item, and the model's timing is corrected by the difference."""
        item = self._items.get(item_id)
        if item is None:
            return
        self._advance(self._now())
        tick_ms = self.params.playout.tick * 1000 // RATE
        heard = audio_end_ms // tick_ms
        if item.plays:
            difference = heard - len(item.plays)
            if abs(difference) >= 2:  # beyond transit and rounding: the model is off
                anchor = self._anchor - difference * self.params.playout.tick
                anchor = max(-_MAX_ANCHOR, min(_MAX_ANCHOR, anchor))
                if self._speex_shift is not None:
                    # Speex's reference stands where GCC-PHAT measured the echo: the model's
                    # own correction must not move it (it would leave its narrow keep range),
                    # and estimates of audio placed before the correction no longer apply.
                    self._speex_shift -= anchor - self._anchor
                    self._speex_hold = self._now() - (self._capture.offset or 0) + WINDOW_SAMPLES
                self._anchor = anchor
                self.stats.anchors += 1
                self.note(
                    "reanchor",
                    item=item_id,
                    frames_off=difference,
                    anchor_ms=self.anchor_ms,
                    speex_shift_ms=self.speex_shift_ms,
                )
                logger.info(
                    "echo reference re-anchored",
                    extra={"item": item_id, "frames_off": difference, "anchor_ms": self.anchor_ms},
                )
        self._clear()  # the client emptied its whole queue, not only this item's frames
        for entry in [entry for entry in self._played if entry.handle.item is item]:
            if entry.handle.index >= heard:
                self._played.remove(entry)  # never played: no reference for it
        del item.plays[heard:]
        item.complete = True
        self.note("truncate", item=item_id, audio_end_ms=audio_end_ms, played_frames=heard)
        self._note_end(item)
        self._changed()

    def delay_estimate(self, end: int, estimate: DelayEstimate) -> bool:
        """A GCC-PHAT window, evaluated while the reference played, that ended at input sample
        ``end``. Returns whether the reference was re-aligned (the estimator must then start
        over)."""
        params = self.params
        self.note_input(
            "gcc",
            end,
            lag_ms=round(estimate.lag * 1000 / RATE, 2),
            psr=round(estimate.psr, 1),
            valid=estimate.valid,
            confirmed=estimate.confirmed,
            shift_ms=self._shift * 1000 / RATE,
            speex_shift_ms=self.speex_shift_ms,
            bypassed=self.echo_absent,
        )
        if not estimate.valid:
            self._resume.clear()
            self._no_echo += HOP_SAMPLES
            if (
                params.bypass_after is not None
                and self._no_echo >= params.bypass_after
                and not self.echo_absent
            ):
                self.echo_absent = True
                self.stats.bypasses += 1
                self.note_input("bypass", end, on=True)
                logger.info("no echo on the line: the echo canceller is bypassed")
            return False
        self.stats.estimates += 1
        if self.echo_absent and not self._resumes(estimate):
            return False  # a chance peak persists across overlapping windows: wait for more
        if not estimate.confirmed:
            return False  # one window can peak by chance (a caller talking over us): wait
        self._no_echo = 0
        if self.echo_absent:
            self.echo_absent = False
            self._resume.clear()
            self.note_input("bypass", end, on=False)
            logger.info("echo on the line: the echo canceller is back on")
        self._learn(end, estimate)
        self._align_speex(end, estimate)
        if params.keep_min <= estimate.lag <= params.keep_max:
            return False
        self._shift += estimate.lag - params.lead
        self.stats.realignments += 1
        self.note_input(
            "realign",
            end,
            engine="aec3",
            residual_ms=round(estimate.lag * 1000 / RATE, 2),
            shift_ms=self._shift * 1000 / RATE,
        )
        logger.info(
            "echo reference re-aligned",
            extra={
                "residual_ms": estimate.lag * 1000 // RATE,
                "shift_ms": self._shift * 1000 // RATE,
            },
        )
        return True

    def _resumes(self, estimate: DelayEstimate) -> bool:
        """While bypassed: whether ``estimate`` completes a run of ``_RESUME_WINDOWS`` valid
        windows in a row that agree, each with a PSR of at least ``_RESUME_PSR``."""
        if estimate.psr < _RESUME_PSR:
            self._resume.clear()
            return False
        if self._resume and abs(estimate.lag - self._resume[-1]) > _RESUME_AGREE:
            self._resume.clear()
        self._resume.append(estimate.lag)
        return len(self._resume) >= _RESUME_WINDOWS

    # Outputs

    @property
    def offset(self) -> int | None:
        """Loop samples minus input samples, once input has arrived."""
        return self._capture.offset

    @property
    def anchor_ms(self) -> int:
        return self._anchor * 1000 // RATE

    @property
    def shift_ms(self) -> int:
        return self._shift * 1000 // RATE

    @property
    def speex_aligned(self) -> bool:
        """Whether the ASR path's linear canceller has a reference (an echo was confirmed)."""
        return self._speex_shift is not None

    @property
    def speex_shift_ms(self) -> int | None:
        return None if self._speex_shift is None else self._speex_shift * 1000 // RATE

    @property
    def tick_phase_ms(self) -> float:
        return self._model.phase * 1000 / RATE

    @property
    def busy(self) -> bool:
        """Whether more audio may still reach the client model."""
        return self._active or any(count > 0 for count in self._pending.values())

    def final_until(self) -> int | None:
        """The input sample before which the references can no longer change; None: anywhere."""
        offset = self._capture.offset
        if offset is None or not self.busy:
            return None
        now = self._now()
        self._advance(now)
        horizon = max(now, self._model.queue_end() or now)
        return horizon - offset - self.params.lead + self._least_shift()

    def final_at(self, end_sample: int) -> float | None:
        """The loop time (s) when the references up to ``end_sample`` are final, if waiting
        will do it (new audio may finalize them earlier)."""
        offset = self._capture.offset
        if offset is None:
            return None
        return (end_sample + offset + self.params.lead - self._least_shift()) / RATE

    def render(self, start: int, count: int) -> npt.NDArray[np.int16]:
        """AEC3's reference for input samples [start, start + count)."""
        return self._render(start, count, self._shift)

    def render_speex(self, start: int, count: int) -> npt.NDArray[np.int16] | None:
        """The ASR path's reference for the same samples; None until it is aligned."""
        if self._speex_shift is None:
            return None
        return self._render(start, count, self._speex_shift)

    def _render(self, start: int, count: int, shift: int) -> npt.NDArray[np.int16]:
        out = np.zeros(count, dtype=np.int16)
        offset = self._capture.offset
        if offset is None:
            return out
        first = start + offset + self.params.lead - shift
        last = first + count
        # Run the model up to the rendered stretch, but never past what is determined: ticks
        # beyond now only while queued frames cover them (later arrivals queue behind those).
        now = self._now()
        self._advance(now)
        determined = max(now, self._model.queue_end() or now)
        self._advance(min(last, determined) - 1)
        tick = self.params.playout.tick
        # Keep what the reference placed furthest back (the larger shift) may still read.
        oldest = start + offset + self.params.lead - self._greatest_shift()
        while self._played and self._played[0].start + tick <= oldest - _RENDER_MARGIN:
            self._played.popleft()
        for entry in self._played:
            if entry.start >= last:
                break
            if entry.start + tick <= first:
                continue
            audio = entry.handle.item.frame(entry.handle.index, tick)
            low, high = max(entry.start, first), min(entry.start + tick, last)
            out[low - first : high - first] = audio[low - entry.start : high - entry.start]
        return out

    def playback(self, end_sample: int) -> Playback:
        """Whether the client is modelled as playing at input ``end_sample``: from the first
        frame of a response's audio to its last (or the flush), plus the echo tail."""
        offset = self._capture.offset
        if offset is None:
            return Playback(False, None)
        at = end_sample + offset
        self._advance(min(at, self._now()))
        params = self.params
        for item in self._items.values():
            if not item.plays or item.plays[0] > at:
                continue
            still = item.queued > 0 or not item.complete
            if still or at < item.plays[-1] + params.playout.tick + params.tail:
                guard = None if params.guard == 0 else item.plays[0] + params.guard - offset
                return Playback(True, guard)
        return Playback(False, None)

    def played_spans(self, start_sample: int, end_sample: int, before: int) -> list[PlayedSpan]:
        """The assistant audio the client is modelled to have played from ``before`` samples
        before input ``start_sample`` up to ``end_sample`` (the echo-turn window)."""
        offset = self._capture.offset
        if offset is None:
            return []
        self._advance(self._now())
        low = start_sample + offset - before - self.params.tail
        high = end_sample + offset
        tick = self.params.playout.tick
        tick_ms = tick * 1000 // RATE
        spans: list[PlayedSpan] = []
        for item in self._items.values():
            first = bisect_left(item.plays, low - tick)
            last = bisect_right(item.plays, high)
            if first < last:
                spans.append(PlayedSpan(item.item_id, first * tick_ms, last * tick_ms))
        return spans

    # Internals

    def _now(self) -> int:
        return samples(self._clock.now())

    def _least_shift(self) -> int:
        """The shift of the reference that reads the furthest ahead of the input."""
        return self._shift if self._speex_shift is None else min(self._shift, self._speex_shift)

    def _greatest_shift(self) -> int:
        return self._shift if self._speex_shift is None else max(self._shift, self._speex_shift)

    def _align_speex(self, end: int, estimate: DelayEstimate) -> None:
        """Place the ASR path's reference from a confirmed estimate: the echo trails the
        unshifted reference by ``_shift + lag``; it should trail Speex's by ``speex_target``,
        and Speex's reference moves again only when the echo leaves [0, ``speex_keep``]."""
        params = self.params
        if not params.speex:
            return
        delay = self._shift + estimate.lag
        current = self._speex_shift
        if current is not None and (
            end < self._speex_hold or 0 <= delay - current <= params.speex_keep
        ):
            return
        self._speex_shift = delay - params.speex_target
        self.note_input(
            "realign",
            end,
            engine="speex",
            first=current is None,
            residual_ms=None if current is None else round((delay - current) * 1000 / RATE, 2),
            shift_ms=self._speex_shift * 1000 / RATE,
        )
        stats = self.stats
        if current is None:
            stats.speex_aligned_at = end
            logger.info(
                "echo reference aligned for the asr path",
                extra={"echo_delay_ms": delay * 1000 // RATE, "at_ms": end * 1000 // RATE},
            )
            return
        stats.speex_realignments += 1
        logger.info(
            "echo reference re-aligned for the asr path",
            extra={
                "residual_ms": (delay - current) * 1000 // RATE,
                "shift_ms": self._speex_shift * 1000 // RATE,
            },
        )

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def _settle(self, generation: int) -> None:
        count = self._pending.get(generation, 0) - 1
        if count > 0:
            self._pending[generation] = count
        else:
            self._pending.pop(generation, None)
            self._finish_if_sent(generation)
        self._changed()

    def _finish_if_sent(self, generation: int) -> None:
        """Once a finished response's audio has all been sent, its items are complete."""
        if generation not in self._ended or self._pending.get(generation, 0) > 0:
            return
        for item in self._items.values():
            if item.generation == generation and not item.complete:
                first = item.frames
                self._arrive(item, first, item.finish())
                self._note_end(item)
        self._ended.discard(generation)
        self._prune()

    def _arrive(self, item: _ItemAudio, first: int, count: int) -> None:
        if count <= 0:
            return
        item.queued += count
        frames = [_Frame(item, index) for index in range(first, first + count)]
        now = self._now()
        self._advance(now + self._anchor - 1)  # what arrive does first: the queue as it stands
        self._audible(now, count, idle=self._model.queued == 0)
        self._record(self._model.arrive(frames, now + self._anchor))

    def _audible(self, now: int, count: int, *, idle: bool) -> None:
        """AICC's queueFrame bookkeeping for ``count`` frames sent at ``now``."""
        if idle and (
            self._audible_since is None or now > self._audible_until + self.params.aicc_gap_slack
        ):
            self._audible_since = self._audible_until = now
        if self._audible_since is not None:
            self._audible_until += count * self.params.playout.tick

    def _advance(self, until: int) -> None:
        self._record(self._model.advance(until))

    def _record(self, played: Sequence[Played[_Frame]]) -> None:
        tick = self.params.playout.tick
        for entry in played:
            frame = entry.handle
            frame.item.queued -= 1
            frame.item.plays.append(entry.start)
            self._played.append(entry)
            if len(frame.item.plays) == 1:
                self._note_play(frame.item, entry)
            self._note_end(frame.item)
            run = next((run for run in reversed(self._runs) if run.start <= entry.start), None)
            if run is not None:
                if entry.start > run.last_end:
                    run.broken = True  # an underrun gap inside the run
                run.last_end = entry.start + tick

    def _run_started(self, start: RunStart) -> None:
        run = _Run(start.start, start.trigger, start.by_count, self._shift, start.start)
        self._runs.append(run)

    def _learn(self, end: int, estimate: DelayEstimate) -> None:
        """Tick-phase learning: the first confirmed estimate whose window lies inside an
        unbroken run locates that run's echo onset."""
        learner, offset = self._learner, self._capture.offset
        if learner is None or offset is None:
            return
        # The model time of the reference the window's echo came from.
        to_model = offset + self.params.lead - self._shift - estimate.lag
        low, high = end - 2 * RATE + to_model, end + to_model
        for run in reversed(self._runs):
            if run.start > low:
                continue
            if run.observed or not run.by_count or run.broken or run.shift != self._shift:
                return
            if high > run.last_end:
                return  # the window runs past the run
            run.observed = True
            # Where the run's echo began, less a constant (lead, echo path, transit).
            learner.observe(run.trigger, run.start + self._shift + estimate.lag)
            phase = learner.phase()
            if phase is not None and phase != self._model.phase:
                self._model.set_phase(phase)
                logger.info(
                    "client tick phase learned",
                    extra={"phase_ms": round(phase * 1000 / RATE, 2), "runs": learner.runs},
                )
            return

    def _prune(self) -> None:
        """Forget items that finished playing long ago."""
        horizon = self._now() - _HISTORY
        for item_id in [
            item_id
            for item_id, item in self._items.items()
            if item.complete and item.queued == 0 and (item.last_end or horizon) <= horizon
        ]:
            del self._items[item_id]


@dataclass(slots=True)
class _Segment:
    stream: int  # position in the canceller's stream (frames fed, gaps removed)
    capture: int  # the input sample it corresponds to


@dataclass(frozen=True, slots=True)
class Processed:
    """A stretch of input after the cancellers: ``pcm`` is AEC3's output (speech and turn
    detection; the input itself for a partial frame let through), ``asr`` the ASR path's for
    the same samples (None: the ASR keeps the input there)."""

    start: int  # input timeline sample
    pcm: bytes
    asr: bytes | None = None


class AsrPath:
    """The ASR path's linear canceller for one session (G7 dual path): the low-pass pre-filter
    on its capture and reference (optional), then the engine. Runs on the canceller executor
    with the stage's frames, in order."""

    def __init__(self, engine: EchoEngine, prefilter_hz: int | None) -> None:
        self.engine = engine
        self._capture_filter = None if prefilter_hz is None else LowPass(prefilter_hz)
        self._reference_filter = None if prefilter_hz is None else LowPass(prefilter_hz)

    def capture(self, frame: Pcm) -> Pcm:
        return frame if self._capture_filter is None else self._capture_filter(frame)

    def reference(self, frame: Pcm) -> Pcm:
        return frame if self._reference_filter is None else self._reference_filter(frame)


@dataclass(slots=True)
class _Batch:
    output: npt.NDArray[np.int16]
    asr: npt.NDArray[np.int16] | None = None
    estimates: list[DelayEstimate] = field(default_factory=list[DelayEstimate])
    engine_ns: int = 0
    asr_ns: int = 0
    active: int = 0  # frames with an active reference, and their energy in and out
    capture_energy: float = 0.0
    output_energy: float = 0.0
    asr_active: int = 0  # the same for the ASR path (its pre-filtered capture)
    asr_capture_energy: float = 0.0
    asr_output_energy: float = 0.0


class EchoStage:
    """The cancellers on one session's VAD feeder path (see the module notes)."""

    def __init__(
        self,
        engine: EchoEngine,
        reference: EchoReference,
        executor: Executor,
        asr: AsrPath | None = None,
    ) -> None:
        self._engine = engine
        self._asr = asr
        self._reference = reference
        self._executor = executor
        self._estimator = DelayEstimator() if reference.params.delay_guard else None
        self._restart = False
        self._pending = bytearray()
        self._pending_start = 0
        self._stream = 0
        self._segments: deque[_Segment] = deque(maxlen=8)

    @property
    def pending_end(self) -> int:
        return self._pending_start + len(self._pending) // SAMPLE_WIDTH

    async def feed(self, start: int, pcm: bytes) -> list[Processed]:
        """Input audio at ``start``; returns processed audio."""
        out: list[Processed] = []
        if self._pending and start != self.pending_end:
            out += await self.release(force=True)  # a gap (a commit or clear ended the stream)
        if not self._pending:
            self._pending_start = start
        self._pending += pcm
        out += await self.release()
        return out

    async def release(self, *, force: bool = False) -> list[Processed]:
        """Process the held frames whose reference is final (``force``: all of them, and let
        a partial frame through as it is)."""
        out: list[Processed] = []
        whole = len(self._pending) // _FRAME_BYTES
        if not force:
            final = self._reference.final_until()
            if final is not None:
                whole = min(whole, max(final - self._pending_start, 0) // FRAME_SAMPLES)
        if whole:
            out.append(await self._process(whole))
        if force and self._pending:
            out.append(Processed(self._pending_start, bytes(self._pending)))
            self._capture_passed(self._pending_start, bytes(self._pending))
            self._pending_start = self.pending_end
            self._pending.clear()
        return out

    def wake_at(self) -> float | None:
        """When held frames will become final by time alone (loop seconds); None if none wait
        or only new audio or events can release them."""
        if len(self._pending) < _FRAME_BYTES or not self._reference.busy:
            return None
        return self._reference.final_at(self._pending_start + FRAME_SAMPLES)

    async def _process(self, whole: int) -> Processed:
        start = self._pending_start
        size = whole * _FRAME_BYTES
        count = whole * FRAME_SAMPLES
        capture = np.frombuffer(bytes(self._pending[:size]), dtype="<i2").astype(np.int16)
        reference = self._reference.render(start, count)
        asr_reference = None if self._asr is None else self._reference.render_speex(start, count)
        del self._pending[:size]
        self._pending_start += count
        if (
            not self._segments
            or self._segments[-1].stream + (start - self._segments[-1].capture) != self._stream
        ):
            self._segments.append(_Segment(self._stream, start))
        self._stream += count
        restart, self._restart = self._restart, False
        bypass = self._reference.echo_absent
        batch = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._run,
            capture.reshape(whole, FRAME_SAMPLES),
            reference.reshape(whole, FRAME_SAMPLES),
            None if asr_reference is None else asr_reference.reshape(whole, FRAME_SAMPLES),
            restart,
            bypass,
        )
        stats = self._reference.stats
        stats.frames += whole
        stats.engine_ns += batch.engine_ns
        stats.speex_ns += batch.asr_ns
        if batch.active and not bypass:
            stats.add(batch.active, batch.capture_energy, batch.output_energy)
        if batch.asr_active:
            stats.speex_erle.add(
                batch.asr_active, batch.asr_capture_energy, batch.asr_output_energy
            )
        if self._asr is not None and isinstance(self._asr.engine, SpeexLinear):
            stats.speex_resets = self._asr.engine.resets
        for estimate in batch.estimates:
            end = self._capture_position(estimate.end)
            if end is not None and self._reference.delay_estimate(end, estimate):
                self._restart = True
                break  # later windows of this batch were measured against the old alignment
        sink = self._reference.capture
        if sink is not None and sink.active:
            sink.audio(
                start,
                capture.reshape(-1),
                reference,
                batch.output,
                None if asr_reference is None else asr_reference,
                batch.asr,
            )
        asr = None if batch.asr is None else batch.asr.tobytes()
        return Processed(start, batch.output.tobytes(), asr)

    def _capture_passed(self, start: int, pcm: bytes) -> None:
        """A partial frame let through as it is: no reference, no cancelling."""
        sink = self._reference.capture
        if sink is not None and sink.active:
            samples = np.frombuffer(pcm, dtype="<i2")
            sink.audio(start, samples, np.zeros_like(samples), samples)

    def _run(
        self,
        capture: npt.NDArray[np.int16],
        reference: npt.NDArray[np.int16],
        asr_reference: npt.NDArray[np.int16] | None,
        restart: bool,
        bypass: bool,
    ) -> _Batch:
        """On the canceller executor: AEC3 over each frame (with a silent reference when
        bypassed), the ASR path's canceller (passing its pre-filtered input through until its
        reference is aligned), then the delay estimator (with the real reference)."""
        output = np.empty_like(capture)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        began = time.thread_time_ns()
        for index in range(len(capture)):
            fed = silence if bypass else reference[index]
            output[index] = self._engine.process(capture[index], fed)
        batch = _Batch(output.reshape(-1), engine_ns=time.thread_time_ns() - began)
        active = _active(reference)
        if np.any(active):
            batch.active = int(np.count_nonzero(active))
            batch.capture_energy = _energy(capture[active])
            batch.output_energy = _energy(output[active])
        if self._asr is not None:
            self._run_asr(self._asr, batch, capture, asr_reference)
        if self._estimator is not None:
            if restart:
                self._estimator.restart()
            batch.estimates = self._estimator.push(capture.reshape(-1), reference.reshape(-1))
        return batch

    @staticmethod
    def _run_asr(
        path: AsrPath,
        batch: _Batch,
        capture: npt.NDArray[np.int16],
        reference: npt.NDArray[np.int16] | None,
    ) -> None:
        """The ASR path never takes the bypass: a linear filter has no suppressor to silence a
        caller on a line without echo (it just adapts to nothing), and the bypass can misfire
        in long double talk, where GCC-PHAT finds no distinct peak."""
        began = time.thread_time_ns()
        # The pre-filter's history stays continuous; its output feeds the filter only once the
        # reference is aligned. Until then the input passes through as received (a line without
        # echo, or a wideband client, keeps its full band).
        filtered = path.capture(capture.reshape(-1)).reshape(capture.shape)
        output = capture.copy()
        if reference is not None:
            fed = path.reference(reference.reshape(-1)).reshape(reference.shape)
            for index in range(len(capture)):
                output[index] = path.engine.process(filtered[index], fed[index])
        batch.asr_ns = time.thread_time_ns() - began
        batch.asr = output.reshape(-1)
        if reference is not None:
            active = _active(reference)
            if np.any(active):
                batch.asr_active = int(np.count_nonzero(active))
                batch.asr_capture_energy = _energy(filtered[active])
                batch.asr_output_energy = _energy(output[active])

    def _capture_position(self, stream_position: int) -> int | None:
        for segment in reversed(self._segments):
            if segment.stream <= stream_position:
                return segment.capture + stream_position - segment.stream
        return None


def _active(reference: npt.NDArray[np.int16]) -> npt.NDArray[np.bool_]:
    """The frames whose reference is above -50 dBFS."""
    return np.mean(reference.astype(np.float64) ** 2, axis=1) > _ACTIVE_REFERENCE


def _energy(frames: npt.NDArray[np.int16]) -> float:
    return float(np.sum(frames.astype(np.float64) ** 2))


class EchoRuntime:
    """Process-wide: the canceller executor and the engine factories (one engine of each kind
    per session)."""

    def __init__(self, aec: AecConfig) -> None:
        self._engine: EngineName = aec.engine
        self._speex: SpeexConfig | None = aec.speex if aec.asr_input == "speex" else None
        self.executor = ThreadPoolExecutor(aec.workers, thread_name_prefix="aec")

    def engine(self) -> EchoEngine:
        return make_engine(self._engine)

    def asr_path(self) -> AsrPath | None:
        """The ASR path's canceller for a new session (``aec.asr_input: speex``)."""
        speex = self._speex
        if speex is None:
            return None
        engine = SpeexLinear(speex.filter_ms, preprocess=speex.preprocess)
        return AsrPath(engine, speex.prefilter_hz)

    async def aclose(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def log_echo_stats(reference: EchoReference, counters: dict[str, int]) -> dict[str, object]:
    """One line per session: the cancellers' work and the echo the barge-in rules met. Returns
    its fields (the capture's ``meta.json`` keeps them)."""
    stats = reference.stats
    aligned = stats.speex_aligned_at
    fields: dict[str, object] = {
        "aec_frames": stats.frames,
        "aec_cpu_ms_per_audio_s": stats.cpu_ms_per_audio_s(),
        "erle_db": stats.erle_db(),
        "realignments": stats.realignments,
        "anchors": stats.anchors,
        "bypassed": stats.bypasses,
        "echo_absent": reference.echo_absent,
        "shift_ms": reference.shift_ms,
        "anchor_ms": reference.anchor_ms,
        "tick_phase_ms": round(reference.tick_phase_ms, 2),
        "speex_cpu_ms_per_audio_s": stats.speex_cpu_ms_per_audio_s(),
        "speex_erle_db": stats.speex_erle.median_db(),
        "speex_aligned_at_ms": None if aligned is None else aligned * 1000 // RATE,
        "speex_shift_ms": reference.speex_shift_ms,
        "speex_realignments": stats.speex_realignments,
        "speex_resets": stats.speex_resets,
        **counters,
    }
    logger.info("echo summary", extra=fields)
    return fields

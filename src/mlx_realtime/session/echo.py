"""Echo cancellation for one session (ARCHITECTURE §4.3, §6.5; D17, G7).

Neither AICC nor FreeSWITCH cancels echo, so the caller's audio carries our own speech back.
Three parts handle it:

- ``CaptureClock`` maps the session's input timeline to loop time, from when the audio arrived:
  the client streams its capture in real time, so the earliest arrivals pin the offset.
- ``EchoReference`` is what the client is modelled to play: every audio delta as the writer
  hands it to the socket, resampled 24 → 16 kHz, framed in 20 ms and played on the AICC playout
  model (``audio/playout.py``). From it come the canceller's reference for any stretch of the
  input (placed ``lead`` early, so it precedes its echo, and moved by the GCC-PHAT guard), whether
  the client was playing at a point of the input (the barge-in rules), and which assistant audio
  played around a user turn (echo-turn suppression). A barge-in or cancel flushes the frames
  the client never played; a client truncate drops the rest of that item and re-anchors the
  model's timing.
- ``EchoStage`` runs the canceller on the VAD feeder's path: whole 10 ms frames, each fed in lock
  step with the reference frame for the same input positions (zeros when nothing plays, so the
  delay is carried in the content), on the canceller executor, in order. While a response may
  still add audio, input is held back until its reference can no longer change.

On a line without echo, AEC3's suppressor takes a caller who talks over our audio for echo and
silences them (measured in M4: up to 45 dB, for seconds). So once the delay estimator has watched
``bypass_after`` of playing reference with no echo of it in the input, the canceller is fed a
silent reference (it then passes the input through); the first echo the estimator finds turns
it back on (AEC3 re-converges within about 0.5 s).

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

from mlx_realtime.audio.aec import FRAME_SAMPLES, EchoEngine, EngineName, make_engine
from mlx_realtime.audio.delay import HOP, DelayEstimate, DelayEstimator
from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, WIRE_RATE
from mlx_realtime.audio.playout import (
    Played,
    PlayoutModel,
    PlayoutParams,
    RunStart,
    TickPhaseLearner,
)
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.config import AecConfig, BargeInConfig
from mlx_realtime.session.clock import Clock

logger = logging.getLogger(__name__)

RATE = CANONICAL_RATE
_FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH
_WIRE_PER_FRAME = WIRE_RATE // 50  # 24 kHz samples in one 20 ms client frame
_CAPTURE_WINDOW = 10 * RATE  # the capture clock's sliding window
_MAX_ANCHOR = RATE // 2  # re-anchoring moves the model by at most 500 ms
_HISTORY = 60 * RATE  # how long played audio is remembered for echo-turn matching
_RENDER_MARGIN = 2 * RATE  # played frames kept behind the reference already rendered
_ACTIVE_REFERENCE = 10 ** (-50 / 10) * 32768.0**2  # a reference frame above -50 dBFS
_ERLE_WINDOW_FRAMES = 50  # 0.5 s of frames with an active reference
HOP_SAMPLES = HOP * RATE // 8000  # the delay estimator's hop at 16 kHz


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

    def __init__(self, item_id: str, generation: int) -> None:
        self.item_id = item_id
        self.generation = generation
        self._resampler = Resampler(WIRE_RATE, RATE)
        self._reference = bytearray()  # 16 kHz, as resampled so far
        self._wire = 0  # 24 kHz samples sent
        self.frames = 0  # 20 ms frames handed to the client model
        self.complete = False
        self.plays: list[int] = []  # play start of frame i (frames play in order)
        self.queued = 0  # frames waiting in the client model's queue

    @property
    def last_end(self) -> int | None:
        return None if not self.plays else self.plays[-1]

    def add(self, pcm: bytes) -> int:
        """Audio sent; returns how many whole 20 ms frames it completed."""
        self._reference += self._resampler.process(pcm)
        self._wire += len(pcm) // SAMPLE_WIDTH
        whole = self._wire // _WIRE_PER_FRAME
        added, self.frames = whole - self.frames, whole
        return added

    def finish(self) -> int:
        """The item's audio is complete: the client pads the last partial frame (its framer
        does at a response's end). Returns the frames added."""
        if self.complete:
            return 0
        self.complete = True
        self._reference += self._resampler.flush()
        if self._wire % _WIRE_PER_FRAME:
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
class EchoStats:
    """The canceller's work in one session. ERLE is the median over 0.5 s windows in which the
    reference was active, so windows where the caller talked over us weigh little."""

    frames: int = 0
    engine_ns: int = 0
    realignments: int = 0
    estimates: int = 0
    anchors: int = 0
    bypasses: int = 0
    _window: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # frames, in, out
    _erle: deque[float] = field(default_factory=lambda: deque[float](maxlen=1200))

    def add(self, frames: int, capture_energy: float, output_energy: float) -> None:
        """``frames`` processed with an active reference, and their energy in and out."""
        window = self._window
        window[0] += frames
        window[1] += capture_energy
        window[2] += output_energy
        if window[0] >= _ERLE_WINDOW_FRAMES:
            if window[1] > 0:
                self._erle.append(10 * float(np.log10(window[1] / max(window[2], 1.0))))
            self._window = [0.0, 0.0, 0.0]

    def erle_db(self) -> float | None:
        if not self._erle:
            return None
        return round(float(np.median(np.asarray(self._erle))), 1)

    def cpu_ms_per_audio_s(self) -> float | None:
        audio_s = self.frames * FRAME_SAMPLES / RATE
        return None if audio_s == 0 else round(self.engine_ns / 1e6 / audio_s, 2)


class EchoReference:
    """The client's modelled playout of this session's output (see the module notes).

    Fed on the event loop by the actor (responses, cancels, truncates, input arrivals) and by
    the writer through the output ledger (audio queued, sent, dropped); read by the VAD feeder.
    """

    def __init__(self, params: EchoParams, clock: Clock) -> None:
        self.params = params
        self._clock = clock
        self._model = PlayoutModel[_Frame](params.playout, on_run_start=self._run_started)
        self._capture = CaptureClock()
        self._items: dict[str, _ItemAudio] = {}
        self._played: deque[Played[_Frame]] = deque()  # for rendering, by start
        self._runs: deque[_Run] = deque(maxlen=64)
        self._learner = TickPhaseLearner(params.playout.tick) if params.learn_tick_phase else None
        self._shift = 0  # the reference is delayed by this much (the guard's re-alignment)
        self._anchor = 0  # added to send times (re-anchored by client truncates)
        self._pending: dict[int, int] = {}  # generation -> deltas queued, not sent or dropped
        self._ended: set[int] = set()  # generations whose response is over
        self._active = False  # a response is in progress
        self._no_echo = 0  # reference played with no echo found since the last echo
        self.echo_absent = False  # the canceller is bypassed (see the module notes)
        self.stats = EchoStats()
        self.on_change: Callable[[], None] | None = None  # the feeder's wakeup

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

    def audio_sent(self, item_id: str, generation: int, pcm: bytes) -> None:
        """The writer handed an audio delta to the socket."""
        item = self._items.get(item_id)
        if item is None:
            item = self._items[item_id] = _ItemAudio(item_id, generation)
        first = item.frames
        added = item.add(pcm)
        self._arrive(item, first, added)
        self._settle(generation)

    def audio_dropped(self, generation: int) -> None:
        """The writer discarded a delta of a cancelled response."""
        self._settle(generation)

    def flush(self, generation: int) -> None:
        """A barge-in or cancel: the client discards the frames of ``generation`` it has not
        played (§6.5 step 4); nothing more of them will come."""
        self._advance(self._now())
        for frame in self._model.drop(keep=lambda frame: frame.item.generation != generation):
            frame.item.queued -= 1
        for item in self._items.values():
            if item.generation == generation:
                item.complete = True
        self._ended.add(generation)
        self._pending.pop(generation, None)
        self._changed()

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
                self._anchor = max(-_MAX_ANCHOR, min(_MAX_ANCHOR, anchor))
                self.stats.anchors += 1
                logger.info(
                    "echo reference re-anchored",
                    extra={"item": item_id, "frames_off": difference, "anchor_ms": self.anchor_ms},
                )
        for frame in self._model.drop(keep=lambda frame: frame.item is not item):
            frame.item.queued -= 1
        for entry in [entry for entry in self._played if entry.handle.item is item]:
            if entry.handle.index >= heard:
                self._played.remove(entry)  # never played: no reference for it
        del item.plays[heard:]
        item.complete = True
        self._changed()

    def delay_estimate(self, end: int, estimate: DelayEstimate) -> bool:
        """A GCC-PHAT window, evaluated while the reference played, that ended at input sample
        ``end``. Returns whether the reference was re-aligned (the estimator must then start
        over)."""
        params = self.params
        if not estimate.valid:
            self._no_echo += HOP_SAMPLES
            if (
                params.bypass_after is not None
                and self._no_echo >= params.bypass_after
                and not self.echo_absent
            ):
                self.echo_absent = True
                self.stats.bypasses += 1
                logger.info("no echo on the line: the echo canceller is bypassed")
            return False
        self.stats.estimates += 1
        if not estimate.confirmed:
            return False  # one window can peak by chance (a caller talking over us): wait
        self._no_echo = 0
        if self.echo_absent:
            self.echo_absent = False
            logger.info("echo on the line: the echo canceller is back on")
        self._learn(end, estimate)
        if params.keep_min <= estimate.lag <= params.keep_max:
            return False
        self._shift += estimate.lag - params.lead
        self.stats.realignments += 1
        logger.info(
            "echo reference re-aligned",
            extra={
                "residual_ms": estimate.lag * 1000 // RATE,
                "shift_ms": self._shift * 1000 // RATE,
            },
        )
        return True

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
    def tick_phase_ms(self) -> float:
        return self._model.phase * 1000 / RATE

    @property
    def busy(self) -> bool:
        """Whether more audio may still reach the client model."""
        return self._active or any(count > 0 for count in self._pending.values())

    def final_until(self) -> int | None:
        """The input sample before which the reference can no longer change; None: anywhere."""
        offset = self._capture.offset
        if offset is None or not self.busy:
            return None
        now = self._now()
        self._advance(now)
        horizon = max(now, self._model.queue_end() or now)
        return horizon - offset - self.params.lead + self._shift

    def final_at(self, end_sample: int) -> float | None:
        """The loop time (s) when the reference up to ``end_sample`` is final, if waiting
        will do it (new audio may finalize it earlier)."""
        offset = self._capture.offset
        if offset is None:
            return None
        return (end_sample + offset + self.params.lead - self._shift) / RATE

    def render(self, start: int, count: int) -> npt.NDArray[np.int16]:
        """The reference for input samples [start, start + count)."""
        out = np.zeros(count, dtype=np.int16)
        offset = self._capture.offset
        if offset is None:
            return out
        first = start + offset + self.params.lead - self._shift
        last = first + count
        # Run the model up to the rendered stretch, but never past what is determined: ticks
        # beyond now only while queued frames cover them (later arrivals queue behind those).
        now = self._now()
        self._advance(now)
        determined = max(now, self._model.queue_end() or now)
        self._advance(min(last, determined) - 1)
        tick = self.params.playout.tick
        while self._played and self._played[0].start + tick <= first - _RENDER_MARGIN:
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
        self._ended.discard(generation)
        self._prune()

    def _arrive(self, item: _ItemAudio, first: int, count: int) -> None:
        if count <= 0:
            return
        item.queued += count
        frames = [_Frame(item, index) for index in range(first, first + count)]
        self._record(self._model.arrive(frames, self._now() + self._anchor))

    def _advance(self, until: int) -> None:
        self._record(self._model.advance(until))

    def _record(self, played: Sequence[Played[_Frame]]) -> None:
        tick = self.params.playout.tick
        for entry in played:
            frame = entry.handle
            frame.item.queued -= 1
            frame.item.plays.append(entry.start)
            self._played.append(entry)
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


@dataclass(slots=True)
class _Batch:
    output: npt.NDArray[np.int16]
    estimates: list[DelayEstimate] = field(default_factory=list[DelayEstimate])
    engine_ns: int = 0
    active: int = 0  # frames with an active reference, and their energy in and out
    capture_energy: float = 0.0
    output_energy: float = 0.0


class EchoStage:
    """The canceller on one session's VAD feeder path (see the module notes)."""

    def __init__(self, engine: EchoEngine, reference: EchoReference, executor: Executor) -> None:
        self._engine = engine
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

    async def feed(self, start: int, pcm: bytes) -> list[tuple[int, bytes]]:
        """Input audio at ``start``; returns processed audio, each piece with its start."""
        out: list[tuple[int, bytes]] = []
        if self._pending and start != self.pending_end:
            out += await self.release(force=True)  # a gap (a commit or clear ended the stream)
        if not self._pending:
            self._pending_start = start
        self._pending += pcm
        out += await self.release()
        return out

    async def release(self, *, force: bool = False) -> list[tuple[int, bytes]]:
        """Process the held frames whose reference is final (``force``: all of them, and let
        a partial frame through as it is)."""
        out: list[tuple[int, bytes]] = []
        whole = len(self._pending) // _FRAME_BYTES
        if not force:
            final = self._reference.final_until()
            if final is not None:
                whole = min(whole, max(final - self._pending_start, 0) // FRAME_SAMPLES)
        if whole:
            out.append(await self._process(whole))
        if force and self._pending:
            out.append((self._pending_start, bytes(self._pending)))
            self._pending_start = self.pending_end
            self._pending.clear()
        return out

    def wake_at(self) -> float | None:
        """When held frames will become final by time alone (loop seconds); None if none wait
        or only new audio or events can release them."""
        if len(self._pending) < _FRAME_BYTES or not self._reference.busy:
            return None
        return self._reference.final_at(self._pending_start + FRAME_SAMPLES)

    async def _process(self, whole: int) -> tuple[int, bytes]:
        start = self._pending_start
        size = whole * _FRAME_BYTES
        capture = np.frombuffer(bytes(self._pending[:size]), dtype="<i2").astype(np.int16)
        reference = self._reference.render(start, whole * FRAME_SAMPLES)
        del self._pending[:size]
        self._pending_start += whole * FRAME_SAMPLES
        if (
            not self._segments
            or self._segments[-1].stream + (start - self._segments[-1].capture) != self._stream
        ):
            self._segments.append(_Segment(self._stream, start))
        self._stream += whole * FRAME_SAMPLES
        restart, self._restart = self._restart, False
        bypass = self._reference.echo_absent
        batch = await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._run,
            capture.reshape(whole, FRAME_SAMPLES),
            reference.reshape(whole, FRAME_SAMPLES),
            restart,
            bypass,
        )
        stats = self._reference.stats
        stats.frames += whole
        stats.engine_ns += batch.engine_ns
        if batch.active and not bypass:
            stats.add(batch.active, batch.capture_energy, batch.output_energy)
        for estimate in batch.estimates:
            end = self._capture_position(estimate.end)
            if end is not None and self._reference.delay_estimate(end, estimate):
                self._restart = True
        return start, batch.output.tobytes()

    def _run(
        self,
        capture: npt.NDArray[np.int16],
        reference: npt.NDArray[np.int16],
        restart: bool,
        bypass: bool,
    ) -> _Batch:
        """On the canceller executor: the engine over each frame (with a silent reference when
        bypassed), then the delay estimator (with the real one)."""
        output = np.empty_like(capture)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        began = time.thread_time_ns()
        for index in range(len(capture)):
            fed = silence if bypass else reference[index]
            output[index] = self._engine.process(capture[index], fed)
        batch = _Batch(output.reshape(-1), engine_ns=time.thread_time_ns() - began)
        power = np.mean(reference.astype(np.float64) ** 2, axis=1)
        active = power > _ACTIVE_REFERENCE
        if np.any(active):
            batch.active = int(np.count_nonzero(active))
            batch.capture_energy = float(np.sum(capture[active].astype(np.float64) ** 2))
            batch.output_energy = float(np.sum(output[active].astype(np.float64) ** 2))
        if self._estimator is not None:
            if restart:
                self._estimator.restart()
            batch.estimates = self._estimator.push(capture.reshape(-1), reference.reshape(-1))
        return batch

    def _capture_position(self, stream_position: int) -> int | None:
        for segment in reversed(self._segments):
            if segment.stream <= stream_position:
                return segment.capture + stream_position - segment.stream
        return None


class EchoRuntime:
    """Process-wide: the canceller executor and engine factory."""

    def __init__(self, engine: EngineName, workers: int) -> None:
        self._engine: EngineName = engine
        self.executor = ThreadPoolExecutor(workers, thread_name_prefix="aec")

    def engine(self) -> EchoEngine:
        return make_engine(self._engine)

    async def aclose(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def log_echo_stats(reference: EchoReference, counters: dict[str, int]) -> None:
    """One line per session: the canceller's work and the echo the barge-in rules met."""
    stats = reference.stats
    logger.info(
        "echo summary",
        extra={
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
            **counters,
        },
    )

"""A model of the client's playout of our output audio (ARCHITECTURE §6.5, D17).

The echo canceller's reference is our own output placed where the client plays it. The client
is modelled as AICC plays a call's audio (ai-native-callcenter voice/rtp.go ``Run``): our audio
arrives as 20 ms frames, and a ticker sends one frame per tick:

- while playing, each tick plays the next queued frame; an empty queue plays silence, and more
  than ``underrun_ticks`` empty ticks in a row stop playback (re-buffer);
- while stopped, a tick starts playback once ``prebuffer_frames`` are queued, or once at least one
  is queued and ``prebuffer`` has passed since the last tick that found the queue empty.

All times are integer sample counts at 16 kHz on one monotonic clock, so arithmetic is exact.
Frames are opaque handles. The ticker's phase is unknown to the gateway: ``TickPhaseLearner``
fits it from where the echo of each playback run was observed (M0 S11: a model that is one tick
off at a playback start makes AEC3 re-acquire its delay and leak echo).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayoutParams:
    tick: int  # samples per tick (320: 20 ms)
    prebuffer_frames: int  # frames queued before playback starts (3)
    prebuffer: int  # samples after an empty tick when fewer frames start anyway (960: 60 ms)
    underrun_ticks: int  # empty ticks tolerated while playing (2)


@dataclass(frozen=True, slots=True)
class Played[T]:
    handle: T
    start: int  # the tick that played it; the frame sounds over [start, start + tick)


@dataclass(frozen=True, slots=True)
class RunStart:
    """Playback started: at tick ``start``, triggered by the frame that arrived at
    ``trigger`` (``by_count``: the prebuffer filled; else its deadline passed)."""

    start: int
    trigger: int
    by_count: bool


@dataclass(slots=True)
class _Queued[T]:
    handle: T
    arrival: int


def ceil_tick(time: int, phase: int, tick: int) -> int:
    """The first tick at or after ``time`` on the grid ``phase + k * tick``."""
    return phase + -((phase - time) // tick) * tick


class PlayoutModel[T]:
    """The client's playout of the frames it receives (see the module notes)."""

    def __init__(
        self,
        params: PlayoutParams,
        *,
        on_run_start: Callable[[RunStart], None] | None = None,
    ) -> None:
        self.params = params
        self.phase = 0
        self._pending_phase: int | None = None
        self._queue: deque[_Queued[T]] = deque()
        self._last_tick: int | None = None
        self._last_arrival: int | None = None
        self._playing = False
        self._underruns = 0
        self._deadline: int | None = None
        self._on_run_start = on_run_start

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def last_tick(self) -> int | None:
        return self._last_tick

    def set_phase(self, phase: int) -> None:
        """Move the tick grid; it applies once playback is idle (never inside a run)."""
        self._pending_phase = phase % self.params.tick
        self._apply_pending_phase()

    def arrive(self, handles: Sequence[T], at: int) -> list[Played[T]]:
        """Frames that reached the client at ``at``, in order; returns frames played before."""
        played = self.advance(at - 1)
        if self._last_tick is not None:
            at = max(at, self._last_tick + 1)  # the model already ran past it: arrive next tick
        if self._last_arrival is not None:
            at = max(at, self._last_arrival)
        self._last_arrival = at
        self._queue.extend(_Queued(handle, at) for handle in handles)
        return played

    def advance(self, until: int) -> list[Played[T]]:
        """Run every tick at or before ``until``; returns the frames those ticks played."""
        played: list[Played[T]] = []
        while True:
            if self._idle():
                self._skip_idle(until)
                self._apply_pending_phase()
            following = self._next_tick()
            if following > until:
                return played
            frame = self._tick(following)
            self._last_tick = following
            if frame is not None:
                played.append(Played(frame, following))

    def drop(self, keep: Callable[[T], bool] = lambda _: False) -> list[T]:
        """Remove queued frames the client discards (a flush); returns them."""
        dropped = [entry.handle for entry in self._queue if not keep(entry.handle)]
        self._queue = deque(entry for entry in self._queue if keep(entry.handle))
        return dropped

    def queue_end(self) -> int | None:
        """While playing: the tick after the last queued frame if playback ran on unbroken."""
        if not self._playing or self._last_tick is None:
            return None
        return self._last_tick + (len(self._queue) + 1) * self.params.tick

    def _idle(self) -> bool:
        return not self._playing and not self._queue

    def _next_tick(self) -> int:
        tick = self.params.tick
        if self._last_tick is None:
            return self.phase
        return ceil_tick(self._last_tick + 1, self.phase, tick)

    def _skip_idle(self, until: int) -> None:
        """Idle ticks change nothing but the prebuffer deadline: jump to the last one."""
        following = self._next_tick()
        if following > until:
            return
        last = following + (until - following) // self.params.tick * self.params.tick
        self._last_tick = last
        self._deadline = last + self.params.prebuffer

    def _apply_pending_phase(self) -> None:
        """The next tick is then the first on the new grid after the last one."""
        if self._pending_phase is None or not self._idle():
            return
        self.phase, self._pending_phase = self._pending_phase, None

    def _tick(self, now: int) -> T | None:
        params = self.params
        arrived = 0
        for entry in self._queue:
            if entry.arrival > now:
                break
            arrived += 1
        if self._playing:
            if arrived:
                self._underruns = 0
                return self._queue.popleft().handle
            self._underruns += 1
            if self._underruns > params.underrun_ticks:
                self._playing, self._underruns = False, 0
            return None
        by_count = arrived >= params.prebuffer_frames
        if by_count or (arrived and self._deadline is not None and now >= self._deadline):
            self._playing = True
            if self._on_run_start is not None:
                index = params.prebuffer_frames - 1 if by_count else 0
                trigger = self._queue[min(index, arrived - 1)].arrival
                self._on_run_start(RunStart(now, trigger, by_count))
            return self._queue.popleft().handle
        if not arrived:
            self._deadline = now + params.prebuffer
        return None


class TickPhaseLearner:
    """Fits the client's tick phase from playback runs whose echo was located (M0 S11).

    For each run started by a full prebuffer, the gateway knows when the triggering frame was
    sent (``trigger``) and, from the delay estimator, the sample where that run's echo really
    began less a constant (the residual delay, the lead and the echo path): ``observed``. With
    the client's phase ``p``, a run starts at ``ceil_tick(trigger, p)``, so
    ``observed - ceil_tick(trigger, p)`` is the same for every run. Each candidate phase is
    scored by how many runs agree on that constant within ``tolerance``; the phase is the middle
    of the best-scoring arc, once at least ``min_runs`` runs agree and they are at least
    ``min_agreement`` of the runs seen. The phase absorbs the mean transit to the client.
    """

    def __init__(
        self,
        tick: int,
        *,
        tolerance: int = 64,
        step: int = 8,
        min_runs: int = 3,
        min_agreement: float = 0.75,
        max_runs: int = 24,
    ) -> None:
        self._tick = tick
        self._tolerance = tolerance
        self._step = step
        self._min_runs = min_runs
        self._min_agreement = min_agreement
        self._runs: deque[tuple[int, int]] = deque(maxlen=max_runs)

    @property
    def runs(self) -> int:
        return len(self._runs)

    def observe(self, trigger: int, observed: int) -> None:
        self._runs.append((trigger, observed))

    def phase(self) -> int | None:
        """The fitted phase, or None while the runs do not determine one."""
        if len(self._runs) < self._min_runs:
            return None
        candidates = range(0, self._tick, self._step)
        scores = [self._agreeing(phase) for phase in candidates]
        best = max(scores)
        if best < self._min_runs or best < self._min_agreement * len(self._runs):
            return None
        # The best candidates form arcs on the circle of phases; take the longest one's middle.
        winners = [score == best for score in scores]
        arc = _longest_circular_run(winners)
        if arc is None:
            return None
        first, length = arc
        middle = (first + (length - 1) / 2) * self._step
        return round(middle) % self._tick

    def _agreeing(self, phase: int) -> int:
        offsets = sorted(
            observed - ceil_tick(trigger, phase, self._tick) for trigger, observed in self._runs
        )
        best = 0
        low = 0
        for high, value in enumerate(offsets):
            while value - offsets[low] > 2 * self._tolerance:
                low += 1
            best = max(best, high - low + 1)
        return best


def _longest_circular_run(flags: Sequence[bool]) -> tuple[int, int] | None:
    """The start and length of the longest run of True on a circle."""
    count = len(flags)
    if not any(flags):
        return None
    if all(flags):
        return 0, count
    best: tuple[int, int] | None = None
    start = next(index for index in range(count) if not flags[index]) + 1
    length = 0
    for offset in range(count):
        index = (start + offset) % count
        if flags[index]:
            length += 1
            run_start = (index - length + 1) % count
            if best is None or length > best[1]:
                best = (run_start, length)
        else:
            length = 0
    return best

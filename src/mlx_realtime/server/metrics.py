"""Metrics (``/metrics``, Prometheus text format) and the turn records of ``/debug/turns``
(ARCHITECTURE §9).

Every series has bounded labels: a fixed stage, status, backend, error kind or refusal reason,
never an id. Each gateway owns its registry, so gateways in one process (the tests) never share
counters.

- Histograms: ``mlxrt_stage_seconds{stage}`` per stage of the turn timeline, ``mlxrt_e2e_seconds``
  (``speech_end`` → ``first_audio_sent``), ``mlxrt_tts_queue_wait_seconds`` (a response's first
  segment's wait for the TTS slot).
- Counters: sessions, refused sessions by reason, responses by status, barge-ins (and those that
  turned out empty or echo), echo-suppressed turns, provider errors by backend and kind, TTS
  fallback uses.
- Gauges: active sessions, backend readiness (probed at each scrape), the TTS queue depth, and
  the windowed P50/P95 of each stage and of end to end over the last
  ``observability.quantile_window_turns`` turns: the Python client's ``Summary`` has no
  quantiles, so they are computed here. The histograms stay the aggregatable source.

Stage metrics count only turns that answer user audio (a response's own timeline, a verbatim
line's for example, has no turn to measure).
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import CONTENT_TYPE_PLAIN_0_0_4, generate_latest
from prometheus_client.metrics_core import GaugeMetricFamily, Metric

from mlx_realtime.config import ObservabilityConfig
from mlx_realtime.session.observer import TurnReport
from mlx_realtime.session.timeline import STAGES, TurnTimeline

CONTENT_TYPE = CONTENT_TYPE_PLAIN_0_0_4

STAGE_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
)  # fmt: skip
E2E_BUCKETS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0)
QUEUE_BUCKETS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
E2E = "e2e"


def quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile with linear interpolation between order statistics (numpy's default)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class WindowedQuantiles:
    """The last ``window`` observations of each series, and their P50/P95."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._series: dict[str, deque[float]] = {}

    def observe(self, series: str, value: float) -> None:
        self._series.setdefault(series, deque(maxlen=self._window)).append(value)

    def quantiles(self, series: str) -> tuple[float, float] | None:
        values = self._series.get(series)
        if not values:
            return None
        return quantile(values, 0.5), quantile(values, 0.95)

    def series(self) -> list[str]:
        return list(self._series)


def turn_record(report: TurnReport) -> dict[str, object]:
    """One turn as ``/debug/turns`` lists it: marks in ms after the earliest, stage durations
    and end to end in ms, and the values logged with the timeline."""
    timeline = report.timeline
    e2e = timeline.e2e_s()
    return {
        "turn_id": timeline.turn_id,
        "item_id": timeline.item_id,
        "response_id": report.response_id,
        "status": report.status,
        "marks_ms": timeline.offsets_ms(),
        "stages_ms": {name: _ms(value) for name, value in timeline.stages_s().items()},
        "e2e_ms": None if e2e is None else _ms(e2e),
        "values": timeline.values,
    }


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class DebugTurns:
    """The turn records of the ``max_sessions`` most recent sessions, at most ``max_turns``
    each (oldest dropped first)."""

    def __init__(self, max_sessions: int, max_turns: int) -> None:
        self._max_sessions = max_sessions
        self._max_turns = max_turns
        self._sessions: OrderedDict[str, deque[dict[str, object]]] = OrderedDict()

    def add(self, report: TurnReport) -> None:
        turns = self._sessions.get(report.session_id)
        if turns is None:
            turns = self._sessions[report.session_id] = deque(maxlen=self._max_turns)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        turns.append(turn_record(report))

    def session(self, session_id: str) -> list[dict[str, object]] | None:
        turns = self._sessions.get(session_id)
        return None if turns is None else list(turns)

    def sessions(self) -> list[str]:
        return list(self._sessions)


class Metrics:
    """The gateway's metrics and debug records; also each session's ``SessionObserver``.

    ``active_sessions`` and ``tts_queue_depth`` are read at each scrape.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        *,
        active_sessions: Callable[[], int] = lambda: 0,
        tts_queue_depth: Callable[[], int] = lambda: 0,
    ) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.debug = DebugTurns(config.debug_sessions, config.debug_turns_per_session)
        self._debug_enabled = config.debug_endpoints
        self._windows = WindowedQuantiles(config.quantile_window_turns)
        self._active_sessions = active_sessions
        self._tts_queue_depth = tts_queue_depth
        registry = self.registry
        self._stage = Histogram(
            "mlxrt_stage_seconds",
            "Duration of each stage of the turn timeline.",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=registry,
        )
        self._e2e = Histogram(
            "mlxrt_e2e_seconds",
            "End of the caller's speech to the first audio delta sent.",
            buckets=E2E_BUCKETS,
            registry=registry,
        )
        self._tts_wait = Histogram(
            "mlxrt_tts_queue_wait_seconds",
            "A response's first segment's wait for the TTS backend's stream slot.",
            buckets=QUEUE_BUCKETS,
            registry=registry,
        )
        self._sessions = Counter("mlxrt_sessions", "Sessions admitted.", registry=registry)
        self._refused = Counter(
            "mlxrt_sessions_refused", "Sessions refused, by reason.", ["reason"], registry=registry
        )
        self._responses = Counter(
            "mlxrt_responses", "Responses finished, by status.", ["status"], registry=registry
        )
        self._barge_ins = Counter(
            "mlxrt_barge_ins", "Responses interrupted by the caller's speech.", registry=registry
        )
        self._barge_ins_empty = Counter(
            "mlxrt_barge_ins_empty_or_echo",
            "Barge-ins whose turn transcribed empty or as our own echo.",
            registry=registry,
        )
        self._echo_turns = Counter(
            "mlxrt_echo_turns",
            "Turns suppressed as the echo of our own speech (D17).",
            registry=registry,
        )
        self._provider_errors = Counter(
            "mlxrt_provider_errors",
            "Provider errors, by backend and kind (absorbed TTS primary failures included).",
            ["backend", "kind"],
            registry=registry,
        )
        self._tts_fallback = Counter(
            "mlxrt_tts_fallback", "Segments the fallback TTS model spoke.", registry=registry
        )
        self._ready = Gauge(
            "mlxrt_backend_ready",
            "Whether each backend reported ready at the last scrape (1) or not (0).",
            ["backend"],
            registry=registry,
        )
        registry.register(_ScrapeTime(self))

    # SessionObserver

    def turn_finished(self, report: TurnReport) -> None:
        timeline = report.timeline
        if self._debug_enabled:
            self.debug.add(report)
        if timeline.item_id is None:
            return  # no user audio turn: nothing to measure
        for stage, seconds in timeline.stages_s().items():
            self._stage.labels(stage).observe(seconds)
            self._windows.observe(stage, seconds)
        e2e = timeline.e2e_s()
        if e2e is not None:
            self._e2e.observe(e2e)
            self._windows.observe(E2E, e2e)
        self._observe_queue_wait(timeline)

    def response_finished(self, status: str) -> None:
        self._responses.labels(status).inc()

    def barge_in(self) -> None:
        self._barge_ins.inc()

    def barge_in_empty_or_echo(self) -> None:
        self._barge_ins_empty.inc()

    def echo_turn(self) -> None:
        self._echo_turns.inc()

    def provider_error(self, backend: str, kind: str) -> None:
        self._provider_errors.labels(backend, kind).inc()

    def tts_fallback(self) -> None:
        self._tts_fallback.inc()

    # The gateway

    def session_started(self) -> None:
        self._sessions.inc()

    def session_refused(self, reason: str) -> None:
        self._refused.labels(reason).inc()

    def set_readiness(self, backends: Mapping[str, Mapping[str, object]]) -> None:
        for name, detail in backends.items():
            self._ready.labels(name).set(1.0 if detail.get("ready") else 0.0)

    def exposition(self) -> bytes:
        return generate_latest(self.registry)

    def windowed(self) -> dict[str, tuple[float, float]]:
        """Each series' P50 and P95 over its window, in seconds."""
        found: dict[str, tuple[float, float]] = {}
        for series in self._windows.series():
            pair = self._windows.quantiles(series)
            if pair is not None:
                found[series] = pair
        return found

    def _observe_queue_wait(self, timeline: TurnTimeline) -> None:
        waited = timeline.values.get("tts_queue_ms")
        if isinstance(waited, int | float):
            self._tts_wait.observe(float(waited) / 1000)

    def scrape_gauges(self) -> Iterable[Metric]:
        """The gauges computed when ``/metrics`` is scraped."""
        active = GaugeMetricFamily("mlxrt_active_sessions", "Sessions open now.")
        active.add_metric([], float(self._active_sessions()))
        depth = GaugeMetricFamily(
            "mlxrt_tts_queue_depth", "TTS requests waiting for the backend's stream slot now."
        )
        depth.add_metric([], float(self._tts_queue_depth()))
        p50 = GaugeMetricFamily(
            "mlxrt_stage_p50_seconds",
            "Median of each stage over the last N turns (observability.quantile_window_turns).",
            labels=["stage"],
        )
        p95 = GaugeMetricFamily(
            "mlxrt_stage_p95_seconds",
            "95th percentile of each stage over the last N turns.",
            labels=["stage"],
        )
        e2e50 = GaugeMetricFamily(
            "mlxrt_e2e_p50_seconds", "Median end to end over the last N turns."
        )
        e2e95 = GaugeMetricFamily(
            "mlxrt_e2e_p95_seconds", "95th percentile of end to end over the last N turns."
        )
        windowed = self.windowed()
        for stage in STAGES:
            if stage in windowed:
                p50.add_metric([stage], windowed[stage][0])
                p95.add_metric([stage], windowed[stage][1])
        if E2E in windowed:
            e2e50.add_metric([], windowed[E2E][0])
            e2e95.add_metric([], windowed[E2E][1])
        return [active, depth, p50, p95, e2e50, e2e95]


class _ScrapeTime:
    """The gauges read when ``/metrics`` is scraped."""

    def __init__(self, metrics: Metrics) -> None:
        self._metrics = metrics

    def collect(self) -> Iterable[Metric]:
        return self._metrics.scrape_gauges()

    def describe(self) -> Iterable[Metric]:
        return self.collect()

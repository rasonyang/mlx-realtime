"""The benchmark report: per-stage and end-to-end P50/P95, and ASR error rates (docs/LATENCY.md).

Latency, over the turns that were answered with audio:

- the server's stages from ``/debug/turns`` (each named after the mark that ends it, ARCHITECTURE
  §9) and its end to end, ``first_audio_sent − speech_end``;
- the client's end to end: from when it sent the end of the speech (trailing silence excluded) to
  the first ``response.output_audio.delta`` it received.

ASR: the transcripts of ``conversation.item.input_audio_transcription.completed``, scored per set
(``zh`` CER, ``en`` WER, ``mix`` MER) and pooled over all rows as MER (CJK characters plus Latin
words); rows with numbers apart, since references keep digits that speakers say as words.

AICC's windows, over every response of the run (ARCHITECTURE §2, Deadlines), from the client's
arrival times: AICC's watchdog arms 3 s at ``response.created``, re-arms 2 s at every audio delta,
and is disarmed by ``response.done``; a turn it fires on is abandoned. So the first audio delta
must come within 3 s of ``response.created``, no two deltas (nor the last delta and
``response.done``) may be 2 s apart, and a response without audio (tool-only) must be done
within 3 s.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, cast

from bench.scoring import KIND_FOR_SET, Edits, Kind, error, pooled
from mlx_realtime.server.metrics import quantile
from mlx_realtime.session.timeline import STAGES

type Outcome = Literal["answered", "no_audio", "no_turn", "timeout"]

E2E_SERVER = "e2e_server"
E2E_CLIENT = "e2e_client"
FIRST_AUDIO_LIMIT_MS = 3000.0  # response.created -> the first audio delta
DELTA_GAP_LIMIT_MS = 2000.0  # between two audio deltas, and from the last one to response.done
NO_AUDIO_LIMIT_MS = 3000.0  # response.created -> response.done of a response without audio


@dataclass
class ResponseTiming:
    """One response as AICC's watchdog sees it: client arrival times, in ms."""

    response_id: str
    status: str | None = None  # None: no response.done before the utterance was given up
    first_audio_ms: float | None = None  # response.created -> the first audio delta
    max_gap_ms: float | None = None  # the longest wait between two of its audio deltas
    tail_ms: float | None = None  # its last audio delta -> response.done
    done_ms: float | None = None  # response.created -> response.done


@dataclass
class TurnResult:
    """One utterance as the benchmark saw it."""

    utterance_id: str
    set: str
    has_number: bool
    text_norm: str
    outcome: Outcome
    session_id: str = ""
    items: list[str] = field(default_factory=list[str])  # the user audio items it committed
    transcript: str = ""
    client_e2e_ms: float | None = None
    server_stages_ms: dict[str, float] = field(default_factory=dict[str, float])
    server_e2e_ms: float | None = None
    response_status: str | None = None
    duration_ms: int = 0
    responses: list[ResponseTiming] = field(default_factory=list[ResponseTiming])

    @property
    def split(self) -> bool:
        """The VAD cut the utterance into more than one turn."""
        return len(self.items) > 1


@dataclass(frozen=True)
class Stat:
    n: int
    p50_ms: float | None
    p95_ms: float | None


def stat(values: Sequence[float]) -> Stat:
    if not values:
        return Stat(0, None, None)
    return Stat(len(values), round(quantile(values, 0.5), 1), round(quantile(values, 0.95), 1))


@dataclass(frozen=True)
class Window:
    """One of AICC's windows over a run: the distribution and how many reached the limit."""

    limit_ms: float
    n: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    over: int  # values at or past the limit: AICC would have abandoned those turns


def window(values: Sequence[float], limit_ms: float) -> Window:
    summary = stat(values)
    worst = round(max(values), 1) if values else None
    over = sum(value >= limit_ms for value in values)
    return Window(limit_ms, summary.n, summary.p50_ms, summary.p95_ms, worst, over)


def windows(turns: Sequence[TurnResult]) -> dict[str, object]:
    """AICC's windows over every response of ``turns`` (the module's docstring)."""
    responses = [response for turn in turns for response in turn.responses]
    spoken = [response for response in responses if response.first_audio_ms is not None]
    silent = [
        response
        for response in responses
        if response.first_audio_ms is None and response.done_ms is not None
    ]
    statuses = Counter(response.status or "unfinished" for response in responses)
    first_audio = [r.first_audio_ms for r in spoken if r.first_audio_ms is not None]
    gaps = [r.max_gap_ms for r in spoken if r.max_gap_ms is not None]
    tails = [r.tail_ms for r in spoken if r.tail_ms is not None]
    done = [r.done_ms for r in silent if r.done_ms is not None]
    return {
        "responses": len(responses),
        "statuses": dict(sorted(statuses.items())),
        "first_audio": asdict(window(first_audio, FIRST_AUDIO_LIMIT_MS)),
        "delta_gap": asdict(window(gaps, DELTA_GAP_LIMIT_MS)),
        "tail": asdict(window(tails, DELTA_GAP_LIMIT_MS)),
        "no_audio_done": asdict(window(done, NO_AUDIO_LIMIT_MS)),
    }


@dataclass(frozen=True)
class Score:
    kind: Kind
    utterances: int
    ref_tokens: int
    edits: int
    sub: int
    dels: int
    ins: int
    rate: float | None


def score(turns: Sequence[TurnResult], kind: Kind) -> Score:
    total = pooled(error(turn.text_norm, turn.transcript, kind) for turn in turns)
    return _score(kind, len(turns), total)


def _score(kind: Kind, count: int, total: Edits) -> Score:
    rate = None if total.rate is None else round(total.rate, 4)
    return Score(kind, count, total.ref, total.edits, total.sub, total.dels, total.ins, rate)


def summarize(
    turns: Sequence[TurnResult], *, meta: dict[str, object] | None = None
) -> dict[str, object]:
    """The report as JSON-ready data: ``meta``, ``counts``, ``latency``, ``asr`` and ``turns``."""
    answered = [turn for turn in turns if turn.outcome == "answered"]
    latency: dict[str, Stat] = {}
    for stage in STAGES:
        latency[stage] = stat(
            [turn.server_stages_ms[stage] for turn in answered if stage in turn.server_stages_ms]
        )
    latency[E2E_SERVER] = stat(
        [turn.server_e2e_ms for turn in answered if turn.server_e2e_ms is not None]
    )
    latency[E2E_CLIENT] = stat(
        [turn.client_e2e_ms for turn in answered if turn.client_e2e_ms is not None]
    )
    transcribed = [turn for turn in turns if turn.outcome != "no_turn"]
    asr: dict[str, dict[str, Score]] = {}
    for name in sorted({turn.set for turn in transcribed}):
        rows = [turn for turn in transcribed if turn.set == name]
        kind = KIND_FOR_SET.get(name, "mer")
        asr[name] = _by_numbers(rows, kind)
    asr["all"] = _by_numbers(transcribed, "mer")
    counts = {
        "utterances": len(turns),
        **{outcome: sum(turn.outcome == outcome for turn in turns) for outcome in _OUTCOMES},
        "split_turns": sum(turn.split for turn in turns),
    }
    return {
        "meta": meta or {},
        "counts": counts,
        "latency": {name: asdict(value) for name, value in latency.items()},
        "windows": windows(turns),
        "asr": {
            name: {group: asdict(value) for group, value in groups.items()}
            for name, groups in asr.items()
        },
        "turns": [{**asdict(turn), "split": turn.split} for turn in turns],
    }


_OUTCOMES: tuple[Outcome, ...] = ("answered", "no_audio", "no_turn", "timeout")


def _by_numbers(rows: Sequence[TurnResult], kind: Kind) -> dict[str, Score]:
    return {
        "without_numbers": score([row for row in rows if not row.has_number], kind),
        "with_numbers": score([row for row in rows if row.has_number], kind),
        "all": score(rows, kind),
    }


def markdown(report: dict[str, object]) -> str:
    """The report's tables (docs/LATENCY.md's template)."""
    meta = _mapping(report["meta"])
    counts = _mapping(report["counts"])
    latency = _mapping(report["latency"])
    asr = _mapping(report["asr"])
    lines = ["## Run", ""]
    lines += [f"- {key}: {value}" for key, value in meta.items()]
    lines += [
        "- utterances: {utterances} (answered {answered}, answered without audio {no_audio}, "
        "no turn detected {no_turn}, timed out {timeout}; split into several turns: "
        "{split_turns})".format(**counts),
        "",
        "## Latency (ms)",
        "",
        "| stage | from | n | P50 | P95 |",
        "|---|---|---|---|---|",
    ]
    for stage, starts in STAGES.items():
        lines.append(_latency_row(stage, " / ".join(starts), _mapping(latency[stage])))
    lines.append(
        _latency_row("**end to end (server)**", "speech_end", _mapping(latency[E2E_SERVER]))
    )
    lines.append(
        _latency_row("**end to end (client)**", "speech sent", _mapping(latency[E2E_CLIENT]))
    )
    if "windows" in report:
        lines += _windows_lines(_mapping(report["windows"]))
    lines += [
        "",
        "## ASR",
        "",
        "| set | metric | rows without numbers | rows with numbers | all rows |",
        "|---|---|---|---|---|",
    ]
    for name, groups in asr.items():
        cells = {group: _mapping(value) for group, value in _mapping(groups).items()}
        kind = str(cells["all"]["kind"]).upper()
        lines.append(
            f"| {name} | {kind} | {_rate(cells['without_numbers'])} "
            f"| {_rate(cells['with_numbers'])} | {_rate(cells['all'])} |"
        )
    return "\n".join(lines) + "\n"


_WINDOWS = {
    "first_audio": "response.created → first audio delta",
    "delta_gap": "longest gap between audio deltas",
    "tail": "last audio delta → response.done",
    "no_audio_done": "response.created → response.done, no audio",
}


def _windows_lines(windows: dict[str, object]) -> list[str]:
    statuses = _mapping(windows["statuses"])
    listed = ", ".join(f"{status} {count}" for status, count in statuses.items())
    lines = [
        "",
        "## AICC windows (ms, client arrival times)",
        "",
        f"- responses: {windows['responses']} ({listed or 'none'})",
        "",
        "| window | limit | n | P50 | P95 | max | at or over the limit |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, name in _WINDOWS.items():
        value = _mapping(windows[key])
        lines.append(
            f"| {name} | {_ms(value['limit_ms'])} | {value['n']} | {_ms(value['p50_ms'])} "
            f"| {_ms(value['p95_ms'])} | {_ms(value['max_ms'])} | {value['over']} |"
        )
    return lines


def _latency_row(name: str, start: str, value: dict[str, object]) -> str:
    return f"| {name} | {start} | {value['n']} | {_ms(value['p50_ms'])} | {_ms(value['p95_ms'])} |"


def _ms(value: object) -> str:
    return "–" if value is None else f"{value:.0f}" if isinstance(value, float) else str(value)


def _rate(score: dict[str, object]) -> str:
    rate = score["rate"]
    if not isinstance(rate, float):
        return "–"
    return f"{rate:.2%} ({score['edits']}/{score['ref_tokens']}, n={score['utterances']})"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in cast(dict[object, object], value).items()}

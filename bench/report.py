"""The benchmark report: per-stage and end-to-end P50/P95, and ASR error rates (docs/LATENCY.md).

Latency, over the turns that were answered with audio:

- the server's stages from ``/debug/turns`` (each named after the mark that ends it, ARCHITECTURE
  §9) and its end to end, ``first_audio_sent − speech_end``;
- the client's end to end: from when it sent the end of the speech (trailing silence excluded) to
  the first ``response.output_audio.delta`` it received.

ASR: the transcripts of ``conversation.item.input_audio_transcription.completed``, scored per set
(``zh`` CER, ``en`` WER, ``mix`` MER) and pooled over all rows as MER (CJK characters plus Latin
words); rows with numbers apart, since references keep digits that speakers say as words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, cast

from bench.scoring import KIND_FOR_SET, Edits, Kind, error, pooled
from mlx_realtime.server.metrics import quantile
from mlx_realtime.session.timeline import STAGES

type Outcome = Literal["answered", "no_audio", "no_turn", "timeout"]

E2E_SERVER = "e2e_server"
E2E_CLIENT = "e2e_client"


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

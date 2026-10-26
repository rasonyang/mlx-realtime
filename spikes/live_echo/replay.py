"""Replay the echo matcher on captured turns (offline evaluation of echo rules).

    uv run --locked python spikes/live_echo/replay.py <capture dir>... [--threshold 0.7]
        [--min-units 4] [--window-ms 3000] [--tail-ms 450] [--rebuild]

Run it in the gateway's environment: it imports ``mlx_realtime.text.similarity``, the matcher the
gateway uses. For every ``transcript`` event of a capture it recomputes the echo-match score and
the decision for the given rule, next to what the gateway recorded:

- by default from the ``played_text`` the gateway compared against (a rule that only changes the
  threshold, the minimum length, or how a score is used);
- with ``--rebuild`` from the capture's ``assistant_text`` and ``play_start`` / ``play_end``
  events: the assistant text of each response (segments on the item's audio timeline) mapped to
  the input timeline through the response's modelled playback, over the turn's audio plus
  ``--window-ms`` before it and ``--tail-ms`` of echo tail (what the gateway's ``played_spans``
  does), so the window itself can be varied.

The result lists each turn with its transcript, the speech onset (``audio_start_ms``), whether
the gateway called it echo, whether it began a barge-in (``barge_in_outcome``), and the replayed
score and verdict. ``replay(events, ...)`` returns the same rows for use from another script,
for example to evaluate an ASR-confirmation rule: a barge-in turn whose transcript matches what
was playing is one the rule would not have confirmed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlx_realtime.text.similarity import overlap, units


@dataclass(slots=True)
class TurnResult:
    item: str
    transcript: str
    audio_start_ms: float
    audio_end_ms: float
    gateway_score: float | None
    gateway_echo: bool
    barge_in_outcome: str | None
    played_text: str
    score: float | None
    echo: bool


def load(directory: Path) -> list[dict]:
    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return sorted((e for e in events if e.get("t") is not None), key=lambda e: e["t"])


def rebuilt_played_text(
    events: list[dict], start_ms: float, end_ms: float, window_ms: float, tail_ms: float
) -> str:
    """The assistant text modelled as playing in [start - window - tail, end]."""
    low, high = start_ms - window_ms - tail_ms, end_ms
    text = {e["item"]: e["segments"] for e in events if e["kind"] == "assistant_text"}
    starts = {e["item"]: e["t"] for e in events if e["kind"] == "play_start"}
    ends = {e["item"]: e["t"] for e in events if e["kind"] == "play_end"}
    played = ""
    for item, begin in sorted(starts.items(), key=lambda kv: kv[1]):
        finish = ends.get(item, float("inf"))
        if finish < low or begin > high or item not in text:
            continue
        first, last = max(low, begin) - begin, min(high, finish) - begin  # on the item's audio timeline
        played += "".join(
            segment["text"]
            for segment in text[item]
            if segment["start_ms"] < last and segment["start_ms"] + segment["audio_ms"] > first
        )
    return played


def replay(
    events: list[dict],
    *,
    threshold: float = 0.7,
    min_units: int = 4,
    window_ms: float = 3000,
    tail_ms: float = 450,
    rebuild: bool = False,
) -> list[TurnResult]:
    outcomes = {e["item"]: e["outcome"] for e in events if e["kind"] == "barge_in_outcome"}
    rows: list[TurnResult] = []
    for event in (e for e in events if e["kind"] == "transcript"):
        text = event.get("text") or ""
        played = (
            rebuilt_played_text(events, event["audio_start_ms"], event["audio_end_ms"], window_ms, tail_ms)
            if rebuild
            else event.get("played_text", "")
        )
        score = overlap(text, played) if played and len(units(text)) >= min_units else None
        rows.append(
            TurnResult(
                item=event["item"],
                transcript=text,
                audio_start_ms=event["audio_start_ms"],
                audio_end_ms=event["audio_end_ms"],
                gateway_score=event.get("echo_score"),
                gateway_echo=bool(event.get("echo")),
                barge_in_outcome=outcomes.get(event["item"]),
                played_text=played,
                score=score,
                echo=score is not None and score >= threshold,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-units", type=int, default=4)
    parser.add_argument("--window-ms", type=float, default=3000)
    parser.add_argument("--tail-ms", type=float, default=450)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    for directory in args.captures:
        rows = replay(
            load(directory),
            threshold=args.threshold,
            min_units=args.min_units,
            window_ms=args.window_ms,
            tail_ms=args.tail_ms,
            rebuild=args.rebuild,
        )
        if args.json:
            print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))
            continue
        print(f"{directory.name}: {len(rows)} turns")
        for r in rows:
            score = "  n/a" if r.score is None else f"{r.score:5.2f}"
            gateway = "  n/a" if r.gateway_score is None else f"{r.gateway_score:5.2f}"
            print(
                f"  {r.audio_start_ms / 1000:7.2f} s  replay {score} {'ECHO' if r.echo else '    '}"
                f"  gateway {gateway} {'ECHO' if r.gateway_echo else '    '}"
                f"  barge-in {r.barge_in_outcome or '-':5}  {r.transcript[:60]!r}"
            )


if __name__ == "__main__":
    main()

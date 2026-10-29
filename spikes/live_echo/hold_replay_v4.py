"""Replay of the v4 rule over the v3 measurement captures (hold_v3, synthetic leaks A/Am and real
callers B/Be/C/F, real ASR). Writes nothing; prints the counts that results/hold_v4_replay.md reports.

v4: on a complete-utterance check (pause or final) whose correlation was computed (psr not null)
and is below confirm_echo_psr (15), the transcript's echo verdict does not stand and the turn is
released, at the pause or at its end, instead of being held or dropped.

From the v3 `hold_check` events (rows/*-fix.json, capture.items[].checks):
- final check, verdict "echo", psr < 15: v3 dropped the turn; v4 releases it at its end;
- utterance check with >= 4 units, verdict "hold", psr < 15: only the echo rule holds such a check
  (a caller verdict would have released it, and "echo_signal" is a different verdict), so v4
  releases at the pause (earlier than the end);
- psr null (no bypass, too short, ...): unchanged.
For a leak (A, Am) the release is a potential self-interruption; the script says whether the
client was still modelled as playing at the check (play_start / play_end in the capture's
events.jsonl), when the speech_started would reach a client that flushes its queue.

    uv run --locked python spikes/live_echo/hold_replay_v4.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ROOT = HERE / "captures" / "hold_v3"
PSR_LEVEL = 15.0
MIN_UNITS = 4
LEAKS, CALLERS = ("A", "Am"), ("B", "Be", "C", "F")


def playing_at(events: list[dict[str, Any]], t: float) -> bool:
    on = False
    for event in events:
        if event["t"] is None or event["t"] > t:
            break
        if event["kind"] == "play_start":
            on = True
        elif event["kind"] in ("play_end", "flush", "truncate"):
            on = False
    return on


def events_of(directory: str) -> list[dict[str, Any]]:
    path = ROOT / "fix" / directory / "events.jsonl"
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted((e for e in events if e.get("t") is not None), key=lambda e: e["t"])


def changed(item: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """('final'|'pause', check) when v4 changes this turn's outcome."""
    drop = item.get("hold_drop")
    for check in item["checks"]:
        psr = check["psr"]
        if psr is None or psr >= PSR_LEVEL:
            continue
        if check["check"] == "utterance" and check["verdict"] == "hold" and (check["units"] or 0) >= MIN_UNITS:
            return "pause", check
        if check["check"] == "final" and check["verdict"] == "echo" and drop is not None:
            return "final", check
    return None


def main() -> None:
    seen: Counter[str] = Counter()
    drops: dict[str, Counter[str]] = {}
    hits: dict[str, list[str]] = {}
    for path in sorted((ROOT / "rows").glob("*-fix.json")):
        for row in json.loads(path.read_text())["rows"]:
            scenario = row["scenario"]
            if "error" in row or "capture" not in row:
                continue
            events = events_of(row["capture_dir"])
            for item in row["capture"]["items"]:
                if not item["held"]:
                    continue
                seen[scenario] += 1
                if "hold_drop" in item:
                    finals = [c for c in item["checks"] if c["check"] == "final"]
                    psr = finals[-1]["psr"] if finals else None
                    bucket = "no_psr" if psr is None else "psr>=15" if psr >= PSR_LEVEL else "psr<15"
                    drops.setdefault(scenario, Counter())[f"{item['hold_drop']['reason']} {bucket}"] += 1
                hit = changed(item)
                if hit is None:
                    continue
                where, check = hit
                text = (check["text"] or "")[:30]
                state = playing_at(events, check["t"]) if events else None
                hits.setdefault(scenario, []).append(
                    f"{row['id']} {where} psr={check['psr']} units={check['units']} "
                    f"unmatched={check['unmatched']} playing={state} text={text!r} "
                    f"v3={'dropped' if 'hold_drop' in item else 'released'}"
                )
    for group, label in ((LEAKS, "leaks"), (CALLERS, "real callers")):
        total = sum(seen[s] for s in group)
        flagged = sum(len(hits.get(s, [])) for s in group)
        print(f"{label}: {flagged} of {total} held turns change outcome")
        for scenario in group:
            print(f"  {scenario}: held turns {seen[scenario]}, changed {len(hits.get(scenario, []))}")
            print(f"    v3 drops (reason, final psr): {dict(drops.get(scenario, {}))}")
            for line in hits.get(scenario, []):
                print("    " + line)


if __name__ == "__main__":
    main()

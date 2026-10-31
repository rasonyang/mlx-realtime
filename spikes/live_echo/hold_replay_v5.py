"""Replay of the v5 echo-turn rule over every recorded hold: the live calls of 2026-10-29/30 and the
synthetic v3 captures. Writes nothing; prints the markdown of results/hold_v5_replay.md.

v5 (issue #9): a live line's echo correlated as weakly as callers, so #7's low-PSR release is
removed and the echo-turn rule scores the transcript by contiguity (the share of its units in runs of
3 or more that also run on in the played text) instead of the in-order overlap.

The replay needs no models. Each `hold_check` event of a capture's events.jsonl carries the ASR text,
the assistant text played around the turn, the correlation (`psr`) and the verdict of the build that
recorded it. A turn is walked check by check under three rules:

- v3: overlap rule, no correlation override (the build before #7);
- v4: overlap rule, and on a complete-utterance check with a computed psr below confirm_echo_psr
  (15) a transcript echo is released instead (#7);
- v5: contiguity rule, no override (this change).

Partial checks are taken as recorded (a verdict `caller` or `echo_signal` means the unmatched units
reached the release count; the unmatched count is unchanged by v5). The correlation veto
(confirm_echo_psr, confirm_echo_drop_psr) is applied the way the gateway does, and also switched off
to count how often it decided anything.

    uv run --locked python spikes/live_echo/hold_replay_v5.py > spikes/live_echo/results/hold_v5_replay.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mlx_realtime.config import BargeInConfig
from mlx_realtime.text.similarity import contiguity, overlap, units, unmatched

HERE = Path(__file__).parent
CAPTURES = HERE / "captures"
CONFIG = BargeInConfig()
PSR = CONFIG.confirm_echo_psr
DROP_PSR = CONFIG.confirm_echo_drop_psr
LIVE = {
    "9sxh4merv8ojrnry": "call 1 (10-29)",
    "t77351ifjdmbgfrb": "call 2 (10-29)",
    "y978thy92hi5k7xr": "V1 (10-30)",
    "x54ly59awo9bn7tg": "V2 (10-30)",
    "bnblfvhsgir6mt70": "C6 (10-30)",
}
DECISIVE = {"caller", "release", "echo", "noise", "echo_signal_drop", "failed"}


def load(directory: Path) -> list[dict[str, Any]]:
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def verdict(rule: str, check: dict[str, Any], *, veto: bool) -> str:
    """The hold's verdict on one recorded check under ``rule`` (v3, v4 or v5)."""
    if check["verdict"] == "failed" or check.get("text") is None:
        return "failed"
    kind = check["check"]
    final = kind == "final"
    text, played, psr = check["text"], check["played_text"] or "", check["psr"]
    count = len(units(text))
    if kind == "partial":
        base = "caller" if check["verdict"] in ("caller", "echo_signal") else "hold"
    else:
        score = contiguity(text, played) if rule == "v5" else overlap(text, played)
        echo = count >= CONFIG.echo_match_min_units and bool(played) and score >= CONFIG.echo_match_threshold
        if count == 0:
            base = "noise" if final else "hold"
        elif count >= CONFIG.echo_match_min_units:
            if echo and rule == "v4" and psr is not None and PSR is not None and psr < PSR:
                base = "release"
            else:
                base = ("echo" if final else "hold") if echo else "caller"
        elif unmatched(text, played) >= 1:
            base = "caller"
        else:
            base = "caller" if final else "hold"
    if veto and base == "caller" and psr is not None and PSR is not None and psr >= PSR:
        if not final:
            return "echo_signal"
        if DROP_PSR is not None and psr >= DROP_PSR:
            return "echo_signal_drop"
    return base


def walk(rule: str, checks: list[dict[str, Any]], *, veto: bool = True) -> tuple[str, int | None]:
    """(outcome, index of the deciding check): outcome is released, dropped or undecided."""
    for index, check in enumerate(checks):
        v = verdict(rule, check, veto=veto)
        if v in DECISIVE:
            return ("dropped" if v in ("echo", "noise", "echo_signal_drop") else "released"), index
    return "undecided", None


def scores(check: dict[str, Any]) -> tuple[float, float]:
    text, played = check.get("text") or "", check.get("played_text") or ""
    return overlap(text, played), contiguity(text, played)


def sessions() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for path in sorted((CAPTURES / "live").glob("2026*")):
        found.append(("live", LIVE.get(path.name.split("_sess_")[1], path.name), path))
    scenario_of: dict[str, str] = {}
    for rows in sorted((CAPTURES / "hold_v3" / "rows").glob("*-fix.json")):
        data = json.loads(rows.read_text(encoding="utf-8"))
        for row in data["rows"]:
            if "capture_dir" in row:
                scenario_of[row["capture_dir"]] = rows.name.split("-")[0]
    for path in sorted((CAPTURES / "hold_v3" / "fix").glob("2026*")):
        found.append(("v3", scenario_of.get(path.name, "?"), path))
    return found


def turns(directory: Path) -> list[dict[str, Any]]:
    events = load(directory)
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["kind"] == "hold_check":
            by_item[event["item"]].append(event)
    recorded: dict[str, str] = {}
    for event in events:
        if event["kind"] == "hold_drop":
            recorded[event["item"]] = "dropped " + str(event.get("reason"))
        elif event["kind"] == "hold_release":
            recorded[event["item"]] = "released " + str(event.get("reason"))
    return [
        {"item": item, "checks": checks, "recorded": recorded.get(item, "-")}
        for item, checks in by_item.items()
    ]


def final_check(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [c for c in checks if c["check"] != "partial" and c.get("text")]
    return complete[-1] if complete else None


def main() -> None:
    live_rows: list[str] = []
    changes: list[str] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    veto: dict[str, Counter[str]] = defaultdict(Counter)
    for kind, label, directory in sessions():
        for turn in turns(directory):
            checks = turn["checks"]
            outcomes = {rule: walk(rule, checks) for rule in ("v3", "v4", "v5")}
            group = f"{kind}:{label}"
            counts[group]["held turns"] += 1
            for rule, (outcome, _) in outcomes.items():
                counts[group][f"{rule} {outcome}"] += 1
            if outcomes["v5"][0] != outcomes["v3"][0]:
                counts[group]["v5 differs from v3"] += 1
            if outcomes["v5"][0] != outcomes["v4"][0]:
                counts[group]["v5 differs from v4"] += 1
            # How often the correlation veto decided anything, with v5's rule.
            decided = 0
            for check in checks:
                if check["psr"] is not None and check["verdict"] != "failed":
                    veto[group]["checks with a psr"] += 1
                    if check["psr"] >= (PSR or 0):
                        veto[group]["checks at or over confirm_echo_psr"] += 1
                if verdict("v5", check, veto=True) != verdict("v5", check, veto=False):
                    veto[group]["checks it changed"] += 1
                    decided += 1
            with_veto, without = walk("v5", checks), walk("v5", checks, veto=False)
            if with_veto != without:
                veto[group]["turns it changed"] += 1
            veto[group]["turns"] += 1
            last = final_check(checks)
            if last is None:
                continue
            ov, ct = scores(last)
            text = (last["text"] or "").replace("|", "/")[:26]
            row = (
                f"| {label} | {turn['item'][-4:]} | {text} | {last['units']} | {last['psr']} | "
                f"{ov:.2f} | {ct:.2f} | {turn['recorded']} | "
                + " | ".join(f"{o[0]}{'' if o[1] is None else f' @{o[1]}'}" for o in outcomes.values())
                + " |"
            )
            if kind == "live":
                live_rows.append(row)
            if outcomes["v5"][0] != outcomes["v3"][0] or outcomes["v5"] != outcomes["v4"]:
                changes.append(row)
    head = "| capture | item | transcript | units | psr | overlap | contiguity | recorded | v3 | v4 | v5 |"
    rule = "|---|---|---|---|---|---|---|---|---|---|---|"
    print("## Turns of the live calls (complete-utterance check, or the last one)\n")
    print("\n".join([head, rule, *live_rows]))
    print("\nOutcome cells: released / dropped / undecided, `@n` the index of the deciding check.\n")
    print("## Turns where v5 differs from v3 or v4 (all captures)\n")
    print("\n".join([head, rule, *changes]) if changes else "none")
    print("\n## Outcomes per capture group\n")
    keys = [
        "held turns",
        "v3 released",
        "v3 dropped",
        "v3 undecided",
        "v4 released",
        "v4 dropped",
        "v4 undecided",
        "v5 released",
        "v5 dropped",
        "v5 undecided",
        "v5 differs from v3",
        "v5 differs from v4",
    ]
    print("| group | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for group, counter in counts.items():
        print(f"| {group} | " + " | ".join(str(counter[key]) for key in keys) + " |")
    print("\n## Correlation veto (confirm_echo_psr 15, confirm_echo_drop_psr 15), under v5\n")
    vkeys = [
        "turns",
        "checks with a psr",
        "checks at or over confirm_echo_psr",
        "checks it changed",
        "turns it changed",
    ]
    print("| group | " + " | ".join(vkeys) + " |")
    print("|---|" + "---|" * len(vkeys))
    for group, counter in veto.items():
        print(f"| {group} | " + " | ".join(str(counter[key]) for key in vkeys) + " |")


if __name__ == "__main__":
    main()

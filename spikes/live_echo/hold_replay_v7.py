"""Replay of the v7 partial-check rule over every recorded hold: the live calls of 2026-10-29/30 and
the synthetic v3 captures. Writes nothing; prints the markdown of results/hold_v7_replay.md.

v7 (issue #10): a hold that already matched our words (a check of it was all ours, `MIN_RUN` units
or more) releases at a partial check that reaches the release count only when `confirm_min_new_units`
of the units that are not ours lie before the transcript's last unit (the cut edge, which the cut
may have truncated into another word), or when the previous partial check reached the count too and
was held. A fresh hold keeps the v6 rule. The mirror of `SessionActor._judge`'s partial branch.

The replay walks each hold's recorded checks. A recorded partial check with verdict `caller` or
`echo_signal` reached the release count (`unmatched >= need`; the replay cannot tell whether the
client still played). The sequences stop at the recording build's own decision, so a release v7
delays to a check that was never recorded cannot be walked further: it is listed as "needs ASR",
and its delay is one check interval at least.

    uv run --locked python spikes/live_echo/hold_replay_v7.py > spikes/live_echo/results/hold_v7_replay.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import hold_replay_v6 as v6
from hold_replay_v5 import CONFIG, DECISIVE, PSR, sessions, turns

from mlx_realtime.text.similarity import MIN_RUN, units, unmatched, unmatched_inside

REACHED = ("caller", "echo_signal")


def walk7(checks: list[dict[str, Any]]) -> tuple[str, int | None, str]:
    """(outcome, index of the deciding check, status): status is `recorded`, or `needs ASR` when
    v7 holds at the last recorded check, which the recording build released."""
    matched = suspect = False
    hold_start = checks[0]["audio_start_ms"]  # a partial check's window starts here until it slides
    for index, check in enumerate(checks):
        text, played = check.get("text") or "", check.get("played_text") or ""
        if check["check"] == "partial" and check["verdict"] in ("hold", *REACHED):
            reached = check["verdict"] in REACHED
            slid = check["audio_start_ms"] > hold_start
            firm = not matched or unmatched_inside(text, played, slid=slid) >= (
                CONFIG.confirm_min_new_units
            )
            v = "caller" if reached and (firm or suspect) else "hold"
            suspect = reached and v == "hold"
            if v == "caller" and check["psr"] is not None and check["psr"] >= PSR:
                v = "echo_signal"  # the correlation veto, as v5 applies it
            if suspect and index == len(checks) - 1:
                return "undecided", None, "needs ASR"
        else:
            v = v6.verdict6(check)
        if text:
            matched = matched or (len(units(text)) >= MIN_RUN and unmatched(text, played) == 0)
        if v in DECISIVE:
            outcome = "dropped" if v in ("echo", "noise", "echo_signal_drop") else "released"
            return outcome, index, "recorded"
    return "undecided", None, "recorded"


def short(text: str, width: int = 30) -> str:
    return text.replace("|", "/")[:width]


def main() -> None:
    rows: list[str] = []
    needs: list[str] = []
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for kind, label, directory in sessions():
        group = f"{kind}:{label}"
        for turn in turns(directory):
            checks = turn["checks"]
            old, new = v6.walk6(checks), walk7(checks)
            totals[group]["held turns"] += 1
            totals[group][f"v7 {new[0]}"] += 1
            totals[group][f"v6 {old[0]}"] += 1
            totals[group]["outcome changed"] += old[0] != new[0]
            if old[1] is None or new[1] is None and new[2] != "needs ASR":
                continue
            at = checks[old[1]]
            delay = "-"
            if new[1] is not None and new[1] != old[1]:
                delay = f"{checks[new[1]]['t'] - at['t']:.0f}"
                totals[group]["delayed"] += 1
                totals[group]["max delay ms"] = max(
                    totals[group]["max delay ms"], round(float(delay))
                )
            if new[2] == "needs ASR" or new[1] != old[1] or old[0] != new[0]:
                inside = unmatched_inside(
                    at["text"] or "",
                    at["played_text"] or "",
                    slid=at["audio_start_ms"] > checks[0]["audio_start_ms"],
                )
                cell = (
                    f"| {group} | {turn['item'][-4:]} | {turn['recorded']} | "
                    f"{old[0]} @{old[1]} | {new[0]}{'' if new[1] is None else f' @{new[1]}'} "
                    f"({new[2]}) | {delay} | {short(at['text'] or '')} | {at['unmatched']} | "
                    f"{inside} |"
                )
                (needs if new[2] == "needs ASR" else rows).append(cell)
    head = (
        "| capture | item | recorded | v6 | v7 | delay ms | v6 deciding check | unmatched | "
        "inside the cut edges |\n|---|---|---|---|---|---|---|---|---|"
    )
    print("## Holds v7 decides at another check, from the recorded checks\n")
    print(head)
    print("\n".join(rows) if rows else "| none | | | | | | | | |")
    print("\n## Holds whose release v7 defers to a check that was never recorded\n")
    print(head)
    print("\n".join(needs) if needs else "| none | | | | | | | | |")
    keys = [
        "held turns",
        "v6 released",
        "v6 dropped",
        "v7 released",
        "v7 dropped",
        "v7 undecided",
        "outcome changed",
        "delayed",
        "max delay ms",
    ]
    print("\n## Outcomes per capture group\n")
    print("| group | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for group, counter in totals.items():
        print(f"| {group} | " + " | ".join(str(counter[key]) for key in keys) + " |")


if __name__ == "__main__":
    main()

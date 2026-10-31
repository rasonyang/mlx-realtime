"""Replay of the v6 echo-turn rule over every recorded turn: the live calls of 2026-10-29/30 and the
synthetic v3 captures. Writes nothing; prints the markdown of results/hold_v6_replay.md.

v6 (issue #8): a caller who starts talking while echo is on the line lands in the same VAD turn as
the echo, and the whole turn used to be dropped as echo. v6 cuts the echo prefix off the transcript
(`text.similarity.after_echo`: the leading runs of 3 or more units that are contiguous in the played
text, joined across gaps shorter than 3 units) and judges the remainder:

- a held turn (a complete-utterance or final check the echo-turn rule calls echo): a caller when the
  remainder has `echo_match_min_units` (4) units, its own contiguity is below the threshold, and
  `confirm_min_new_units` (2) of its units are not ours (the replay cannot tell whether the client
  still played at the check, which would make it 1); the turn is then released with the remainder
  as its transcript;
- an unheld turn (the `transcript` event of a turn the gateway announced, whether released by a
  partial check or never held): a caller when the remainder has 4 units and its contiguity is below
  the threshold; the transcript becomes the remainder.

The two mirror `SessionActor._judge` and `_after_echo`, as `hold_replay_v5.py` mirrors the v5 rule.

    uv run --locked python spikes/live_echo/hold_replay_v6.py > spikes/live_echo/results/hold_v6_replay.md
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import hold_replay_v5 as v5
from hold_replay_v5 import CONFIG, DECISIVE, DROP_PSR, PSR, load, sessions, turns

from mlx_realtime.text.similarity import after_echo, contiguity, units, unmatched


def remainder(text: str, played: str, *, held: bool) -> str | None:
    """The mirror of `SessionActor._after_echo`."""
    rest = after_echo(text, played)
    if (
        rest == text
        or len(units(rest)) < CONFIG.echo_match_min_units
        or contiguity(rest, played) >= CONFIG.echo_match_threshold
    ):
        return None
    return rest if not held or unmatched(rest, played) >= CONFIG.confirm_min_new_units else None


def is_echo(text: str, played: str) -> bool:
    return (
        len(units(text)) >= CONFIG.echo_match_min_units
        and bool(played)
        and contiguity(text, played) >= CONFIG.echo_match_threshold
    )


def verdict6(check: dict[str, Any]) -> str:
    """v5's verdict on a check, except that an echo with a caller after it is a caller."""
    if (
        check["check"] != "partial"
        and check.get("text")
        and v5.verdict("v5", check, veto=True)
        in (
            "echo",
            "hold",
            "echo_signal_drop",
        )
    ):
        text, played = check["text"], check["played_text"] or ""
        if is_echo(text, played) and remainder(text, played, held=True) is not None:
            base = "caller"
            psr = check["psr"]
            if psr is not None and PSR is not None and psr >= PSR:
                if check["check"] != "final":
                    return "echo_signal"
                if DROP_PSR is not None and psr >= DROP_PSR:
                    return "echo_signal_drop"
            return base
    return v5.verdict("v5", check, veto=True)


def walk6(checks: list[dict[str, Any]]) -> tuple[str, int | None]:
    for index, check in enumerate(checks):
        v = verdict6(check)
        if v in DECISIVE:
            return ("dropped" if v in ("echo", "noise", "echo_signal_drop") else "released"), index
    return "undecided", None


def short(text: str, width: int = 40) -> str:
    return text.replace("|", "/")[:width]


def main() -> None:
    held_rows: list[str] = []
    transcript_rows: list[str] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for kind, label, directory in sessions():
        group = f"{kind}:{label}"
        events = load(directory)
        for turn in turns(directory):
            checks = turn["checks"]
            old, new = v5.walk("v5", checks), walk6(checks)
            counts[group]["held turns"] += 1
            counts[group][f"v6 {new[0]}"] += 1
            counts[group][f"v5 {old[0]}"] += 1
            last = v5.final_check(checks)
            cut = None
            if last is not None and is_echo(last["text"], last["played_text"] or ""):
                cut = remainder(last["text"], last["played_text"] or "", held=True)
            if new[0] != old[0]:
                counts[group]["held turns whose outcome changed"] += 1
                held_rows.append(
                    f"| {label} | {turn['item'][-4:]} | {short(last['text'] if last else '')} | "
                    f"{turn['recorded']} | {old[0]} | {new[0]} | {short(cut or '')} |"
                )
        for event in events:
            if event["kind"] != "transcript" or not event["detected"]:
                continue
            text, played = event["text"], event["played_text"] or ""
            counts[group]["transcripts of detected turns"] += 1
            old_echo = is_echo(text, played)
            rest = remainder(text, played, held=False) if old_echo else None
            counts[group]["transcripts echo (v5)"] += old_echo
            counts[group]["transcripts echo (v6)"] += old_echo and rest is None
            if rest is not None:
                counts[group]["transcripts made caller turns"] += 1
                transcript_rows.append(
                    f"| {label} | {event['item'][-4:]} | {event['audio_start_ms'] / 1000:.2f}-"
                    f"{event['audio_end_ms'] / 1000:.2f} | {short(text, 60)} | {event['units']} | "
                    f"{contiguity(text, played):.2f} | {short(rest)} | {len(units(rest))} | "
                    f"{contiguity(rest, played):.2f} |"
                )
    print("## Unheld turns that become caller turns (`transcript` events)\n")
    head = (
        "| capture | item | input s | heard | units | contiguity | remainder | units | contiguity |"
    )
    print("\n".join([head, "|---|---|---|---|---|---|---|---|---|", *transcript_rows]) or "none")
    print("\n## Held turns whose outcome changes\n")
    if held_rows:
        print("| capture | item | final check | recorded | v5 | v6 | remainder |")
        print("|---|---|---|---|---|---|---|")
        print("\n".join(held_rows))
    else:
        print("none")
    keys = [
        "held turns",
        "v5 released",
        "v5 dropped",
        "v5 undecided",
        "v6 released",
        "v6 dropped",
        "v6 undecided",
        "held turns whose outcome changed",
        "transcripts of detected turns",
        "transcripts echo (v5)",
        "transcripts echo (v6)",
        "transcripts made caller turns",
    ]
    print("\n## Outcomes per capture group\n")
    print("| group | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for group, counter in counts.items():
        print(f"| {group} | " + " | ".join(str(counter[key]) for key in keys) + " |")


if __name__ == "__main__":
    main()

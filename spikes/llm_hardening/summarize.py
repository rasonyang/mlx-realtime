"""Tables from ``spikes/results/llm_hardening_<name>.json`` (run.py), for docs/PROGRESS.md.

    uv run --project spikes/llm_hardening python spikes/llm_hardening/summarize.py baseline warm ...

Per case and run: the first segment boundary of spoken replies (the gateway's
``first_sentence_ready`` after ``response_created``) and ``response.done`` of tool-only replies
(client side), p50 / p95 in s, with n; the prompt tokens the LLM had to prefill (p50). First
audio is estimated as M0 measured its other stages: ASR 0.3-0.5 s when a caller turn created the
response (the gateway creates it at the commit, before the transcript), plus the boundary, plus
the TTS's first chunk (0.12 s). A tool-only reply to a caller turn adds the ASR time to its done.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
TTS_FIRST_CHUNK = 0.12
ASR = (0.3, 0.5)
# Which cases answer a caller turn (ASR inside AICC's window) and which a response.create.
CALLER_TURN = {
    "cold", "warm", "change_turn", "change_turn_next", "change_transfer", "warm_transfer",
    "aicc_next_spoken", "aicc_next_transfer", "override_next", "transfer_contended_1",
    "transfer_contended_2", "interleaved_4",
}  # fmt: skip
ORDER = [
    "line", "cold", "warm", "change_turn", "change_turn_next", "change_transfer", "warm_transfer",
    "aicc_lookup", "aicc_result", "aicc_next_spoken", "aicc_next_transfer", "override",
    "override_next", "transfer_contended_1", "transfer_contended_2", "interleaved_4",
]  # fmt: skip


def fmt(stat: dict, key: str = "p50") -> str:
    return "–" if not stat or stat.get("n", 0) == 0 else f"{stat[key]:.2f}"


def pair(stat: dict) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "–"
    return f"{stat['p50']:.2f} / {stat['p95']:.2f} ({stat['n']})"


def first_audio(case: str, boundary: dict) -> str:
    if not boundary or boundary.get("n", 0) == 0:
        return "–"
    p95 = boundary["p95"] + TTS_FIRST_CHUNK
    if case in CALLER_TURN:
        return f"{p95 + ASR[0]:.2f}–{p95 + ASR[1]:.2f}"
    return f"{p95:.2f}"


def tool_done(case: str, done: dict) -> str:
    if not done or done.get("n", 0) == 0:
        return "–"
    if case in CALLER_TURN:
        return f"{done['p95'] + ASR[0]:.2f}–{done['p95'] + ASR[1]:.2f}"
    return f"{done['p95']:.2f}"


def main() -> None:
    names = sys.argv[1:]
    runs = {name: json.loads((RESULTS / f"llm_hardening_{name}.json").read_text()) for name in names}
    print("| case | run | boundary p50 / p95 (n) | est. first audio p95 | tool-only done p50 / p95 (n)"
          " | est. tool-only p95 | prefilled tokens p50 |")  # fmt: skip
    print("|---|---|---|---|---|---|---|")
    for case in ORDER:
        for name, run in runs.items():
            stat = run["stats"].get(case)
            if stat is None:
                continue
            print(
                f"| {case} | {name} | {pair(stat['boundary'])} | {first_audio(case, stat['boundary'])}"
                f" | {pair(stat['tool_only_done'])} | {tool_done(case, stat['tool_only_done'])}"
                f" | {fmt(stat['uncached_tokens'])} |"
            )
    for name, run in runs.items():
        c = run["conditions"]
        print(f"\n{name}: warm={c['warm_on_update']} flags={' '.join(c['server_flags'][10:])} "
              f"footprint {c['footprint_gb_start']}→{c['footprint_gb_end']} GB, cache {c['prompt_cache']}, "
              f"warmups {run['warmups']}, load {c['env_before']['loadavg'][0]:.1f}→"
              f"{c['env_after']['loadavg'][0]:.1f}, wall {c['wall_s']} s")  # fmt: skip


if __name__ == "__main__":
    main()

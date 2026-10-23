"""Combine bench reports into the LATENCY.md tables.

    python3 -I spikes/realstack/bench_tables.py RUN [RUN ...]
    (RUN: a bench/results/<name> directory, or spikes/results/realstack_bench/<name>.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = ["speech_stopped", "turn_confirmed", "asr_done", "llm_first_token",
          "first_sentence_ready", "tts_first_chunk", "first_audio_sent"]

def report(path: Path) -> dict:
    """A bench report: a directory with report.json, or the JSON itself (spikes/results)."""
    return json.loads((path / "report.json" if path.is_dir() else path).read_text())


runs = [(Path(p).stem.removeprefix("real-"), report(Path(p))) for p in sys.argv[1:]]


def ms(v: object) -> str:
    return "–" if v is None else f"{v:.0f}"


def pct(score: dict) -> str:
    r = score["rate"]
    return "–" if r is None else f"{100 * r:.1f}%"


print("### Counts\n")
print("| run | utterances | answered | no audio | no turn | timeout | split |")
print("|---|---|---|---|---|---|---|")
for name, rep in runs:
    c = rep["counts"]
    print(f"| {name} | {c['utterances']} | {c['answered']} | {c['no_audio']} | {c['no_turn']} | {c['timeout']} | {c['split_turns']} |")

print("\n### Stages, P50 / P95 (ms)\n")
print("| stage | " + " | ".join(name for name, _ in runs) + " |")
print("|---|" + "---|" * len(runs))
for stage in [*STAGES, "e2e_server", "e2e_client"]:
    cells = []
    for _, rep in runs:
        s = rep["latency"][stage]
        cells.append(f"{ms(s['p50_ms'])} / {ms(s['p95_ms'])}")
    print(f"| {stage} | " + " | ".join(cells) + " |")

def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


print("\n### End to end (server) without Smart Turn holds, P50 / P95 (ms)\n")
print("Turns whose `turn_confirmed` stage took over 1 s are the ones Smart Turn held as incomplete")
print("until `max_silence_ms`; the rest ended on a complete verdict.\n")
print("| | " + " | ".join(name for name, _ in runs) + " |")
print("|---|" + "---|" * len(runs))
cells, held = [], []
for _, rep in runs:
    answered = [t for t in rep["turns"] if t["outcome"] == "answered" and t["server_e2e_ms"]]
    fast = [t["server_e2e_ms"] for t in answered
            if t["server_stages_ms"].get("turn_confirmed", 0) <= 1000]
    held.append(str(len(answered) - len(fast)))
    cells.append(f"{quantile(fast, 0.5):.0f} / {quantile(fast, 0.95):.0f} (n={len(fast)})")
print("| e2e, complete verdicts | " + " | ".join(cells) + " |")
print("| turns held | " + " | ".join(held) + " |")

print("\n### AICC windows: P50 / P95 / max (ms), count at or over the limit\n")
print("| window | " + " | ".join(name for name, _ in runs) + " |")
print("|---|" + "---|" * len(runs))
for key in ["first_audio", "delta_gap", "tail", "no_audio_done"]:
    cells = []
    for _, rep in runs:
        w = rep.get("windows", {}).get(key)
        if not w:
            cells.append("–")
            continue
        cells.append(f"{ms(w['p50_ms'])} / {ms(w['p95_ms'])} / {ms(w['max_ms'])} ({w['over']}/{w['n']})")
    print(f"| {key} | " + " | ".join(cells) + " |")
print("\n| responses | " + " | ".join(
    json.dumps(rep.get("windows", {}).get("statuses", {})) for _, rep in runs) + " |")

print("\n### ASR (all rows / rows without numbers)\n")
print("| set | " + " | ".join(name for name, _ in runs) + " |")
print("|---|" + "---|" * len(runs))
for key, label in [("zh", "zh CER"), ("en", "en WER"), ("mix", "mix MER"), ("all", "all MER")]:
    cells = []
    for _, rep in runs:
        a = rep["asr"].get(key)
        cells.append("–" if a is None else f"{pct(a['all'])} / {pct(a['without_numbers'])}")
    print(f"| {label} | " + " | ".join(cells) + " |")

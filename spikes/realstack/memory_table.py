"""The soak's memory samples (memsample.py's JSON lines) as a markdown table: per-process
footprint (and its peak) in GB at each sample, plus free memory and swap.

    python3 -I spikes/realstack/memory_table.py spikes/results/realstack_soak_memory.jsonl
"""

from __future__ import annotations

import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
names = ["llm", "tts", "asr", "gateway"]
print("| time | " + " | ".join(f"{name} (peak)" for name in names) + " | free | swap used |")
print("|---|" + "---|" * (len(names) + 2))
for row in rows:
    cells = []
    for name in names:
        proc = row.get(name)
        cells.append("–" if proc is None else f"{proc['footprint_gb']:.2f} ({proc['peak_gb']:.2f})")
    system = row["system"]
    swap = system["swap"].split("used = ")[1].split()[0] if "used = " in system["swap"] else "?"
    print(f"| {row['t']} | " + " | ".join(cells) + f" | {system['free_gb']} GB | {swap} |")

"""ASR time against utterance length from the ``sweep`` runs.

    uv run --locked --project spikes/transcription python spikes/transcription/sweep_report.py spikes/results/transcription_sweep_alone.json [LOADED.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

import common as c

for path in sys.argv[1:]:
    d = json.load(open(path))
    by = defaultdict(list)
    for r in d["rows"]:
        by[r["target_s"]].append(r)
    print(f"## {path} (load sessions: {d['load']})")
    print("target s | n | audio s (min-max) | commit->completed ms (P50 / max) | CER")
    for t, rs in sorted(by.items()):
        lat = [r["latency_ms"] for r in rs]
        ref = sum(r["ref_chars"] for r in rs)
        err = sum(r["cer"] * r["ref_chars"] for r in rs)
        print(f"{t:3} | {len(rs)} | {min(r['audio_s'] for r in rs):.1f}-{max(r['audio_s'] for r in rs):.1f} | "
              f"{statistics.median(lat):.0f} / {max(lat):.0f} | {100 * err / max(ref, 1):.1f}% | errors {sum(bool(r['error']) for r in rs)}")

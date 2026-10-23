"""Summarize aiccflow.py rows per label: spoken first audio / tool-only done (AICC's 3 s windows),
the longest delta gap and the tail (2 s)."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def q(values: list[float]) -> str:
    if not values:
        return "–"
    v = sorted(values)
    p95 = v[max(0, min(len(v) - 1, round(0.95 * len(v) + 0.5) - 1))]
    return f"n={len(v)} p50={statistics.median(v):.0f} p95={p95:.0f} max={v[-1]:.0f}"


for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print(f"== {path}")
    by = defaultdict(list)
    for row in rows:
        by[row["label"]].append(row)
    for label, items in by.items():
        spoken = [r for r in items if r["first_audio_ms"] is not None]
        tool_only = [r for r in items if r["tool_only"]]
        other = [r for r in items if r["first_audio_ms"] is None and not r["tool_only"]]
        calls = sorted({c[0] for r in items for c in r["calls"]})
        statuses = sorted({r["status"] for r in items})
        print(f"  {label}: {len(items)} responses {statuses} calls={calls}")
        if spoken:
            print(f"    spoken first audio: {q([r['first_audio_ms'] for r in spoken])}; "
                  f"max gap: {q([r['max_gap_ms'] for r in spoken if r['max_gap_ms'] is not None])}; "
                  f"tail: {q([r['tail_ms'] for r in spoken if r['tail_ms'] is not None])}")
        if tool_only:
            print(f"    tool-only done: {q([r['done_ms'] for r in tool_only])}")
        if other:
            print(f"    no audio, no call: {len(other)} {[r['done_ms'] for r in other]}")
        over = [r for r in items if (r["first_audio_ms"] or 0) >= 3000 or (r["tool_only"] and (r["done_ms"] or 0) >= 3000)
                or (r["max_gap_ms"] or 0) >= 2000 or (r["tail_ms"] or 0) >= 2000]
        if over:
            print(f"    OVER AICC WINDOWS: {len(over)}: {[(r['first_audio_ms'], r['done_ms'], r['max_gap_ms'], r['tail_ms']) for r in over]}")

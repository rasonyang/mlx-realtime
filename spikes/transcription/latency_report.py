"""Tables from the latency runs: commit-to-completed (client side) per condition, the bots' first audio
and their ASR stage time (the gateway's /debug/turns).

    uv run --locked --project spikes/transcription python spikes/transcription/latency_report.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import httpx2

import common as c


def q(v: list[float]) -> str:
    if not v:
        return "-"
    return f"{statistics.median(v):.0f} / {c.percentile(v, 0.95):.0f} / {max(v):.0f} (n={len(v)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", default="clean")
    a = ap.parse_args()
    print("## commit -> completed (ms, P50 / P95 / max)")
    for cond in ("1", "2", "4"):
        path = c.RESULTS / f"transcription_latency_{a.tag}_{cond}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        lat = [s["latency_ms"] for leg in d["legs"].values() for s in leg["segments"] if s["latency_ms"] is not None]
        failed = sum(bool(s.get("failed")) for leg in d["legs"].values() for s in leg["segments"])
        ended = [n for n, leg in d["legs"].items() if leg["ended_by"] != "audio_done"]
        long = [s["latency_ms"] for leg in d["legs"].values() for s in leg["segments"] if s["latency_ms"] and s["commit_t"] - s["speech_t"] > 8]
        print(f"sessions={cond}: {q(lat)}; segments > 8 s: {q(long)}; failed={failed}; ended early={ended}")
    print("\n## bot responses (aiccflow rows)")
    with httpx2.Client(base_url=a.base, timeout=10, trust_env=False) as http:
        for cond in ("bots2", "4"):
            rows_path = c.RESULTS / f"transcription_latency_{a.tag}_{cond}.bots.jsonl"
            if not rows_path.exists():
                continue
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            by: dict[str, list[dict]] = {}
            for r in rows:
                by.setdefault(r["label"], []).append(r)
            asr: list[float] = []
            for sid in {r["session"] for r in rows}:
                resp = http.get("/debug/turns", params={"session_id": sid})
                if resp.status_code == 200:
                    asr += [t["stages_ms"]["asr_done"] for t in resp.json()["turns"] if t.get("stages_ms", {}).get("asr_done") is not None]
            print(f"{cond}: responses={len(rows)} statuses={sorted({r['status'] for r in rows})}")
            for label, items in by.items():
                spoken = [r["first_audio_ms"] for r in items if r["first_audio_ms"] is not None]
                gaps = [r["max_gap_ms"] for r in items if r["max_gap_ms"] is not None]
                print(f"  {label:12} first audio {q(spoken)}; max gap {q(gaps)}")
            allspoken = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None and r["label"] != "announce"]
            print(f"  turns (not announce) first audio {q(allspoken)}; asr_done stage {q(asr)}")


if __name__ == "__main__":
    main()

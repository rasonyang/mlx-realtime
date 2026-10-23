"""Find sweep requests perturbed by another process's brief GPU use (B2, S3).

At about 01:52:30-01:54 local time on 2026-10-23 another agent's test started mlx_lm.server (possibly
loading weights) and tried to start mlx_audio servers for at most ~15 s. The sweep records hold no
wall-clock time, but its server's tee'd log stamps every access line with perf_counter_ns
(mach_absolute_time), and requests were strictly sequential: access line k is request k (3 warmups,
then the records in order). perf_counter is converted to wall time with the current offset
(time.time_ns() - time.perf_counter_ns()), valid while the machine has not slept since.

Writes spikes/results/s3_sweep_perturbation.json: wall time per request, every request in the window
with its TTFC and total time relative to its cell's median, the largest outliers of the whole run,
and the `replace` list (excluded requests, to be re-run with `s3_tts.py sweep --topup`).

    uv run --locked --project spikes/audio python spikes/audio/s3_sweep_check.py [--start 01:52:00 --end 01:54:45]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import time
from pathlib import Path

import b2_common as c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="s3_sweep.json")
    ap.add_argument("--date", default="2026-10-23")
    ap.add_argument("--start", default="01:52:00")
    ap.add_argument("--end", default="01:54:45")
    a = ap.parse_args()
    d = json.loads((c.RESULTS / a.sweep).read_text())
    log = Path(d["servers"][0]["log"])
    posts = [int(line.split(" ", 1)[0]) for line in log.read_text().splitlines()
             if '"POST /v1/audio/speech' in line]
    recs = d["records"]
    n_warm = len(d["warmup"])
    assert len(posts) == n_warm + len(recs), (len(posts), n_warm, len(recs))
    offset = time.time_ns() - time.perf_counter_ns()
    tz = dt.datetime.now().astimezone().tzinfo

    def wall(ns: int) -> dt.datetime:
        return dt.datetime.fromtimestamp((ns + offset) / 1e9, tz)

    started = dt.datetime.fromisoformat(d["started_at"])
    first_post = wall(posts[0])
    w0 = dt.datetime.fromisoformat(f"{a.date}T{a.start}").replace(tzinfo=tz)
    w1 = dt.datetime.fromisoformat(f"{a.date}T{a.end}").replace(tzinfo=tz)

    def cell(r):
        return (r["interval"], r["lang"], r["length"], r["speaker"])

    meas = [r for r in recs if r["phase"] == "measured" and r.get("ttfc_ms")]
    by_cell: dict = {}
    for r in meas:
        by_cell.setdefault(cell(r), []).append(r)

    rows = []
    for k, r in enumerate(recs):
        t = wall(posts[n_warm + k])
        peers = [x for x in by_cell.get(cell(r), []) if x is not r]
        med_t = statistics.median([x["ttfc_ms"] for x in peers]) if peers else None
        med_total = statistics.median([x["total_ms"] for x in peers]) if peers else None
        rows.append({"k": k, "wall": t.strftime("%H:%M:%S.%f")[:-3], "phase": r["phase"], "round": r["round"],
                     "cell": "|".join(map(str, cell(r))), "text_i": r["text_i"], "ttfc_ms": r["ttfc_ms"],
                     "ttfc_ratio": round(r["ttfc_ms"] / med_t, 2) if med_t and r["ttfc_ms"] else None,
                     "total_ratio": round(r["total_ms"] / med_total, 2) if med_total and r["total_ms"] else None,
                     "in_window": w0 <= t <= w1})
    window = [x for x in rows if x["in_window"]]
    outliers = sorted([x for x in rows if x["phase"] == "measured" and x["ttfc_ratio"]],
                      key=lambda x: -x["ttfc_ratio"])[:15]
    replace = [{"interval": recs[x["k"]]["interval"], "lang": recs[x["k"]]["lang"],
                "length": recs[x["k"]]["length"], "speaker": recs[x["k"]]["speaker"],
                "text_i": recs[x["k"]]["text_i"], "round": recs[x["k"]]["round"], "k": x["k"]}
               for x in window if x["phase"] == "measured"]
    out = {
        "window_local": [w0.isoformat(), w1.isoformat()],
        "reason": "another agent briefly started mlx_lm.server on 8101 (possibly loading Qwen3.6 weights) and "
                  "tried to start mlx_audio servers, ~01:52:30-01:54, for at most ~15 s (orchestrator note)",
        "clock": {"method": "access-line perf_counter_ns + (time.time_ns() - time.perf_counter_ns()) at check time",
                  "sweep_started_at": d["started_at"], "first_post_wall": first_post.isoformat(),
                  "first_post_minus_started_s": round((first_post - started).total_seconds(), 1)},
        "n_in_window": len(window), "window_requests": window,
        "window_ttfc_ratio": c.stats([x["ttfc_ratio"] for x in window if x["ttfc_ratio"]]),
        "window_total_ratio": c.stats([x["total_ratio"] for x in window if x["total_ratio"]]),
        "outside_ttfc_ratio": c.stats([x["ttfc_ratio"] for x in rows if not x["in_window"] and x["ttfc_ratio"]
                                       and x["phase"] == "measured"]),
        "outside_total_ratio": c.stats([x["total_ratio"] for x in rows if not x["in_window"] and x["total_ratio"]
                                        and x["phase"] == "measured"]),
        "top_ttfc_outliers": outliers,
        "replace": replace,
        "per_request_wall": [[x["k"], x["wall"]] for x in rows],
    }
    print(c.save_json("s3_sweep_perturbation.json", out))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_request_wall", "window_requests")},
                     indent=1, ensure_ascii=False)[:4000])
    for x in window:
        print(x)


if __name__ == "__main__":
    main()

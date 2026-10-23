"""The G7 dual path's echo counters on its cached calls (spikes/g7dual), with the working tree's
echo path: did the stricter rule for leaving the bypass change anything on lines with echo?

    cd spikes/g7dual && uv run --locked python ../realstack/g7_counters.py OUT.json

Runs every cached call of both layouts (`dt`, `bargein`; G.168 echo, AICC's playout) through
the gateway's echo path with the default `aec` configuration, as `g7_run.py run` does for its
`pp` variant, but writes no clips; compares the counters with the stored `pp` runs
(spikes/data/g7dual/runs/<layout>/pp.json, made with the rule before this change).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g7dual"))

import g7_chain as gc  # noqa: E402 - first: it puts spikes/aec on the path, caps BLAS threads

import json  # noqa: E402
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import g7_gateway as gg  # noqa: E402
from mlx_realtime.config import AecConfig, SpeexConfig  # noqa: E402

AEC = AecConfig(speex=SpeexConfig(preprocess=True))  # the `pp` variant: the gateway's default


def counters(calls: list[dict]) -> dict[str, object]:
    aligned = [c["speex_aligned_at_s"] for c in calls if c["speex_aligned_at_s"] is not None]
    return {
        "calls": len(calls),
        "bypasses": sum(c["bypasses"] for c in calls),
        "calls_with_bypass": sum(1 for c in calls if c["bypasses"]),
        "ending_bypassed": sum(1 for c in calls if c["echo_absent"]),
        "aec3_realignments": sum(c["realignments"] for c in calls),
        "speex_realignments": sum(c["speex_realignments"] for c in calls),
        "speex_aligned": len(aligned),
        "speex_aligned_at_s_p50": round(float(np.median(aligned)), 2) if aligned else None,
        "speex_aligned_at_s_max": round(float(max(aligned)), 2) if aligned else None,
        "erle_db_p50": round(float(np.median([c["erle_db"] for c in calls if c["erle_db"] is not None])), 2),
    }


def main() -> None:
    out: dict[str, object] = {}
    for layout in ("dt", "bargein"):
        stored = json.loads((gc.CACHE / "runs" / layout / "pp.json").read_text())
        now = []
        for cond in gc.conds():
            path = gc.CACHE / "calls" / layout / f"{cond.cid}.pkl"
            with path.open("rb") as file:
                call = pickle.load(file)  # noqa: S301 - our own cache
            now.append(gg.run_call_sync(call, AEC)["summary"])
        out[layout] = {"before": counters(list(stored["calls"].values())), "after": counters(now)}
        print(layout, json.dumps(out[layout]), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


main()

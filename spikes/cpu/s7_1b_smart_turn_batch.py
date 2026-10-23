"""S7.1b (extra) — does batching a burst of Smart Turn evaluations into one call help on CPU?

    uv run python s7_1b_smart_turn_batch.py

Both files, intra-op 2 and 4, spinning off: model call with B = 1 and B = 4 random feature sets,
20 warm-up + 200 timed calls each. Reports per call and per item. Features are excluded (they are
per clip either way). Writes spikes/results/s7_1b_smart_turn_batch.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ST_FILES, environment, load_guard, model_path, ort_session, save_json, stats_ms  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(99)
    feats = rng.standard_normal((8, 80, 800)).astype(np.float32) * 0.5
    results = []
    for name, fname in ST_FILES.items():
        for intra in (2, 4):
            sess = ort_session(model_path(fname), intra=intra, spinning=False)
            for b in (1, 4):
                rec = {"model": name, "intra_op": intra, "B": b}
                with load_guard(rec):
                    ns = []
                    for i in range(220):
                        x = np.roll(feats, i, axis=0)[:b]
                        a = time.perf_counter_ns()
                        sess.run(None, {"input_features": x})
                        if i >= 20:
                            ns.append(time.perf_counter_ns() - a)
                rec["per_call_ms"] = stats_ms(ns)
                rec["per_item_ms_p50"] = round(rec["per_call_ms"]["p50"] / b, 2)
                results.append(rec)
                print(f"{name} intra={intra} B={b}: call p50 {rec['per_call_ms']['p50']:.1f} p95 {rec['per_call_ms']['p95']:.1f} "
                      f"ms, per item {rec['per_item_ms_p50']:.1f} ms | load {rec['load_before']}->{rec['load_after']}", flush=True)
    save_json("s7_1b_smart_turn_batch.json", {"spike": "S7.1b Smart Turn batched calls (extra)", "env": environment(),
                                             "results": results})


if __name__ == "__main__":
    main()

"""S7.1 — Smart Turn v3.2 timing on CPU: {cpu (int8), gpu (fp32)} x intra-op {1, 2, 4} x spinning {on, off}.

    uv run python s7_1_smart_turn_timing.py [--runs 200] [--warmup 20]

Per configuration: a fresh CPUExecutionProvider session (sequential, inter-op 1, ORT_ENABLE_ALL),
20 warm-up runs, then 200 runs on random 8 s inputs. Each run times the numpy Whisper features
(pipecat port) and the model call separately. Cold start (session creation, first features, first
call) is reported separately. The 1-minute load average is recorded before and after each
configuration; a configuration waits (up to 5 min) for load <= 4 before it starts.
Writes spikes/results/s7_1_smart_turn_timing.json.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ST_FILES, environment, load_guard, model_path, ort_session, save_json, stats_ms  # noqa: E402
from whisper_features import compute_whisper_log_mel_features  # noqa: E402


def numpy_blas() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return buf.getvalue()


def run_config(name: str, intra: int, spinning: bool, inputs: list[np.ndarray], warmup: int, runs: int) -> dict:
    rec: dict = {"model": name, "file": ST_FILES[name], "intra_op_num_threads": intra, "allow_spinning": spinning}
    path = model_path(ST_FILES[name])
    with load_guard(rec):
        t0 = time.perf_counter_ns()
        sess = ort_session(path, intra=intra, spinning=spinning)
        t1 = time.perf_counter_ns()
        feats = compute_whisper_log_mel_features(inputs[0])
        t2 = time.perf_counter_ns()
        sess.run(None, {"input_features": feats[None]})
        t3 = time.perf_counter_ns()
        rec["cold_ms"] = {
            "session_create": round((t1 - t0) / 1e6, 2),
            "first_features": round((t2 - t1) / 1e6, 2),
            "first_model_call": round((t3 - t2) / 1e6, 2),
        }
        fe_ns, model_ns, total_ns = [], [], []
        cpu0 = wall0 = None
        for i in range(warmup + runs):
            if i == warmup:
                cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
            audio = inputs[(i + 1) % len(inputs)]
            a = time.perf_counter_ns()
            x = compute_whisper_log_mel_features(audio)
            b = time.perf_counter_ns()
            p = sess.run(None, {"input_features": x[None]})[0]
            c = time.perf_counter_ns()
            if i >= warmup:
                fe_ns.append(b - a)
                model_ns.append(c - b)
                total_ns.append(c - a)
        cpu = time.process_time_ns() - cpu0
        wall = time.perf_counter_ns() - wall0
    assert p.shape == (1, 1) and 0.0 <= float(p[0, 0]) <= 1.0
    rec["features_ms"] = stats_ms(fe_ns)
    rec["model_ms"] = stats_ms(model_ns)
    rec["total_ms"] = stats_ms(total_ns)
    rec["cpu_cores_busy"] = round(cpu / wall, 2)  # process CPU time / wall time during the timed runs
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(1234)
    # Distinct random 8 s inputs (16 kHz); the audio content does not change the work done.
    inputs = [(0.1 * rng.standard_normal(128000)).astype(np.float32) for _ in range(args.runs + args.warmup + 1)]

    configs = [(m, t, s) for m in ST_FILES for t in (1, 2, 4) for s in (True, False)]
    results = []
    for m, t, s in configs:
        r = run_config(m, t, s, inputs, args.warmup, args.runs)
        results.append(r)
        print(
            f"{m:9s} intra={t} spin={'on ' if s else 'off'}  features p50 {r['features_ms']['p50']:6.2f} "
            f"p95 {r['features_ms']['p95']:6.2f} max {r['features_ms']['max']:6.2f} | model p50 "
            f"{r['model_ms']['p50']:6.2f} p95 {r['model_ms']['p95']:6.2f} max {r['model_ms']['max']:6.2f} ms"
            f" | cores {r['cpu_cores_busy']:.2f} | load {r['load_before']}->{r['load_after']}",
            flush=True,
        )
    save_json(
        "s7_1_smart_turn_timing.json",
        {
            "spike": "S7.1 Smart Turn v3.2 CPU timing",
            "env": environment(),
            "numpy_config": numpy_blas(),
            "protocol": {"warmup": args.warmup, "runs": args.runs, "input": "random normal * 0.1, 8 s at 16 kHz",
                         "provider": "CPUExecutionProvider", "execution_mode": "ORT_SEQUENTIAL",
                         "inter_op_num_threads": 1, "graph_optimization_level": "ORT_ENABLE_ALL"},
            "results": results,
        },
    )


if __name__ == "__main__":
    main()

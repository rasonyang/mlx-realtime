"""S7.5b (extra) — GIL behaviour of the CPU work the gateway puts on executor threads.

    uv run python s7_5b_gil_probe.py

A pure-Python counter thread spins while the main thread repeatedly runs (a) the Smart Turn fp32
model call, (b) the numpy Whisper features, (c) a Silero window, (d) pure Python. Reported:
the counter rate (does the work release the GIL?) and the per-call time of the work (does the work
suffer when another thread holds the GIL?), with sys.setswitchinterval 5 ms (default) and 1 ms.
Writes spikes/results/s7_5b_gil_probe.json.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SILERO_FILE, ST_FILES, environment, load_guard, model_path, ort_session, save_json  # noqa: E402
from silero_vad import SileroOnnxModel  # noqa: E402
from whisper_features import compute_whisper_log_mel_features  # noqa: E402


def measure(work, seconds: float, competitor: bool) -> dict:
    stop = threading.Event()
    count = [0]

    def spin():
        c = 0
        while not stop.is_set():
            c += 1
        count[0] = c

    t = threading.Thread(target=spin) if competitor else None
    if t:
        t.start()
    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter_ns()
        work()
        times.append(time.perf_counter_ns() - a)
    wall = time.perf_counter() - t0
    stop.set()
    if t:
        t.join()
    a = np.array(times) / 1e6
    return {"calls": len(times), "per_call_ms_p50": round(float(np.percentile(a, 50)), 3),
            "per_call_ms_p95": round(float(np.percentile(a, 95)), 3),
            "counter_rate_M_per_s": round(count[0] / wall / 1e6, 2) if competitor else None}


def main() -> None:
    st = ort_session(model_path(ST_FILES["v3.2-gpu"]), intra=1, spinning=False)
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(128000)).astype(np.float32)
    feats = compute_whisper_log_mel_features(audio)[None]
    vad = SileroOnnxModel(ort_session(model_path(SILERO_FILE), intra=1, spinning=False))
    win = (0.1 * rng.standard_normal(512)).astype(np.float32)
    work = {
        "smart_turn_fp32_model_intra1": lambda: st.run(None, {"input_features": feats}),
        "whisper_features_numpy": lambda: compute_whisper_log_mel_features(audio),
        "silero_window_16k": lambda: vad(win, 16000),
        "pure_python": lambda: sum(range(20000)),
    }
    out: dict = {"protocol": "2 s per cell; competitor = a pure-Python busy counter thread"}
    with load_guard(out):
        base = measure(lambda: time.sleep(0.01), 2.0, True)["counter_rate_M_per_s"]
        out["counter_rate_idle_main_M_per_s"] = base
        out["cells"] = []
        for interval in (0.005, 0.001):
            sys.setswitchinterval(interval)
            for name, fn in work.items():
                alone = measure(fn, 2.0, False)
                shared = measure(fn, 2.0, True)
                cell = {"switchinterval_ms": interval * 1000, "work": name,
                        "alone_ms_p50": alone["per_call_ms_p50"], "with_competitor_ms_p50": shared["per_call_ms_p50"],
                        "with_competitor_ms_p95": shared["per_call_ms_p95"],
                        "competitor_rate_M_per_s": shared["counter_rate_M_per_s"]}
                out["cells"].append(cell)
                print(f"switch {interval*1000:.0f} ms  {name:30s} alone {cell['alone_ms_p50']:8.3f} ms | with busy thread "
                      f"{cell['with_competitor_ms_p50']:8.3f} ms | competitor {cell['competitor_rate_M_per_s']} M/s "
                      f"(idle main: {base})", flush=True)
        sys.setswitchinterval(0.005)
    save_json("s7_5b_gil_probe.json", {"spike": "S7.5b GIL probe (extra)", "env": environment(), **out})


if __name__ == "__main__":
    main()

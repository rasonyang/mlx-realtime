"""S7.6 — soxr.ResampleStream fed random 10-40 ms chunks vs one-shot soxr.resample.

    uv run python s7_6_resample.py

For 24k->16k, 8k->16k and 24k->8k, qualities MQ/HQ/VHQ, dtypes int16 and float32:
- equality (or max |delta|) of the concatenated stream output against soxr.resample on the whole
  signal, over several random chunkings;
- delay(): the pending output reported after each chunk, in samples and ms;
- the lag actually seen by a chunked caller: expected output so far (input x ratio) minus output
  received so far, in ms;
- cost of one 20 ms chunk (n = 1000 after 200 warm-up).
Writes spikes/results/s7_6_resample.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment, load_guard, save_json, stats_ms  # noqa: E402

PAIRS = [(24000, 16000), (8000, 16000), (24000, 8000)]
QUALITIES = ["MQ", "HQ", "VHQ"]
DTYPES = ["int16", "float32"]
SECONDS = 20
CHUNKINGS = 5


def test_signal(rate: int, dtype: str, seed: int = 1) -> np.ndarray:
    """Noise at -12 dBFS plus a log chirp from 50 Hz to just under Nyquist."""
    rng = np.random.default_rng(seed)
    t = np.arange(SECONDS * rate) / rate
    f0, f1 = 50.0, 0.45 * rate
    k = np.log(f1 / f0) / SECONDS
    chirp = np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1) / k)
    x = 0.25 * rng.standard_normal(t.size) + 0.25 * chirp
    x = np.clip(x, -1, 1).astype(np.float32)
    if dtype == "int16":
        return np.round(x * 32767).astype(np.int16)
    return x


def chunk_sizes(rate: int, n: int, rng: np.random.Generator) -> list[int]:
    sizes, total = [], 0
    while total < n:
        s = int(rate * rng.uniform(0.010, 0.040))
        s = min(s, n - total)
        sizes.append(s)
        total += s
    return sizes


def stream(x: np.ndarray, rin: int, rout: int, q: str, sizes: list[int]) -> dict:
    rs = soxr.ResampleStream(rin, rout, 1, dtype=x.dtype.name, quality=q)
    outs, delays, lags_ms = [], [], []
    pos = produced = 0
    for s in sizes:
        y = rs.resample_chunk(x[pos : pos + s])
        pos += s
        produced += y.size
        outs.append(y)
        delays.append(rs.delay())
        lags_ms.append((pos * rout / rin - produced) / rout * 1000)
    tail = rs.resample_chunk(np.zeros(0, dtype=x.dtype), last=True)
    outs.append(tail)
    return {
        "y": np.concatenate(outs),
        "delay_first": delays[0],
        "delay_samples": delays,
        "lag_ms": lags_ms,
        "flush_samples": int(tail.size),
    }


def one_case(rin: int, rout: int, q: str, dtype: str) -> dict:
    x = test_signal(rin, dtype)
    ref = soxr.resample(x, rin, rout, quality=q)
    results = []
    for seed in range(CHUNKINGS):
        sizes = chunk_sizes(rin, x.size, np.random.default_rng(100 + seed))
        st = stream(x, rin, rout, q, sizes)
        y = st["y"]
        n = min(y.size, ref.size)
        d = np.abs(y[:n].astype(np.float64) - ref[:n].astype(np.float64))
        results.append(
            {
                "chunks": len(sizes),
                "len_stream": int(y.size),
                "len_oneshot": int(ref.size),
                "identical": bool(y.size == ref.size and np.array_equal(y, ref)),
                "max_abs_diff": float(d.max()) if n else None,
                "n_diff_samples": int(np.count_nonzero(d)),
                "delay_first_chunk": round(st["delay_first"], 2),
                "delay_steady_min_max": [round(min(st["delay_samples"][5:]), 2), round(max(st["delay_samples"][5:]), 2)],
                "lag_ms_max": round(max(st["lag_ms"]), 2),
                "lag_ms_mean": round(float(np.mean(st["lag_ms"])), 2),
                "flush_samples": st["flush_samples"],
            }
        )
    delays = [r["delay_steady_min_max"][1] for r in results]
    return {
        "in_rate": rin,
        "out_rate": rout,
        "quality": q,
        "dtype": dtype,
        "all_identical": all(r["identical"] for r in results),
        "max_abs_diff": max(r["max_abs_diff"] for r in results),
        "delay_max_samples": max(delays),
        "delay_max_ms": round(max(delays) / rout * 1000, 3),
        "lag_ms_max": max(r["lag_ms_max"] for r in results),
        "chunkings": results,
    }


def chunk_cost(rin: int, rout: int, q: str, dtype: str) -> dict:
    x = test_signal(rin, dtype)
    rs = soxr.ResampleStream(rin, rout, 1, dtype=dtype, quality=q)
    step = rin // 50  # 20 ms
    ns = []
    for i in range(1200):
        c = x[(i * step) % (x.size - step) :][:step]
        a = time.perf_counter_ns()
        rs.resample_chunk(c)
        b = time.perf_counter_ns()
        if i >= 200:
            ns.append(b - a)
    # construction cost, for one stream per session and direction
    ctor = []
    for _ in range(200):
        a = time.perf_counter_ns()
        soxr.ResampleStream(rin, rout, 1, dtype=dtype, quality=q)
        ctor.append(time.perf_counter_ns() - a)
    return {"in_rate": rin, "out_rate": rout, "quality": q, "dtype": dtype,
            "chunk_20ms": stats_ms(ns), "construct": stats_ms(ctor)}


def dither_probe() -> dict:
    """int16 I/O: is the output dithered? (libsoxr's default TPDF dither for int16 output)."""
    z = np.zeros(24000, np.int16)
    yz = soxr.resample(z, 24000, 16000, quality="HQ")
    dc = soxr.resample(np.full(24000, 3, np.int16), 24000, 16000, quality="HQ")[2000:-2000]
    vals, counts = np.unique(dc, return_counts=True)
    again = soxr.resample(z, 24000, 16000, quality="HQ")
    return {
        "zeros_in_nonzero_out": int(np.count_nonzero(yz)),
        "zeros_out_samples": int(yz.size),
        "dc3_in_values_out": {int(v): int(c) for v, c in zip(vals, counts)},
        "oneshot_repeatable": bool(np.array_equal(yz, again)),
        "quality_enum": {str(k): v for k, v in soxr._QUALITY_ENUM_DICT.items()},
    }


def latency_by_quality() -> list[dict]:
    """Availability lag for fixed 20 ms chunks, all five qualities (float32)."""
    out = []
    for q in ["QQ", "LQ", "MQ", "HQ", "VHQ"]:
        for rin, rout in PAIRS:
            rs = soxr.ResampleStream(rin, rout, 1, dtype="float32", quality=q)
            sig = np.random.default_rng(1).standard_normal(rin * 5).astype(np.float32) * 0.1
            step, pos, produced, lags = rin // 50, 0, 0, []
            while pos + step <= sig.size:
                produced += rs.resample_chunk(sig[pos : pos + step]).size
                pos += step
                lags.append((pos * rout / rin - produced) / rout * 1000)
            steady = lags[10:]
            out.append({"quality": q, "in_rate": rin, "out_rate": rout,
                        "lag_ms_min": round(min(steady), 2), "lag_ms_mean": round(float(np.mean(steady)), 2),
                        "lag_ms_max": round(max(steady), 2)})
    return out


def main() -> None:
    cases = [one_case(rin, rout, q, dt) for rin, rout in PAIRS for q in QUALITIES for dt in DTYPES]
    timing = {"note": "one stream, one thread; per 20 ms input chunk"}
    with load_guard(timing):
        timing["runs"] = [chunk_cost(rin, rout, q, "int16") for rin, rout in PAIRS for q in QUALITIES]
    save_json("s7_6_resample.json", {"spike": "S7.6 soxr streaming vs one-shot", "env": environment(),
                                     "cases": cases, "timing": timing, "dither_probe": dither_probe(),
                                     "latency_by_quality_20ms_chunks": latency_by_quality()})
    print(f"{'pair':>12} {'q':>4} {'dtype':>8} {'identical':>9} {'max|d|':>9} {'delay(smp)':>10} "
          f"{'delay ms':>8} {'lag ms':>7}")
    for c in cases:
        print(f"{c['in_rate']//1000:>4}k->{c['out_rate']//1000:>2}k {c['quality']:>5} {c['dtype']:>8} "
              f"{str(c['all_identical']):>9} {c['max_abs_diff']:>9.3g} {c['delay_max_samples']:>10.1f} "
              f"{c['delay_max_ms']:>8.2f} {c['lag_ms_max']:>7.2f}")
    for r in timing["runs"]:
        print(f"cost {r['in_rate']}->{r['out_rate']} {r['quality']}: 20 ms chunk p50 {r['chunk_20ms']['p50']*1000:.1f} us"
              f" p95 {r['chunk_20ms']['p95']*1000:.1f} us; construct p50 {r['construct']['p50']*1000:.1f} us")
    print("load", {k: timing[k] for k in ("load_before", "load_after")})


if __name__ == "__main__":
    main()

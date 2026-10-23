"""S7.5 — 64 real-time streams in one asyncio loop: Silero per-stream vs batched, Smart Turn executor K in {1, 2, 4}.

    uv run python s7_5_concurrency.py [--duration 30] [--st-model v3.2-gpu] [--st-intra 2]

Each stream delivers a 20 ms frame every 20 ms (random phase), as the gateway receives it: a JSON
`input_audio_buffer.append` with base64 PCM16 at 24 kHz. On the loop, per frame: json.loads,
base64 decode, soxr.ResampleStream 24k->16k (HQ, float32), then 512-sample windows.
Silero modes:
- per_stream: every window is submitted at once to a single-thread executor (one call per window);
- batched: windows wait for a 32 ms tick; one call per tick runs all ready streams (B <= 64).
Smart Turn: ThreadPoolExecutor(K), one onnxruntime session per worker (intra-op = --st-intra,
spinning off). Each job = numpy Whisper features + model on a snapshot of the stream's last 8 s.
Load: Poisson end-of-turn evaluations (--st-rate per s over all streams) plus a burst of
--burst-size simultaneous evaluations every --burst-every s.
Measured after a 3 s warm-up: event-loop lag (5 ms sleeper), Silero window latency (window
complete -> result on the loop), Smart Turn queueing delay (submit -> worker start), service time
and total latency, process CPU (CPU time / wall, 100% = one core), and load average.
Writes spikes/results/s7_5_concurrency.json.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SILERO_FILE,
    ST_FILES,
    environment,
    load_guard,
    load_zh_rows,
    model_path,
    ort_session,
    save_json,
)
from silero_vad import BatchedSilero, SileroOnnxModel  # noqa: E402
from whisper_features import compute_whisper_log_mel_features  # noqa: E402

FRAME_S = 0.020
TICK_S = 0.032
WARMUP_S = 3.0
WIN = 512
RING = 8 * 16000


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def summ_ms(v_s: list[float]) -> dict:
    v = np.asarray(v_s) * 1000
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99), "max": round(float(v.max()), 3),
            "mean": round(float(v.mean()), 3)}


class Stream:
    def __init__(self, i: int, msgs: list[str], phase: float, silero_sess):
        self.i = i
        self.msgs = msgs
        self.k = int(i * 37) % len(msgs)
        self.phase = phase
        self.rs = soxr.ResampleStream(24000, 16000, 1, dtype="float32", quality="HQ")
        self.pending = np.zeros(0, dtype=np.float32)
        self.ring = np.zeros(RING, dtype=np.float32)
        self.ring_pos = 0
        self.model = SileroOnnxModel(silero_sess)
        self.ready: collections.deque = collections.deque()

    def ingest(self) -> list[np.ndarray]:
        ev = json.loads(self.msgs[self.k % len(self.msgs)])
        self.k += 1
        pcm = np.frombuffer(base64.b64decode(ev["audio"]), dtype=np.int16)
        x = self.rs.resample_chunk(pcm.astype(np.float32) / 32768.0)
        if x.size:
            n = x.size
            end = self.ring_pos + n
            if end <= RING:
                self.ring[self.ring_pos : end] = x
            else:
                cut = RING - self.ring_pos
                self.ring[self.ring_pos :] = x[:cut]
                self.ring[: n - cut] = x[cut:]
            self.ring_pos = end % RING
            self.pending = np.concatenate([self.pending, x])
        wins = []
        while self.pending.size >= WIN:
            wins.append(self.pending[:WIN])
            self.pending = self.pending[WIN:]
        return wins

    def last_8s(self) -> np.ndarray:
        return np.concatenate([self.ring[self.ring_pos :], self.ring[: self.ring_pos]])


_tls = threading.local()


def st_init(path: str, intra: int) -> None:
    _tls.sess = ort_session(Path(path), intra=intra, spinning=False)
    x = compute_whisper_log_mel_features(np.zeros(RING, dtype=np.float32))
    _tls.sess.run(None, {"input_features": x[None]})


def st_job(audio: np.ndarray) -> tuple[float, float, float, float]:
    t0 = time.perf_counter()
    feats = compute_whisper_log_mel_features(audio)
    t1 = time.perf_counter()
    p = float(_tls.sess.run(None, {"input_features": feats[None]})[0][0, 0])
    t2 = time.perf_counter()
    return t0, t1, t2, p


def st_schedule(duration: float, n_streams: int, rate: float, burst_size: int, burst_every: float, seed: int):
    rng = np.random.default_rng(seed)
    ev = []
    t = 0.0
    while True:
        t += rng.exponential(1 / rate) if rate > 0 else duration + 1
        if t >= duration:
            break
        ev.append((t, int(rng.integers(n_streams)), "poisson"))
    if burst_size > 0:
        t = burst_every / 2
        while t < duration:
            for s in rng.choice(n_streams, size=burst_size, replace=False):
                ev.append((t, int(s), "burst"))
            t += burst_every
    return sorted(ev)


async def scenario(args, mode: str, k_workers: int, msgs_pool: list[list[str]]) -> dict:
    loop = asyncio.get_running_loop()
    total = WARMUP_S + args.duration
    silero_sess = ort_session(model_path(SILERO_FILE), intra=1, spinning=False)
    rng = np.random.default_rng(5)
    streams = [Stream(i, msgs_pool[i % len(msgs_pool)], float(rng.uniform(0, FRAME_S)), silero_sess)
               for i in range(args.streams)]
    batched = BatchedSilero(silero_sess, args.streams)
    silero_ex = cf.ThreadPoolExecutor(1, thread_name_prefix="silero")
    st_ex = None
    if k_workers:
        st_ex = cf.ThreadPoolExecutor(k_workers, thread_name_prefix="smartturn", initializer=st_init,
                                      initargs=(str(model_path(ST_FILES[args.st_model])), args.st_intra))
        barrier = threading.Barrier(k_workers)  # make sure all K workers exist and are warm
        await asyncio.gather(*[loop.run_in_executor(st_ex, barrier.wait) for _ in range(k_workers)])
    # warm the Silero thread
    await loop.run_in_executor(silero_ex, SileroOnnxModel(silero_sess), np.zeros(WIN, np.float32), 16000)

    t_start = time.perf_counter() + 0.05
    t_measure = t_start + WARMUP_S
    t_end = t_start + total
    lags, vad_lat, batch_sizes, st_rec = [], [], [], []
    backlog = {"max": 0, "now": 0}

    def on_vad(t_ready, fut):
        backlog["now"] -= 1
        if t_ready >= t_measure:
            vad_lat.append(time.perf_counter() - t_ready)

    async def sleep_until(t):
        d = t - time.perf_counter()
        if d > 0:
            await asyncio.sleep(d)

    async def stream_task(s: Stream):
        nxt = t_start + s.phase
        while True:
            await sleep_until(nxt)
            if time.perf_counter() >= t_end:
                return
            for w in s.ingest():
                t_ready = time.perf_counter()
                if mode == "per_stream":
                    backlog["now"] += 1
                    backlog["max"] = max(backlog["max"], backlog["now"])
                    fut = loop.run_in_executor(silero_ex, s.model, w, 16000)
                    fut.add_done_callback(lambda f, t=t_ready: on_vad(t, f))
                elif mode == "batched":
                    s.ready.append((w, t_ready))
                # ingest_only: the window is formed and dropped
            nxt += FRAME_S

    async def batch_ticker():
        nxt = t_start + TICK_S
        while time.perf_counter() < t_end:
            await sleep_until(nxt)
            while True:
                ready = [s for s in streams if s.ready]
                if not ready:
                    break
                idx = np.array([s.i for s in ready])
                items = [s.ready.popleft() for s in ready]
                frames = np.stack([w for w, _ in items])
                await loop.run_in_executor(silero_ex, batched.run, idx, frames)
                now = time.perf_counter()
                if now >= t_measure:
                    batch_sizes.append(len(ready))
                    vad_lat.extend(now - t for _, t in items if t >= t_measure)
            nxt += TICK_S

    async def lag_monitor():
        while time.perf_counter() < t_end:
            a = time.perf_counter()
            await asyncio.sleep(0.005)
            if a >= t_measure:
                lags.append(time.perf_counter() - a - 0.005)

    async def st_driver():
        if not st_ex:
            return
        sched = st_schedule(args.duration, args.streams, args.st_rate, args.burst_size, args.burst_every, seed=9)
        pend = []
        for t_rel, si, kind in sched:
            await sleep_until(t_measure + t_rel)
            audio = streams[si].last_8s()  # snapshot on the loop
            t_sub = time.perf_counter()
            fut = loop.run_in_executor(st_ex, st_job, audio)

            def done(f, t_sub=t_sub, kind=kind):
                t0, t1, t2, _ = f.result()
                st_rec.append({"kind": kind, "queue": t0 - t_sub, "features": t1 - t0, "model": t2 - t1,
                               "service": t2 - t0, "total": time.perf_counter() - t_sub})

            fut.add_done_callback(done)
            pend.append(fut)
        if pend:
            await asyncio.wait(pend)

    rec: dict = {"silero_mode": mode, "st_workers": k_workers, "switch_interval_ms": sys.getswitchinterval() * 1000}
    with load_guard(rec, max_wait_s=args.max_load_wait):
        # The schedule starts only after the load guard has finished waiting.
        t_start = time.perf_counter() + 0.05
        t_measure = t_start + WARMUP_S
        t_end = t_start + total
        cpu0 = wall0 = None

        async def cpu_mark():
            nonlocal cpu0, wall0
            await sleep_until(t_measure)
            cpu0, wall0 = time.process_time(), time.perf_counter()

        tasks = [stream_task(s) for s in streams] + [lag_monitor(), cpu_mark()]
        if mode == "batched":
            tasks.append(batch_ticker())
        st_task = asyncio.ensure_future(st_driver())
        await asyncio.gather(*tasks)
        cpu1, wall1 = time.process_time(), time.perf_counter()
        await st_task
    silero_ex.shutdown(wait=True)
    if st_ex:
        st_ex.shutdown(wait=True)

    windows_expected = args.streams * args.duration / TICK_S
    rec.update({
        "cpu_pct": round(100 * (cpu1 - cpu0) / (wall1 - wall0), 1),
        "loop_lag_ms": summ_ms(lags),
        "silero_latency_ms": summ_ms(vad_lat),
        "silero_windows": len(vad_lat),
        "silero_windows_expected_approx": int(windows_expected),
        "per_stream_executor_backlog_max": backlog["max"] if mode == "per_stream" else None,
        "batch_size": {"mean": round(float(np.mean(batch_sizes)), 2), "min": int(min(batch_sizes)),
                       "max": int(max(batch_sizes)), "calls": len(batch_sizes)} if batch_sizes else None,
    })
    if st_rec:
        rec["smart_turn"] = {
            "model": args.st_model, "intra_op": args.st_intra, "jobs": len(st_rec),
            "queue_ms": summ_ms([r["queue"] for r in st_rec]),
            "queue_ms_burst": summ_ms([r["queue"] for r in st_rec if r["kind"] == "burst"]),
            "queue_ms_poisson": summ_ms([r["queue"] for r in st_rec if r["kind"] == "poisson"]),
            "service_ms": summ_ms([r["service"] for r in st_rec]),
            "features_ms": summ_ms([r["features"] for r in st_rec]),
            "model_ms": summ_ms([r["model"] for r in st_rec]),
            "total_ms": summ_ms([r["total"] for r in st_rec]),
            "total_ms_burst": summ_ms([r["total"] for r in st_rec if r["kind"] == "burst"]),
        }
    return rec


def build_msgs(rows, n_pools: int = 8, seconds: float = 10.0) -> list[list[str]]:
    """Pre-encoded append events from real speech at 24 kHz, 20 ms each (480 samples = 960 bytes)."""
    pools = []
    for p in range(n_pools):
        audio = np.concatenate([r["audio"] for r in rows[p * 5 : p * 5 + 5]])
        a24 = soxr.resample(audio, 16000, 24000, quality="HQ")[: int(24000 * seconds)]
        pcm = np.clip(np.round(a24 * 32767), -32768, 32767).astype(np.int16)
        pools.append([
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i : i + 480].tobytes()).decode()})
            for i in range(0, len(pcm) - 480 + 1, 480)
        ])
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--st-model", default="v3.2-gpu", choices=list(ST_FILES))
    ap.add_argument("--st-intra", type=int, default=2)
    ap.add_argument("--st-rate", type=float, default=8.0, help="Poisson Smart Turn evaluations per second, all streams")
    ap.add_argument("--burst-size", type=int, default=8)
    ap.add_argument("--burst-every", type=float, default=5.0)
    ap.add_argument("--workers", default="0,1,2,4")
    ap.add_argument("--modes", default="per_stream,batched")
    ap.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                    help="also run ingest only (no Silero, no Smart Turn)")
    ap.add_argument("--max-load-wait", type=float, default=300.0,
                    help="seconds to wait for load <= 4 before each scenario; then run and flag")
    ap.add_argument("--switch-interval-ms", type=float, default=5.0, help="sys.setswitchinterval (CPython default 5 ms)")
    ap.add_argument("--out", default="s7_5_concurrency.json")
    args = ap.parse_args()
    sys.setswitchinterval(args.switch_interval_ms / 1000)

    rows = load_zh_rows()
    msgs = build_msgs(rows)
    runs = []
    if args.baseline:
        runs.append(asyncio.run(scenario(args, "ingest_only", 0, msgs)))
    for mode in args.modes.split(","):
        for k in (int(x) for x in args.workers.split(",")):
            r = asyncio.run(scenario(args, mode, k, msgs))
            runs.append(r)
    for r in runs:
        st = r.get("smart_turn")
        print(f"{r['silero_mode']:11s} K={r['st_workers']} cpu {r['cpu_pct']:6.1f}% lag p50 {r['loop_lag_ms'].get('p50')} "
              f"p99 {r['loop_lag_ms'].get('p99')} max {r['loop_lag_ms'].get('max')} ms | silero p50 "
              f"{r['silero_latency_ms'].get('p50')} p99 {r['silero_latency_ms'].get('p99')} ms"
              + (f" | ST queue p50 {st['queue_ms']['p50']} p99 {st['queue_ms']['p99']} max {st['queue_ms']['max']}"
                 f" total p99 {st['total_ms']['p99']} ms" if st else "")
              + f" | load {r['load_before']}->{r['load_after']}", flush=True)
    save_json(args.out, {
        "spike": "S7.5 concurrency: 64 streams, Silero per-stream vs batched, Smart Turn executor K",
        "env": environment(),
        "config": vars(args),
        "notes": ["ingest_only = json + base64 + soxr per frame, windows formed, no inference",
                  "Silero latency includes the batching wait in batched mode (up to one 32 ms tick)"],
        "runs": runs,
    })


if __name__ == "__main__":
    main()

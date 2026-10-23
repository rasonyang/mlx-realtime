"""S11 part 2: CPU per session per audio second, per-frame latency, and 8 sessions vs 1 (GIL).

    cd spikes/aec
    uv run python s11_cpu.py            # every engine, one child process each -> spikes/results/s11_cpu.json
    uv run python s11_cpu.py --child lk # one engine (used by the parent)

Input: one cached core call (core_m4_e10_d120, 60 s), fed frame by frame with the oracle reference.
Per engine, in its own child process:
  - single session: 200 warm-up frames (discarded), then 6000 timed frames; per frame
    time.thread_time_ns (CPU) and perf_counter_ns (wall); CPU per audio second = sum(thread time) /
    60 s, cross-checked with time.process_time_ns over the loop;
  - concurrency: N sessions (own engine instance each) on ThreadPoolExecutor(N), N = 1 and 8, each
    processing the same 6000 frames; throughput = N x 60 s / wall; scaling = throughput(8) / throughput(1).
Also measured: the 3.8 kHz pre-filter alone (two streaming FIRs) and the GCC-PHAT tracker over the call.
Numerical threads are capped at 2 (s11_common). The 1/5/15-minute load averages are recorded before and
after each measurement; other agents may be running. Throwaway spike code.
"""

from __future__ import annotations

import s11_common as c

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import s11_align as al  # noqa: E402
import s11_chain as ch  # noqa: E402
import s11_engines as en  # noqa: E402

CALL = "core_m4_e10_d120"
ENGINES = ["lk", "lknd", "pw", "lk_pf", "sx1024_pf", "sx2048_pf", "sx1024pp_pf", "sx1024ppd_pf", "sx2048ppd_pf",
           "bl03_pf", "bl05_pf", "none_pf"]
WARM, TIMED = 200, 6000


def frames():
    arr, _ = ch.load(c.CACHE / "core" / f"{CALL}.npz")
    cap = arr["capture"]
    nf = len(cap) // c.FRAME
    caps = cap[: nf * c.FRAME].reshape(nf, c.FRAME)
    refs = al.reference_frames(arr["ref_oracle"], None, nf)
    return arr, np.ascontiguousarray(caps), refs


def session(engine: str, caps: np.ndarray, refs: np.ndarray, record: bool) -> dict:
    e = en.make(engine)
    nf = len(caps)
    for j in range(WARM):
        e.process(caps[j % nf], refs[j % nf])
    tt = np.empty(TIMED, dtype=np.int64) if record else None
    wt = np.empty(TIMED, dtype=np.int64) if record else None
    p0 = time.process_time_ns()
    t_all = time.perf_counter_ns()
    for i in range(TIMED):
        j = (WARM + i) % nf
        if record:
            a, b = time.thread_time_ns(), time.perf_counter_ns()
            e.process(caps[j], refs[j])
            wt[i] = time.perf_counter_ns() - b
            tt[i] = time.thread_time_ns() - a
        else:
            e.process(caps[j], refs[j])
    wall = time.perf_counter_ns() - t_all
    proc = time.process_time_ns() - p0
    out = {"wall_s": wall / 1e9}
    if record:
        audio_s = TIMED * c.FRAME / c.SR
        out.update({
            "frames": TIMED,
            "cpu_ms_per_audio_s_thread": round(tt.sum() / 1e6 / audio_s, 3),
            "cpu_ms_per_audio_s_process": round(proc / 1e6 / audio_s, 3),
            "frame_cpu_us": c.stats(tt / 1e3, 2),
            "frame_wall_us": c.stats(wt / 1e3, 2),
        })
    return out


def child(engine: str) -> dict:
    arr, caps, refs = frames()
    res: dict = {"engine": engine, "description": en.DESCRIPTIONS.get(engine, ""), "load_before": c.loadavg()}
    res["single"] = session(engine, caps, refs, record=True)
    res["load_after_single"] = c.loadavg()
    conc = {}
    for n in (1, 8):
        with ThreadPoolExecutor(n) as ex:
            t0 = time.perf_counter_ns()
            outs = list(ex.map(lambda _: session(engine, caps, refs, record=False), range(n)))
            wall = (time.perf_counter_ns() - t0) / 1e9
        audio = n * TIMED * c.FRAME / c.SR
        conc[str(n)] = {"wall_s": round(wall, 3), "audio_s": audio, "throughput_x_realtime": round(audio / wall, 1),
                        "per_session_wall_s": [round(o["wall_s"], 3) for o in outs], "load_after": c.loadavg()}
    conc["scaling_8_vs_1"] = round(conc["8"]["throughput_x_realtime"] / conc["1"]["throughput_x_realtime"], 2)
    res["concurrency"] = conc
    if engine == "none_pf":  # the gateway-side extras, measured once
        g = al.GccPhat(al.KEEP_LINEAR)
        t0 = time.process_time_ns()
        g.track(arr["capture"], arr["ref_model"])
        dur = len(arr["capture"]) / c.SR
        res["gcc_phat"] = {"cpu_ms_per_audio_s_thread": round(g.cpu_ns / 1e6 / dur, 3),
                           "cpu_ms_per_audio_s_process": round((time.process_time_ns() - t0) / 1e6 / dur, 3),
                           "windows": len(g.log), "note": "one 32768-point rfft pair + irfft per 0.5 s hop"}
    return res


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2]), default=c._json_default))
        return
    out = {"spike": "S11 part 2", "part": "cpu", "environment": c.environment(), "call": CALL,
           "protocol": __doc__.split("Per engine")[1].split("Throwaway")[0].strip(), "engines": {}}
    for e in ENGINES:
        if e.startswith("sx") and not c.speex_available():
            out["engines"][e] = {"skipped": "SpeexDSP library not found"}
            continue
        p = subprocess.run([sys.executable, str(c.HERE / "s11_cpu.py"), "--child", e], cwd=str(c.HERE),
                           capture_output=True, text=True)
        if p.returncode != 0:
            out["engines"][e] = {"crash": {"returncode": p.returncode, "stderr_tail": p.stderr[-2000:]}}
            print(f"{e}: CRASH rc={p.returncode}", flush=True)
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        out["engines"][e] = r
        s = r["single"]
        print(f"{e:12s} cpu {s['cpu_ms_per_audio_s_thread']:7.3f} ms/s (proc {s['cpu_ms_per_audio_s_process']:7.3f}) "
              f"frame p50 {s['frame_cpu_us']['p50']} p99 {s['frame_cpu_us']['p99']} max {s['frame_cpu_us']['max']} us | "
              f"x8 scaling {r['concurrency']['scaling_8_vs_1']} load {r['load_before'][0]}->{r['concurrency']['8']['load_after'][0]}",
              flush=True)
    p = c.save_json("s11_cpu.json", out)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

"""S11 part 2 child process: one engine variant + one alignment method over a batch of cached calls.

    uv run python s11_child.py <job.json>

The driver (s11_run.py) starts one child per batch, so a native crash (livekit's SIGABRT on a bad frame
size) kills only the child. Before each call the child writes `<job>.current` with the call id; after
each call it writes one result JSON. Throwaway spike code.
"""

from __future__ import annotations

import s11_common as c

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import s11_align as al  # noqa: E402
import s11_chain as ch  # noqa: E402
import s11_engines as en  # noqa: E402
import s11_metrics as mt  # noqa: E402


def run_call(engine: str, align: str, npz: Path, silero: c.Silero, asr_dir: Path | None,
             spectral: bool = False) -> dict:
    arr, meta = ch.load(npz)
    cap = arr["capture"]
    nf = len(cap) // c.FRAME
    gcc_info = None
    is_aec3 = engine.startswith(("lk", "pw"))
    if align == "oracle":
        refs = al.reference_frames(arr["ref_oracle"], None, nf)
    elif align in ("engine", "locked"):
        refs = al.reference_frames(arr["ref_model" if align == "engine" else "ref_locked"], None, nf)
    elif align == "learned":
        # A gateway that learns AICC's tick phase from its GCC-PHAT observations of the first playout
        # runs, and keeps the bulk delay with AEC3: emulated as the blind model for the first two
        # responses and the locked model from response 3 on (switched in the silent gap before it; the
        # third response starts after 7-15 s in the std60 layout).
        resp = meta["responses"]
        src = arr["ref_model"].copy()
        if len(resp) >= 3 and resp[2].get("model_start") is not None and resp[2].get("true_start") is not None:
            t_sw = min(resp[2]["model_start"], resp[2]["true_start"]) - ch.LEAD_S - 0.03
            n_a = max(0, int(t_sw * c.SR))
            src[n_a:] = arr["ref_locked"][n_a:]
        refs = al.reference_frames(src, None, nf)
    elif align in ("gcc", "gcc_locked"):
        src = arr["ref_model" if align == "gcc" else "ref_locked"]
        g = al.GccPhat(al.KEEP_AEC3 if is_aec3 else al.KEEP_LINEAR)
        shifts = g.track(cap, src)
        refs = al.reference_frames(src, shifts, nf)
        applied = [r for r in g.log if "applied_shift_ms" in r]
        valid = [r for r in g.log if r.get("valid")]
        gcc_info = {"cpu_ms": round(g.cpu_ns / 1e6, 2), "windows": len(g.log), "valid": len(valid),
                    "applied": applied[:20], "final_shift_ms": round(g.shift16 / c.SR * 1000, 3),
                    "first_applied_t": applied[0]["t"] if applied else None,
                    "psr_valid_p50": round(float(np.median([r["psr"] for r in valid])), 2) if valid else None}
    else:
        raise ValueError(align)

    eng = en.make(engine)
    out = np.empty(nf * c.FRAME, dtype=np.int16)
    tt = np.empty(nf, dtype=np.int64)
    wall = np.empty(nf, dtype=np.int64)
    p0 = time.process_time_ns()
    for j in range(nf):
        near = cap[j * c.FRAME:(j + 1) * c.FRAME]
        far = refs[j]
        a = time.thread_time_ns()
        b = time.perf_counter_ns()
        y = eng.process(near, far)
        wall[j] = time.perf_counter_ns() - b
        tt[j] = time.thread_time_ns() - a
        out[j * c.FRAME:(j + 1) * c.FRAME] = y
    proc_ns = time.process_time_ns() - p0
    dur = nf * c.FRAME / c.SR

    fa, eo, npres = arr["far_active"], arr["eo"], arr["near_present"]
    end = len(fa)
    onset = mt.onset_frame(fa) or 0
    t_change = meta["t_change"]
    fc = int(round(t_change / mt.F_S)) if t_change is not None else end
    w = mt.erle_windows(cap, out, eo)
    erl1 = meta["erl_eff"]
    erl2 = meta["erl2_eff"]
    res: dict = {"cid": meta["cond"]["cid"], "engine": engine, "align": align, "duration_s": dur,
                 "meta": {k: meta.get(k) for k in ("erl_eff", "erl2_eff", "path2", "t_change", "tick_phase_ms",
                                                   "residual_delay_ms_nominal", "true_minus_model_ms",
                                                   "true_minus_model_by_response_ms", "locked_model_tick_errors",
                                                   "underrun_restarts_true", "underrun_restarts_model",
                                                   "playout_onsets")}}
    res["erle"] = {
        "eo_windows": int(len(w["erle"])),
        "ss": mt.median_in(w, fa, onset, fc, 5.0),
        "ss_after_change": mt.median_in(w, fa, fc, end, 5.0) if t_change is not None else None,
        "conv20_s": mt.convergence(w, fa, onset, fc, 20.0),
        "conv20_after_change_s": mt.convergence(w, fa, fc, end, 20.0) if t_change is not None else None,
        "conv_lret20_s": mt.convergence(w, fa, onset, fc, 20.0 - erl1) if erl1 is not None else None,
        "conv_lret20_after_change_s": (mt.convergence(w, fa, fc, end, 20.0 - erl2)
                                       if (t_change is not None and erl2 is not None) else None),
        "curve": mt.erle_curve(w, fa, onset, fc),
        "curve_after_change": mt.erle_curve(w, fa, fc, end) if t_change is not None else None,
        "p10_after_5s": (round(float(np.percentile(w["erle"][(w["start"] >= onset + 500) & (w["start"] + 20 <= fc)], 10)), 2)
                         if np.any((w["start"] >= onset + 500) & (w["start"] + 20 <= fc)) else None),
    }
    msk = mt.measurable(cap, arr["clean"], w)
    wm = {"start": w["start"][msk], "erle": w["erle"][msk]}
    res["erle"]["meas"] = {
        "enr_db_min": 25.0, "windows": int(msk.sum()),
        "ss": mt.median_in(wm, fa, onset, fc, 5.0),
        "ss_after_change": mt.median_in(wm, fa, fc, end, 5.0) if t_change is not None else None,
        "conv20_s": mt.convergence(wm, fa, onset, fc, 20.0),
        "conv20_after_change_s": mt.convergence(wm, fa, fc, end, 20.0) if t_change is not None else None,
        "curve": mt.erle_curve(wm, fa, onset, fc),
        "curve_after_change": mt.erle_curve(wm, fa, fc, end) if t_change is not None else None,
    }
    res["post_dt_meas"] = mt.post_dt(wm, meta, fa)
    p_after = silero.probs(out)
    onsets = meta["playout_onsets"]
    res["silero_before"] = mt.silero_metrics(arr["silero_before"], eo, fa, npres, onsets, t_change)
    res["silero_after"] = mt.silero_metrics(p_after, eo, fa, npres, onsets, t_change)
    res["near"] = mt.near_metrics(arr, out, meta, spectral=spectral)
    res["post_dt"] = mt.post_dt(w, meta, fa)
    res["cpu"] = {
        "engine_thread_ms_per_audio_s": round(tt.sum() / 1e6 / dur, 3),
        "engine_process_ms_per_audio_s": round(proc_ns / 1e6 / dur, 3),
        "frame_thread_us": c.stats(tt[100:] / 1e3, 2),
        "frame_wall_us": c.stats(wall[100:] / 1e3, 2),
        "loadavg": c.loadavg(),
    }
    if gcc_info is not None:
        gcc_info["cpu_ms_per_audio_s"] = round(gcc_info["cpu_ms"] / dur, 3)
        res["gcc"] = gcc_info
    if asr_dir is not None:
        asr_dir.mkdir(parents=True, exist_ok=True)
        for k, u in enumerate(meta["near"]):
            a = max(0, int((u["start"] - 0.3) * c.SR))
            b = min(len(out), int((u["end"] + 0.3) * c.SR))
            c.write_wav(asr_dir / f"{meta['cond']['cid']}__{k:02d}.wav", out[a:b], c.SR)
    return res


def main() -> None:
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    silero = c.Silero()
    cur = job_path.with_suffix(".current")
    for npz in job["calls"]:
        npz = Path(npz)
        cid = npz.stem
        dst = out_dir / f"{cid}.json"
        if dst.exists():
            continue
        cur.write_text(cid)
        t0 = time.perf_counter()
        try:
            asr_dir = Path(job["asr_dir"]) if job.get("asr_dir") else None
            res = run_call(job["engine"], job["align"], npz, silero, asr_dir, bool(job.get("spectral")))
            res["wall_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:  # noqa: BLE001 - record and continue with the next call
            import traceback

            res = {"cid": cid, "engine": job["engine"], "align": job["align"], "error": repr(e),
                   "trace": traceback.format_exc()[-2000:]}
        dst.write_text(json.dumps(res, default=c._json_default) + "\n")
        print(f"{job['engine']}/{job['align']} {cid} {res.get('wall_s')} s", flush=True)
    cur.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

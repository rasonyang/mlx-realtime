"""S11 part 2 driver: grids, cached call generation, engine runs in child processes, summaries.

    cd spikes/aec
    uv run python s11_run.py gen --part core          # build + cache the calls (spikes/data/s11/, git-ignored)
    uv run python s11_run.py run --part core          # every default variant, one child per batch
    uv run python s11_run.py run --part core --variants lk:engine,pw:engine
    uv run python s11_run.py summarize --part core    # -> spikes/results/s11_core.json

Parts: core, timeline, dt, slices (A3 section 5 "Grid"). Variants are engine:alignment
(s11_engines.py, s11_align.py). Children run one at a time (B4: run candidates sequentially) with
VECLIB_MAXIMUM_THREADS = OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 2. A child that dies (livekit
aborts with SIGABRT on a bad frame size) is recorded for the call it was on, and a new child
continues with the rest. Throwaway spike code.
"""

from __future__ import annotations

import s11_common as c

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import s11_chain as ch  # noqa: E402

PATHS = [f"m{i}" for i in range(1, 9)]
RESULTS_DIR = c.CACHE / "results"
ASR_DIR = c.CACHE / "asr"
JOBS_DIR = c.CACHE / "jobs"


# ------------------------------------------------------------------------------------------- grids


def grid(part: str) -> list[ch.Cond]:
    out: list[ch.Cond] = []
    if part == "core":  # m1-m8 x ERL {6, 10, 15, 20} x D {20, 120, 300}, noise -65 dBm0, mu-law
        i = 0
        for p in PATHS:
            for erl in (6, 10, 15, 20):
                for d in (20, 120, 300):
                    out.append(ch.Cond(cid=f"core_{p}_e{erl}_d{d}", part="core", path=p, erl=erl, d_ms=d,
                                       lang=("zh", "en")[i % 2]))
                    i += 1
    elif part == "timeline":  # m4/m5, ERL 10, D {20, 120, 300} x eps {-200, -50, 0, 50, 200}, plus eps steps
        i = 0
        for p in ("m4", "m5"):
            for d in (20, 120, 300):
                for eps in (-200, -50, 0, 50, 200):
                    out.append(ch.Cond(cid=f"tl_{p}_d{d}_eps{eps:+d}", part="timeline", path=p, erl=10, d_ms=d,
                                       eps_ms=eps, lang=("zh", "en")[i % 2]))
                    i += 1
            for eps2 in (-200, -50, 50, 200):  # step at 20 s of model time, mid-call
                out.append(ch.Cond(cid=f"tl_{p}_d120_step{eps2:+d}", part="timeline", path=p, erl=10, d_ms=120,
                                   eps_ms=0, eps_step_t=20.0, eps2_ms=eps2, lang=("zh", "en")[i % 2]))
                i += 1
    elif part == "dt":  # ERL {6, 15} x near end 0 / -15 dB x 40 zh + 40 en near-end utterances
        k = 0
        for lang in ("zh", "en"):
            ids = ch.lang_ids(lang)[:40]
            for blk in range(4):
                near = ids[blk * 10:(blk + 1) * 10]
                for erl in (6, 15):
                    for rel in (0, -15):
                        p = PATHS[(blk * 2 + (0 if lang == "zh" else 1)) % 8]
                        out.append(ch.Cond(cid=f"dt_{lang}{blk}_e{erl}_n{-rel}", part="dt", layout="dt", path=p,
                                           erl=erl, d_ms=120, near_rel_db=rel, lang=lang, near_ids=near))
                        k += 1
    elif part == "slices":  # noise -40 dBm0; A-law; open path; RTF 1.1 (underruns)
        for p in ("m4", "m5"):
            out.append(ch.Cond(cid=f"sl_noise40_{p}", part="slices", path=p, erl=10, d_ms=120, noise_dbm0=-40.0, lang="zh"))
            out.append(ch.Cond(cid=f"sl_alaw_{p}", part="slices", path=p, erl=10, d_ms=120, law="alaw", lang="en"))
            out.append(ch.Cond(cid=f"sl_open_{p}", part="slices", path="open", erl=10, d_ms=120,
                               lang="zh" if p == "m4" else "en"))
            out.append(ch.Cond(cid=f"sl_rtf11_{p}", part="slices", path=p, erl=10, d_ms=120, rtf=1.1,
                               lang="en" if p == "m4" else "zh"))
            out.append(ch.Cond(cid=f"sl_noise40_e6_{p}", part="slices", path=p, erl=6, d_ms=120, noise_dbm0=-40.0,
                               lang="en"))
    else:
        raise ValueError(part)
    return out


DEFAULT_VARIANTS = {
    "core": ["lk:engine", "lknd:engine", "lk:locked", "lk:oracle", "lk_pf:engine", "pw:engine", "pw:locked",
             "pw:oracle", "sx1024_pf:gcc", "sx1024_pf:oracle", "sx2048_pf:gcc", "sx1024pp_pf:gcc",
             "sx1024pp_pf:gcc_locked", "sx1024pp_pf:oracle", "sx2048pp_pf:gcc", "bl03_pf:gcc", "bl05_pf:gcc",
             "bl05_pf:oracle"],
    # sx<taps>ppd = Speex preprocessor with denoise on: with denoise off (A3's pp variant) speexdsp applies no
    # spectral gain at all, so its residual echo suppression never acts (found in the core run).
    "timeline": ["lk:engine", "lk:gcc", "lk:locked", "lk:gcc_locked", "pw:engine", "pw:gcc", "pw:locked",
                 "pw:gcc_locked", "sx1024ppd_pf:gcc", "sx1024ppd_pf:gcc_locked"],
    "dt": ["none:engine", "lk:engine", "lk:locked", "pw:engine", "pw:locked", "sx1024_pf:gcc_locked",
           "sx1024ppd_pf:gcc_locked", "bl05_pf:gcc_locked"],
    "slices": ["lk:engine", "lk:locked", "lk:oracle", "pw:engine", "pw:locked", "sx1024ppd_pf:gcc",
               "sx1024ppd_pf:gcc_locked", "bl05_pf:gcc"],
}


# --------------------------------------------------------------------------------------------- gen


def cmd_gen(part: str, force: bool) -> None:
    conds = grid(part)
    sil = c.Silero()
    t0 = time.perf_counter()
    for i, cond in enumerate(conds):
        p = ch.cache_path(cond)
        if p.exists() and not force:
            continue
        ch.generate(cond, sil, force=force)
        if part == "dt":  # clean-chain segments: the ASR reference condition (no echo, no canceller)
            arr, meta = ch.load(p)
            d = ASR_DIR / "clean"
            for k, u in enumerate(meta["near"]):
                a = max(0, int((u["start"] - 0.3) * c.SR))
                b = min(len(arr["clean"]), int((u["end"] + 0.3) * c.SR))
                c.write_wav(d / f"{cond.cid}__{k:02d}.wav", arr["clean"][a:b], c.SR)
        print(f"[{i + 1}/{len(conds)}] {cond.cid} {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"gen {part}: {len(conds)} calls in {c.CACHE / part}")


# --------------------------------------------------------------------------------------------- run


def run_variant(part: str, variant: str, batch: int) -> dict:
    engine, align = variant.split(":")
    conds = grid(part)
    out_dir = RESULTS_DIR / part / f"{engine}__{align}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [ch.cache_path(cd) for cd in conds if not (out_dir / f"{cd.cid}.json").exists()]
    missing = [p for p in todo if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} calls not generated; run `gen --part {part}` first")
    crashes = []
    t0 = time.perf_counter()
    while todo:
        chunk = todo[:batch]
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        job = JOBS_DIR / f"{part}__{engine}__{align}.json"
        job.write_text(json.dumps({"engine": engine, "align": align, "out_dir": str(out_dir),
                                   "asr_dir": str(ASR_DIR / f"{engine}__{align}") if part == "dt" else None,
                                   "spectral": part in ("dt", "slices", "timeline"),
                                   "calls": [str(p) for p in chunk]}))
        cur = job.with_suffix(".current")
        cur.unlink(missing_ok=True)
        proc = subprocess.run([sys.executable, str(c.HERE / "s11_child.py"), str(job)], cwd=str(c.HERE),
                              capture_output=True, text=True, env=dict(os.environ))
        if proc.returncode != 0:
            cid = cur.read_text().strip() if cur.exists() else None
            rec = {"cid": cid, "engine": engine, "align": align, "crash": {"returncode": proc.returncode,
                                                                             "stderr_tail": proc.stderr[-3000:]}}
            print(f"CRASH {variant} {cid} rc={proc.returncode}: {proc.stderr[-400:]}", flush=True)
            crashes.append(rec)
            if cid is not None:
                (out_dir / f"{cid}.json").write_text(json.dumps(rec) + "\n")
        todo = [p for p in todo if not (out_dir / f"{p.stem}.json").exists()]
        done = len(conds) - len(todo)
        print(f"{part} {variant}: {done}/{len(conds)} ({time.perf_counter() - t0:.0f} s, load {c.loadavg()[0]})", flush=True)
    return {"variant": variant, "crashes": crashes, "wall_s": round(time.perf_counter() - t0, 1)}


def cmd_run(part: str, variants: list[str], batch: int) -> None:
    log = []
    for v in variants:
        log.append(run_variant(part, v, batch))
    p = RESULTS_DIR / part / "_runlog.json"
    old = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(old + [{"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": log}], indent=1))


# ------------------------------------------------------------------------------------- summarize


def _q(xs, q):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, q)), 3) if xs else None


def _frac(xs, pred):
    xs = [x for x in xs]
    return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None


def load_results(part: str) -> dict[str, list[dict]]:
    out = {}
    base = RESULTS_DIR / part
    for d in sorted(base.glob("*__*")):
        if not d.is_dir():
            continue
        rs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        out[d.name.replace("__", ":")] = rs
    return out


_NEAR_END: dict[str, dict[str, float]] = {}


def _post_dt_valid(r: dict, x: dict) -> bool:
    """Results written before post_dt learned about the path change (the first core run) may hold
    'after' windows past the change at 45 s: keep only bursts that end >= 2 s before it."""
    t_change = r["meta"].get("t_change")
    if t_change is None:
        return True
    end = x.get("end")
    if end is None:
        if r["cid"] not in _NEAR_END:
            cond = next(cd for p in ("core", "timeline", "slices") for cd in grid(p) if cd.cid == r["cid"])
            _, meta = ch.load(ch.cache_path(cond))
            _NEAR_END[r["cid"]] = {u["utt"]: u["end"] for u in meta["near"]}
        end = _NEAR_END[r["cid"]].get(x["utt"])
    return end is not None and end <= t_change - 2.0


def summarize_variant(rs: list[dict], part: str) -> dict:
    ok = [r for r in rs if "erle" in r]
    bad = [r for r in rs if "erle" not in r]
    s: dict = {"calls": len(rs), "ok": len(ok), "errors": [{"cid": r.get("cid"), "error": r.get("error"),
                                                             "crash": r.get("crash")} for r in bad]}
    if not ok:
        return s
    E = [r["erle"] for r in ok]
    sa = [r["silero_after"] for r in ok]
    sb = [r["silero_before"] for r in ok]
    erl = [r["meta"]["erl_eff"] for r in ok]
    s["erle_ss_db"] = {"p10": _q([e["ss"] for e in E], 10), "p50": _q([e["ss"] for e in E], 50),
                       "p90": _q([e["ss"] for e in E], 90)}
    if part != "dt":
        s["erle_ss_after_change_db"] = {"p10": _q([e["ss_after_change"] for e in E], 10),
                                        "p50": _q([e["ss_after_change"] for e in E], 50)}
        s["conv20_s"] = {"p50": _q([e["conv20_s"] for e in E], 50), "p90": _q([e["conv20_s"] for e in E], 90),
                         "frac_le_1s": _frac([e["conv20_s"] for e in E], lambda x: x is not None and x <= 1.0),
                         "frac_never": _frac([e["conv20_s"] for e in E], lambda x: x is None)}
        s["conv20_after_change_s"] = {
            "p50": _q([e["conv20_after_change_s"] for e in E], 50), "p90": _q([e["conv20_after_change_s"] for e in E], 90),
            "frac_le_1s": _frac([e["conv20_after_change_s"] for e in E], lambda x: x is not None and x <= 1.0),
            "frac_never": _frac([e["conv20_after_change_s"] for e in E], lambda x: x is None)}
        s["conv_lret20_s"] = {"p50": _q([e["conv_lret20_s"] for e in E], 50),
                              "frac_le_1s": _frac([e["conv_lret20_s"] for e in E], lambda x: x is not None and x <= 1.0)}
        s["conv_lret20_after_change_s"] = {
            "p50": _q([e["conv_lret20_after_change_s"] for e in E], 50),
            "frac_le_1s": _frac([e["conv_lret20_after_change_s"] for e in E], lambda x: x is not None and x <= 1.0)}
        s["erl_plus_erle_ss_ge_27"] = _frac([(e["ss"] or 0) + (l or 0) for e, l in zip(E, erl)], lambda x: x >= 27)
        s["erle_curve_p50"] = [_q([e["curve"][i] for e in E], 50) for i in range(len(E[0]["curve"]))]
        s["erle_curve_after_change_p50"] = ([_q([e["curve_after_change"][i] for e in E], 50)
                                             for i in range(len(E[0]["curve_after_change"]))]
                                            if E[0]["curve_after_change"] else None)
    for name, key in (("rate_after_1s", "after_1s"), ("rate_after_2s", "after_2s"),
                      ("rate_first_1s_of_playbacks", "first_1s_of_playbacks"), ("rate_change_first_2s", "change_first_2s"),
                      ("rate_all_eo", "all_eo")):
        vals = [x[key]["rate"] for x in sa]
        s[f"silero_{name}"] = {"p50": _q(vals, 50), "p90": _q(vals, 90), "max": _q(vals, 100),
                               "frac_calls_le_1pct": _frac(vals, lambda v: v is not None and v <= 0.01)}
    s["silero_before_rate_all_eo_p50"] = _q([x["all_eo"]["rate"] for x in sb], 50)
    s["silero_before_false_starts_150"] = sum(x["speech_started"]["150"]["false_total"] for x in sb)
    for ms in ("150", "400"):
        ev = [x["speech_started"][ms] for x in sa]
        since = [e["since_onset_s"] for x in ev for e in x["events"] if e["since_onset_s"] is not None]
        s[f"false_starts_{ms}"] = {
            "total": sum(x["false_total"] for x in ev),
            "after_1s": sum(x["false_after_1s"] for x in ev),
            "after_2s": sum(x["false_after_2s"] for x in ev),
            "calls_with_any_after_1s": sum(1 for x in ev if x["false_after_1s"] > 0),
            "calls_with_any_after_2s": sum(1 for x in ev if x["false_after_2s"] > 0),
            "since_playback_onset_s": {"n": len(since), "p50": _q(since, 50), "p90": _q(since, 90),
                                       "max": _q(since, 100),
                                       "within_1s": sum(1 for v in since if v <= 1.0),
                                       "within_2s": sum(1 for v in since if v <= 2.0),
                                       "within_3s": sum(1 for v in since if v <= 3.0)},
        }
    # barge_in defaults: false starts left after an onset guard G (measured from the playback start the
    # gateway models, i.e. D before the echo reaches the capture) and with min_speech 150 / 400 ms
    guard = {}
    for ms in ("150", "400"):
        rows = []
        for r in ok:
            d_s = (r["meta"].get("residual_delay_ms_nominal") or 0) / 1000.0 - ch.LEAD_S  # ~ D + tick offset
            for e in r["silero_after"]["speech_started"][ms]["events"]:
                if e["since_onset_s"] is not None:
                    rows.append((r["cid"], e["since_onset_s"] + d_s))
        guard[ms] = {f"{g}ms": {"false_starts": sum(1 for _, t in rows if t * 1000 > g),
                                "calls": len({cid for cid, t in rows if t * 1000 > g})}
                     for g in (0, 500, 1000, 1500, 2000, 2500, 3000, 4000)}
    s["false_starts_vs_echo_guard"] = guard
    nm = [r["near"] for r in ok]
    s["near_single_talk_gain_db"] = {"p50": _q([n.get("single_talk_gain_db") for n in nm], 50),
                                     "worst_utt_p10": _q([n.get("single_talk_gain_db_worst_utt") for n in nm], 10)}
    s["dt_sdr_db"] = {"out_p50": _q([n.get("dt_sdr_out_db") for n in nm], 50),
                      "in_p50": _q([n.get("dt_sdr_in_db") for n in nm], 50),
                      "delta_p50": _q([(n["dt_sdr_out_db"] - n["dt_sdr_in_db"]) for n in nm
                                       if n.get("dt_sdr_out_db") is not None and n.get("dt_sdr_in_db") is not None], 50)}
    if any("dt_spec_sdr_out_db" in n or "st_spec_sdr_out_db" in n for n in nm):
        s["spec"] = {"dt_out_p50": _q([n.get("dt_spec_sdr_out_db") for n in nm], 50),
                     "dt_in_p50": _q([n.get("dt_spec_sdr_in_db") for n in nm], 50),
                     "st_out_p50": _q([n.get("st_spec_sdr_out_db") for n in nm], 50),
                     "st_in_p50": _q([n.get("st_spec_sdr_in_db") for n in nm], 50)}
    pd = [x for r in ok for x in r["post_dt"] if _post_dt_valid(r, x)]
    s["post_dt_drop_db"] = {"n": len(pd), "p50": _q([x["drop_db"] for x in pd], 50), "p90": _q([x["drop_db"] for x in pd], 90),
                            "max": _q([x["drop_db"] for x in pd], 100)}
    cpu = [r["cpu"] for r in ok]
    s["cpu"] = {"engine_ms_per_audio_s_p50": _q([x["engine_thread_ms_per_audio_s"] for x in cpu], 50),
                "frame_us_p99_p50": _q([x["frame_thread_us"]["p99"] for x in cpu], 50),
                "frame_us_max_max": _q([x["frame_thread_us"]["max"] for x in cpu], 100),
                "loadavg1_p50": _q([x["loadavg"][0] for x in cpu], 50)}
    g = [r["gcc"] for r in ok if "gcc" in r]
    if g:
        s["gcc"] = {"cpu_ms_per_audio_s_p50": _q([x["cpu_ms_per_audio_s"] for x in g], 50),
                    "first_applied_t_p50": _q([x["first_applied_t"] for x in g], 50),
                    "first_applied_t_p90": _q([x["first_applied_t"] for x in g], 90),
                    "never_applied": sum(1 for x in g if x["first_applied_t"] is None),
                    "reshifts_p50": _q([len(x["applied"]) for x in g], 50),
                    "reshifts_max": _q([len(x["applied"]) for x in g], 100)}
    M = [(r["erle"]["meas"], r["meta"]["erl_eff"]) for r in ok if "meas" in r["erle"]]
    if M:
        mm = {"calls": len(M), "erle_ss_db": {"p10": _q([m["ss"] for m, _ in M], 10), "p50": _q([m["ss"] for m, _ in M], 50)}}
        if part != "dt":
            for key in ("conv20_s", "conv20_after_change_s"):
                vals = [m[key] for m, _ in M]
                mm[key] = {"p50": _q(vals, 50), "p90": _q(vals, 90),
                           "frac_le_1s": _frac(vals, lambda x: x is not None and x <= 1.0),
                           "frac_le_2s": _frac(vals, lambda x: x is not None and x <= 2.0),
                           "frac_never": _frac(vals, lambda x: x is None)}
            mm["erle_ss_after_change_db_p50"] = _q([m["ss_after_change"] for m, _ in M], 50)
            mm["curve_p50"] = [_q([m["curve"][i] for m, _ in M], 50) for i in range(len(M[0][0]["curve"]))]
            if M[0][0]["curve_after_change"]:
                mm["curve_after_change_p50"] = [_q([m["curve_after_change"][i] for m, _ in M], 50)
                                                for i in range(len(M[0][0]["curve_after_change"]))]
            mm["c3_calls_pass"] = (f"{sum(1 for m, l in M if m['conv20_s'] is not None and m['conv20_s'] <= 1.0 and m['conv20_after_change_s'] is not None and m['conv20_after_change_s'] <= 1.0 and (m['ss'] or 0) + (l or 0) >= 27)}"
                                   f"/{len(M)}")
        pdm = [x for r in ok for x in r.get("post_dt_meas", []) if _post_dt_valid(r, x)]
        mm["post_dt_drop_db"] = {"n": len(pdm), "p50": _q([x["drop_db"] for x in pdm], 50),
                                 "p90": _q([x["drop_db"] for x in pdm], 90), "max": _q([x["drop_db"] for x in pdm], 100),
                                 "n_gt_10db": sum(1 for x in pdm if x["drop_db"] > 10),
                                 "n_gt_3db": sum(1 for x in pdm if x["drop_db"] > 3)}
        s["measurable_windows"] = mm
    # G7 criteria 1 and 3, per call (core grid; |eps| <= 50 ms)
    c1 = [x["after_1s"]["rate"] is not None and x["after_1s"]["rate"] <= 0.01 and x["speech_started"]["150"]["false_after_1s"] == 0
          for x in sa]
    c3 = [(e["conv20_s"] is not None and e["conv20_s"] <= 1.0) and
          (e["conv20_after_change_s"] is not None and e["conv20_after_change_s"] <= 1.0) and
          ((e["ss"] or 0) + (l or 0) >= 27) for e, l in zip(E, erl)] if part != "dt" else []
    s["criteria"] = {"c1_echo_only_calls_pass": f"{sum(c1)}/{len(c1)}",
                     "c3_convergence_calls_pass": f"{sum(c3)}/{len(c3)}" if c3 else None}
    return s


def by_group(rs: list[dict], key) -> dict:
    groups: dict = {}
    for r in rs:
        groups.setdefault(key(r), []).append(r)
    return groups


def cmd_summarize(part: str) -> None:
    res = load_results(part)
    out = {"spike": "S11 part 2", "part": part, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "environment": c.environment(), "grid": [vars(cd) for cd in grid(part)], "variants": {}}
    for v, rs in res.items():
        s = summarize_variant(rs, part)
        if part in ("core", "slices", "timeline"):
            if part == "core":
                s["by_erl"] = {k: summarize_variant(g, part) for k, g in sorted(by_group(rs, lambda r: r["cid"].split("_")[2]).items())}
                s["by_d"] = {k: summarize_variant(g, part) for k, g in sorted(by_group(rs, lambda r: r["cid"].split("_")[3]).items())}
                for sub in list(s["by_erl"].values()) + list(s["by_d"].values()):
                    sub.pop("errors", None)
            if part in ("slices", "timeline"):
                s["by_cond"] = {r["cid"]: _call_brief(r) for r in rs}
        if part == "timeline":
            s["by_eps"] = {k: summarize_variant(g, part) for k, g in sorted(by_group(rs, lambda r: r["cid"].split("_")[3]).items())}
        out["variants"][v] = s
    p = c.save_json(f"s11_{part}.json", out)
    print(f"wrote {p}")


def _call_brief(r: dict) -> dict:
    if "erle" not in r:
        return {"error": r.get("error") or r.get("crash")}
    sa = r["silero_after"]
    return {"erle_ss": r["erle"]["ss"], "erle_ss_after_change": r["erle"]["ss_after_change"],
            "conv20_s": r["erle"]["conv20_s"], "conv20_after_change_s": r["erle"]["conv20_after_change_s"],
            "silero_after_1s": sa["after_1s"]["rate"], "silero_after_2s": sa["after_2s"]["rate"],
            "silero_all_eo": sa["all_eo"]["rate"],
            "false_starts_150": sa["speech_started"]["150"]["false_total"],
            "false_starts_150_after_1s": sa["speech_started"]["150"]["false_after_1s"],
            "false_starts_150_after_2s": sa["speech_started"]["150"]["false_after_2s"],
            "false_starts_400": sa["speech_started"]["400"]["false_total"],
            "residual_delay_ms_nominal": r["meta"]["residual_delay_ms_nominal"],
            "near_single_talk_gain_db": r["near"].get("single_talk_gain_db"),
            "gcc_final_shift_ms": r.get("gcc", {}).get("final_shift_ms"),
            "gcc_first_applied_t": r.get("gcc", {}).get("first_applied_t")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--part", required=True)
    g.add_argument("--force", action="store_true")
    r = sp.add_parser("run")
    r.add_argument("--part", required=True)
    r.add_argument("--variants", default=None)
    r.add_argument("--batch", type=int, default=24)
    s = sp.add_parser("summarize")
    s.add_argument("--part", required=True)
    a = ap.parse_args()
    if a.cmd == "gen":
        cmd_gen(a.part, a.force)
    elif a.cmd == "run":
        vs = a.variants.split(",") if a.variants else DEFAULT_VARIANTS[a.part]
        if not c.speex_available():
            print("SpeexDSP library not found: skipping Speex variants", flush=True)
            vs = [v for v in vs if not v.startswith("sx")]
        cmd_run(a.part, vs, a.batch)
    elif a.cmd == "summarize":
        cmd_summarize(a.part)


if __name__ == "__main__":
    main()

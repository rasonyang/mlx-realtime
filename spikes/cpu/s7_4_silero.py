"""S7.4 — Silero VAD v6.2.3 (numpy wrapper, onnxruntime CPU): cost, and 8 kHz native vs upsampled 16 kHz.

    uv run python s7_4_silero.py [--stream-clips 300]

A. Cost. One call = one 32 ms window (512 samples + 64 context at 16 kHz; 256 + 32 at 8 kHz).
   B = 1 through the pipecat-style wrapper (what one session pays per window), and batched calls
   with B = 4, 16, 64 (per call and per stream-window). 2000 timed calls after 200 warm-up.
B. Telephone audio, per clip with fresh state. The Chinese Smart Turn test clips are degraded
   16 kHz -> 8 kHz (soxr HQ) -> mu-law round trip; then
     tel8  = Silero's native 8 kHz mode on the 8 kHz telephone audio,
     tel16 = the same audio upsampled to 16 kHz (soxr HQ, one-shot so content stays aligned),
     clean = the original 16 kHz clip (reference).
   Window i covers [32 i, 32 i + 32) ms in all three. Reported: frame agreement (p >= 0.5 and
   hysteresis 0.5 / 0.35), mean |dp|, and delays against clean for the first speech window, for
   speech confirmed after min_speech 150 ms (5 windows), and for the last speech window.
C. Same, but one continuous stream per path (one state): each clip is preceded by 2 s of white
   noise at -50 dBFS (added before the telephone chain), which exposes state drift. Run without
   resets, and with pipecat v1.13.0's periodic reset policy (see probs()).
Writes spikes/results/s7_4_silero.json.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SILERO_FILE,
    environment,
    load_guard,
    load_zh_rows,
    model_path,
    ort_session,
    save_json,
    stats_ms,
    telephone_degrade,
)
from silero_vad import BatchedSilero, SileroOnnxModel  # noqa: E402

ON, OFF = 0.5, 0.35
MIN_SPEECH_WIN = math.ceil(150 / 32)  # 5 windows
WIN_MS = 32


# ------------------------------------------------------------------ A. cost


def cost(frames16: np.ndarray, frames8: np.ndarray) -> dict:
    out: dict = {"protocol": "2000 timed calls after 200 warm-up; intra_op 1 unless stated"}
    with load_guard(out):
        sess1 = ort_session(model_path(SILERO_FILE), intra=1, spinning=False)
        for sr, frames in ((16000, frames16), (8000, frames8)):
            m = SileroOnnxModel(sess1)
            ns = []
            for i in range(2200):
                a = time.perf_counter_ns()
                m(frames[i % len(frames)], sr)
                if i >= 200:
                    ns.append(time.perf_counter_ns() - a)
            out[f"B1_wrapper_{sr // 1000}k"] = stats_ms(ns)
        runs = []
        for intra in (1, 2, 4):
            sess = sess1 if intra == 1 else ort_session(model_path(SILERO_FILE), intra=intra, spinning=False)
            for b in (1, 4, 16, 64):
                if intra > 1 and b < 16:
                    continue
                bs = BatchedSilero(sess, b)
                idx = np.arange(b)
                ns = []
                for i in range(2200):
                    k = (i * b) % (len(frames16) - b)
                    x = frames16[k : k + b]
                    a = time.perf_counter_ns()
                    bs.run(idx, x)
                    if i >= 200:
                        ns.append(time.perf_counter_ns() - a)
                s = stats_ms(ns)
                runs.append({"B": b, "intra_op": intra, "per_call_ms": s,
                             "per_stream_window_us_p50": round(s["p50"] * 1000 / b, 2)})
        out["batched_16k"] = runs
    return out


# ------------------------------------------------------- B/C. 8 kHz vs 16 kHz


def probs(m: SileroOnnxModel, audio: np.ndarray, sr: int, reset_policy: bool = False) -> np.ndarray:
    """Per-window probabilities. reset_policy mimics pipecat v1.13.0 SileroVADAnalyzer: reset the
    model state once >= 2 s of audio has passed since the last reset, the VAD is quiet (here: the
    0.5/0.35 hysteresis is off) and the last 5 windows all scored < 0.1."""
    n = 512 if sr == 16000 else 256
    nf = len(audio) // n
    out = np.empty(nf, dtype=np.float32)
    since, low, on = 0, 0, False
    for i in range(nf):
        p = float(m(audio[i * n : (i + 1) * n], sr)[0, 0])
        out[i] = p
        if reset_policy:
            since += 1
            low = low + 1 if p < 0.1 else 0
            if since * n >= 2.0 * sr and not on and low >= 5:
                m.reset_states()
                since = 0
        if not on and p >= ON:
            on = True
        elif on and p < OFF:
            on = False
    return out


def hysteresis(p: np.ndarray) -> np.ndarray:
    s = np.zeros(len(p), dtype=bool)
    on = False
    for i, v in enumerate(p):
        if not on and v >= ON:
            on = True
        elif on and v < OFF:
            on = False
        s[i] = on
    return s


def events(p: np.ndarray) -> dict:
    """Times in ms from the clip start: first speech window, confirmed start, end of last speech."""
    h = hysteresis(p)
    if not h.any():
        return {"first": None, "confirmed": None, "end": None}
    first = int(np.argmax(h))
    run, confirmed = 0, None
    for i, v in enumerate(h):
        run = run + 1 if v else 0
        if run >= MIN_SPEECH_WIN:
            confirmed = (i + 1) * WIN_MS
            break
    last = len(h) - 1 - int(np.argmax(h[::-1]))
    return {"first": first * WIN_MS, "confirmed": confirmed, "end": (last + 1) * WIN_MS}


def compare(pa: np.ndarray, pb: np.ndarray) -> dict:
    n = min(len(pa), len(pb))
    pa, pb = pa[:n], pb[:n]
    ra, rb = pa >= ON, pb >= ON
    ha, hb = hysteresis(pa), hysteresis(pb)
    return {"n": n, "agree_raw": int(np.sum(ra == rb)), "agree_hyst": int(np.sum(ha == hb)),
            "speech_a": int(ha.sum()), "speech_b": int(hb.sum()), "sum_abs_dp": float(np.abs(pa - pb).sum())}


def summarize(per_clip: list[dict], ref: str, path: str) -> dict:
    tot = {"n": 0, "agree_raw": 0, "agree_hyst": 0, "speech_a": 0, "speech_b": 0, "sum_abs_dp": 0.0}
    delays = {"first": [], "confirmed": [], "end": []}
    missed = {"first": 0, "confirmed": 0}
    extra = {"first": 0}
    for c in per_clip:
        cmpd = c["cmp"][f"{ref}|{path}"]
        for k in tot:
            tot[k] += cmpd[k]
        er, ep = c["events"][ref], c["events"][path]
        for k in delays:
            if er[k] is not None and ep[k] is not None:
                delays[k].append(ep[k] - er[k])
        for k in missed:
            if er[k] is not None and ep[k] is None:
                missed[k] += 1
        if er["first"] is None and ep["first"] is not None:
            extra["first"] += 1

    def dstats(v):
        if not v:
            return None
        a = np.array(v, dtype=float)
        return {"n": len(v), "mean": round(float(a.mean()), 1), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max()),
                "share_later": round(float(np.mean(a > 0)), 3), "share_earlier": round(float(np.mean(a < 0)), 3)}

    return {
        "reference": ref, "path": path, "windows": tot["n"],
        "agreement_raw_pct": round(100 * tot["agree_raw"] / tot["n"], 2),
        "agreement_hysteresis_pct": round(100 * tot["agree_hyst"] / tot["n"], 2),
        "speech_windows_ref": tot["speech_a"], "speech_windows_path": tot["speech_b"],
        "mean_abs_dp": round(tot["sum_abs_dp"] / tot["n"], 4),
        "delay_ms_first_speech": dstats(delays["first"]),
        "delay_ms_confirmed_start": dstats(delays["confirmed"]),
        "delay_ms_end_of_speech": dstats(delays["end"]),
        "missed_onsets": missed, "extra_onsets": extra,
    }


def per_clip_eval(sess, rows) -> tuple[list[dict], dict]:
    per_clip = []
    for r in rows:
        clean = r["audio"]
        tel8, _ = telephone_degrade(clean)
        tel16 = soxr.resample(tel8, 8000, 16000, quality="HQ").astype(np.float32)
        p = {
            "clean": probs(SileroOnnxModel(sess), clean, 16000),
            "tel16": probs(SileroOnnxModel(sess), tel16, 16000),
            "tel8": probs(SileroOnnxModel(sess), tel8, 8000),
        }
        per_clip.append({
            "row_idx": r["row_idx"],
            "events": {k: events(v) for k, v in p.items()},
            "cmp": {"clean|tel16": compare(p["clean"], p["tel16"]), "clean|tel8": compare(p["clean"], p["tel8"]),
                    "tel16|tel8": compare(p["tel16"], p["tel8"])},
        })
    summary = {k: summarize(per_clip, *k.split("|")) for k in ("clean|tel16", "clean|tel8", "tel16|tel8")}
    return per_clip, summary


def stream_eval(sess, rows, n_clips: int, reset_policy: bool = False) -> dict:
    rng = np.random.default_rng(11)
    gap = 2 * 16000  # 2 s = 62.5... -> use 2.016 s so every clip starts on a 512 boundary
    gap = (gap // 512 + 1) * 512
    pieces, starts, lens, pos = [], [], [], 0
    for r in rows[:n_clips]:
        noise = (rng.standard_normal(gap) * 10 ** (-50 / 20)).astype(np.float32)
        a = r["audio"]
        pad = (-len(a)) % 512
        a = np.concatenate([a, (rng.standard_normal(pad) * 10 ** (-50 / 20)).astype(np.float32)])
        pieces += [noise, a]
        starts.append(pos + gap)
        lens.append(len(a))
        pos += gap + len(a)
    clean = np.concatenate(pieces)
    tel8, _ = telephone_degrade(clean)
    tel16 = soxr.resample(tel8, 8000, 16000, quality="HQ").astype(np.float32)
    p = {"clean": probs(SileroOnnxModel(sess), clean, 16000, reset_policy),
         "tel16": probs(SileroOnnxModel(sess), tel16, 16000, reset_policy),
         "tel8": probs(SileroOnnxModel(sess), tel8, 8000, reset_policy)}
    per_clip = []
    for (s, n), r in zip(zip(starts, lens), rows[:n_clips]):
        w0, w1 = s // 512, (s + n) // 512
        seg = {k: v[w0:w1] for k, v in p.items()}
        per_clip.append({
            "row_idx": r["row_idx"],
            "events": {k: events(v) for k, v in seg.items()},
            "cmp": {"clean|tel16": compare(seg["clean"], seg["tel16"]), "clean|tel8": compare(seg["clean"], seg["tel8"]),
                    "tel16|tel8": compare(seg["tel16"], seg["tel8"])},
        })
    gaps = {}
    for k, v in p.items():  # speech windows inside the noise gaps (false alarms)
        mask = np.ones(len(v), dtype=bool)
        for s, n in zip(starts, lens):
            mask[s // 512 : (s + n) // 512] = False
        gaps[k] = {"gap_windows": int(mask.sum()), "speech_windows_hyst": int(hysteresis(v)[mask].sum())}
    return {"clips": n_clips, "gap_ms": gap / 16, "noise_dbfs": -50, "reset_policy": reset_policy,
            "summary": {k: summarize(per_clip, *k.split("|")) for k in ("clean|tel16", "clean|tel8", "tel16|tel8")},
            "speech_in_gaps": gaps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-clips", type=int, default=300)
    args = ap.parse_args()
    rows = load_zh_rows()

    # Cost inputs: real speech windows.
    cat16 = np.concatenate([r["audio"] for r in rows[:40]])
    frames16 = cat16[: len(cat16) // 512 * 512].reshape(-1, 512)
    cat8 = soxr.resample(cat16, 16000, 8000, quality="HQ").astype(np.float32)
    frames8 = cat8[: len(cat8) // 256 * 256].reshape(-1, 256)
    c = cost(frames16, frames8)
    print(f"cost B1 wrapper 16k p50 {c['B1_wrapper_16k']['p50']:.4f} ms p95 {c['B1_wrapper_16k']['p95']:.4f}; "
          f"8k p50 {c['B1_wrapper_8k']['p50']:.4f} ms", flush=True)
    for r in c["batched_16k"]:
        print(f"  B={r['B']:3d} intra={r['intra_op']} per call p50 {r['per_call_ms']['p50']:.4f} ms "
              f"p95 {r['per_call_ms']['p95']:.4f}; per stream-window {r['per_stream_window_us_p50']} us", flush=True)

    sess = ort_session(model_path(SILERO_FILE), intra=1, spinning=False)
    t0 = time.time()
    per_clip, summary = per_clip_eval(sess, rows)
    print(f"per-clip eval {time.time() - t0:.0f}s", flush=True)
    for k, s in summary.items():
        print(f"  {k}: agree raw {s['agreement_raw_pct']}% hyst {s['agreement_hysteresis_pct']}% mean|dp| {s['mean_abs_dp']} "
              f"first {s['delay_ms_first_speech'] and s['delay_ms_first_speech']['mean']} "
              f"confirmed {s['delay_ms_confirmed_start'] and s['delay_ms_confirmed_start']['mean']} "
              f"end {s['delay_ms_end_of_speech'] and s['delay_ms_end_of_speech']['mean']} missed {s['missed_onsets']}",
              flush=True)
    streams = {}
    for reset in (False, True):
        t0 = time.time()
        stream = streams["pipecat_resets" if reset else "no_resets"] = stream_eval(sess, rows, args.stream_clips, reset)
        print(f"stream eval (resets={reset}) {time.time() - t0:.0f}s", flush=True)
        for k, s in stream["summary"].items():
            print(f"  stream {k}: agree hyst {s['agreement_hysteresis_pct']}% "
                  f"first {s['delay_ms_first_speech'] and s['delay_ms_first_speech']['mean']} "
                  f"confirmed {s['delay_ms_confirmed_start'] and s['delay_ms_confirmed_start']['mean']} "
                  f"missed {s['missed_onsets']}", flush=True)
        print("  speech in gaps:", stream["speech_in_gaps"], flush=True)

    save_json("s7_4_silero.json", {
        "spike": "S7.4 Silero VAD v6.2.3 cost and 8 kHz vs upsampled 16 kHz on telephone audio",
        "env": environment(),
        "cost": c,
        "telephone_per_clip": {"clips": len(rows), "thresholds": {"on": ON, "off": OFF, "min_speech_windows": MIN_SPEECH_WIN},
                               "summary": summary},
        "telephone_stream": streams,
        "per_clip_events": [{"row_idx": x["row_idx"], **x["events"]} for x in per_clip],
    })


if __name__ == "__main__":
    main()

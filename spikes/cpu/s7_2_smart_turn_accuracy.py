"""S7.2 — Smart Turn v3.2 accuracy on the Chinese rows of pipecat-ai/smart-turn-data-v3.2-test,
clean and telephone-degraded, for both model files, at threshold 0.5.

    uv run python s7_2_smart_turn_accuracy.py

Pipeline per clip (as the reference inference.py / pipecat): 16 kHz audio -> last 8 s, zero-padded
at the front -> numpy Whisper features (pipecat port) -> model. Label: endpoint_bool (True =
complete = positive class).
Telephone copy: 16 kHz -> 8 kHz (soxr HQ) -> G.711 mu-law round trip (audioop) -> 16 kHz (soxr HQ),
on the whole clip, then the same 8 s window.
Writes spikes/results/s7_2_smart_turn_accuracy.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ST_FILES,
    environment,
    last_8s_front_padded,
    load_zh_rows,
    model_path,
    ort_session,
    save_json,
    telephone_degrade,
)
from whisper_features import compute_whisper_log_mel_features  # noqa: E402

THRESHOLD = 0.5
BATCH = 32
# Published v3.2 Chinese ("zho") accuracy from the model card benchmarks (HF f766f81d).
PUBLISHED_ACC = {"v3.2-cpu": 85.79, "v3.2-gpu": 90.53}


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    pred = p > THRESHOLD
    pos = y == 1
    tp = int(np.sum(pred & pos))
    tn = int(np.sum(~pred & ~pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy_pct": round(100 * (tp + tn) / n, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        # Standard false-negative rate: missed ends of turn / all real ends of turn (= 1 - recall).
        "fnr_pct": round(100 * fn / (tp + fn), 2) if tp + fn else 0.0,
        "fpr_pct": round(100 * fp / (fp + tn), 2) if fp + tn else 0.0,
        # The smart-turn benchmark.py convention: FN / all samples, FP / all samples.
        "fn_over_all_pct": round(100 * fn / n, 2),
        "fp_over_all_pct": round(100 * fp / n, 2),
    }


def predict(sess, feats: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(feats), BATCH):
        out.append(sess.run(None, {"input_features": feats[i : i + BATCH]})[0][:, 0])
    return np.concatenate(out)


def main() -> None:
    t0 = time.time()
    rows = load_zh_rows()
    y = np.array([1 if r["endpoint_bool"] else 0 for r in rows])
    clean, tel = [], []
    for r in rows:
        clean.append(compute_whisper_log_mel_features(last_8s_front_padded(r["audio"])))
        _, x16 = telephone_degrade(r["audio"])
        tel.append(compute_whisper_log_mel_features(last_8s_front_padded(x16)))
    clean, tel = np.stack(clean), np.stack(tel)
    print(f"{len(rows)} clips, {int(y.sum())} complete; features in {time.time() - t0:.0f}s", flush=True)

    out = {}
    for name, fname in ST_FILES.items():
        sess = ort_session(model_path(fname), intra=4, spinning=False)
        pc, pt = predict(sess, clean), predict(sess, tel)
        mc, mt = metrics(pc, y), metrics(pt, y)
        d = np.abs(pt - pc)
        flips = (pc > THRESHOLD) != (pt > THRESHOLD)
        by_source = {}
        for src in sorted({r["dataset"] for r in rows}):
            idx = np.array([r["dataset"] == src for r in rows])
            by_source[src] = {"clean": metrics(pc[idx], y[idx]), "telephone": metrics(pt[idx], y[idx])}
        synth = np.array([bool(r["synthetic"]) for r in rows])
        out[name] = {
            "file": fname,
            "published_zho_accuracy_pct": PUBLISHED_ACC[name],
            "clean": mc,
            "telephone": mt,
            "delta_telephone_minus_clean": {
                k: round(mt[k] - mc[k], 4)
                for k in ("accuracy_pct", "precision", "recall", "f1", "fnr_pct", "fpr_pct", "fn_over_all_pct")
            },
            "mean_abs_dp": round(float(d.mean()), 4),
            "p95_abs_dp": round(float(np.percentile(d, 95)), 4),
            "max_abs_dp": round(float(d.max()), 4),
            "prediction_flips": int(flips.sum()),
            "flips_complete_to_incomplete": int(np.sum(flips & (pc > THRESHOLD))),
            "flips_incomplete_to_complete": int(np.sum(flips & (pc <= THRESHOLD))),
            "by_dataset_source": by_source,
            "synthetic_vs_human": {
                "synthetic": {"clean": metrics(pc[synth], y[synth]), "telephone": metrics(pt[synth], y[synth])},
                "human": {"clean": metrics(pc[~synth], y[~synth]), "telephone": metrics(pt[~synth], y[~synth])}
                if (~synth).any() else None,
            },
            "probabilities": {"row_idx": [r["row_idx"] for r in rows], "clean": pc.round(5), "telephone": pt.round(5)},
        }
        print(f"{name}: clean acc {mc['accuracy_pct']}% P {mc['precision']} R {mc['recall']} FNR {mc['fnr_pct']}% | "
              f"tel acc {mt['accuracy_pct']}% P {mt['precision']} R {mt['recall']} FNR {mt['fnr_pct']}% | "
              f"mean|dp| {d.mean():.4f}, flips {int(flips.sum())}", flush=True)

    # Agreement between the two files on the same inputs.
    pcc, pgc = np.array(out["v3.2-cpu"]["probabilities"]["clean"]), np.array(out["v3.2-gpu"]["probabilities"]["clean"])
    save_json(
        "s7_2_smart_turn_accuracy.json",
        {
            "spike": "S7.2 Smart Turn v3.2 accuracy, Chinese subset, clean vs telephone",
            "env": environment(),
            "dataset": {"repo": "pipecat-ai/smart-turn-data-v3.2-test", "revision": "dfc179241fee65b0efe81ca37786f41ce26f38ec",
                        "license": "CC-BY-4.0", "language": "zho", "n": len(rows), "complete": int(y.sum()),
                        "native_sample_rates": sorted({r["native_sr"] for r in rows}),
                        "durations_s": {"min": round(min(len(r["audio"]) for r in rows) / 16000, 2),
                                        "median": round(float(np.median([len(r["audio"]) for r in rows])) / 16000, 2),
                                        "max": round(max(len(r["audio"]) for r in rows) / 16000, 2)}},
            "threshold": THRESHOLD,
            "telephone_chain": "soxr HQ 16k->8k, audioop lin2ulaw/ulaw2lin, soxr HQ 8k->16k (whole clip)",
            "int8_vs_fp32_clean": {"mean_abs_dp": round(float(np.abs(pcc - pgc).mean()), 4),
                                    "decision_disagreements": int(np.sum((pcc > THRESHOLD) != (pgc > THRESHOLD)))},
            "models": out,
        },
    )


if __name__ == "__main__":
    main()

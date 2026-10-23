"""S7.3 — Feature parity: pipecat's numpy Whisper log-mel vs transformers' WhisperFeatureExtractor(chunk_length=8).

    uv run python s7_3_feature_parity.py [--clips 50]

On 50 Chinese test clips (seeded random choice; clean, plus their telephone copies for
information): max |delta| of the [80, 800] features, then max/mean |delta p| of both Smart Turn
files fed the two feature sets. transformers is called exactly as the smart-turn reference
inference.py does (padding="max_length", max_length=128000, truncation=True, do_normalize=True)
on the same front-padded 8 s window. transformers is a dev-only dependency; torch is not installed,
so transformers takes its numpy path (recorded in the output).
Writes spikes/results/s7_3_feature_parity.json.
"""

from __future__ import annotations

import argparse
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
    stats_ms,
    telephone_degrade,
)
from whisper_features import compute_whisper_log_mel_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=50)
    args = ap.parse_args()

    from transformers import WhisperFeatureExtractor
    from transformers.utils import is_torch_available

    fe = WhisperFeatureExtractor(chunk_length=8)
    rows = load_zh_rows()
    pick = sorted(np.random.default_rng(3).choice(len(rows), size=args.clips, replace=False))
    sessions = {name: ort_session(model_path(f), intra=4, spinning=False) for name, f in ST_FILES.items()}

    variants = {}
    for variant in ("clean", "telephone"):
        f_np, f_tf, t_np, t_tf = [], [], [], []
        for i in pick:
            audio = rows[i]["audio"]
            if variant == "telephone":
                audio = telephone_degrade(audio)[1]
            w = last_8s_front_padded(audio)
            a = time.perf_counter_ns()
            x_np = compute_whisper_log_mel_features(w)
            b = time.perf_counter_ns()
            x_tf = fe(w, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8 * 16000,
                      truncation=True, do_normalize=True).input_features[0].astype(np.float32)
            c = time.perf_counter_ns()
            f_np.append(x_np)
            f_tf.append(x_tf)
            t_np.append(b - a)
            t_tf.append(c - b)
        f_np, f_tf = np.stack(f_np), np.stack(f_tf)
        assert f_np.shape == f_tf.shape == (len(pick), 80, 800)
        d = np.abs(f_np.astype(np.float64) - f_tf.astype(np.float64))
        per_clip_max = d.reshape(len(pick), -1).max(axis=1)
        res = {
            "features_max_abs_delta": float(d.max()),
            "features_mean_abs_delta": float(d.mean()),
            "clips_bit_identical": int(np.sum(per_clip_max == 0)),
            "features_value_range": [float(f_tf.min()), float(f_tf.max())],
            "time_ms": {"pipecat_numpy": stats_ms(t_np), "transformers": stats_ms(t_tf)},
            "models": {},
        }
        for name, sess in sessions.items():
            p_np = sess.run(None, {"input_features": f_np})[0][:, 0]
            p_tf = sess.run(None, {"input_features": f_tf})[0][:, 0]
            dp = np.abs(p_np - p_tf)
            res["models"][name] = {
                "max_abs_dp": float(dp.max()),
                "mean_abs_dp": float(dp.mean()),
                "decision_flips_at_0.5": int(np.sum((p_np > 0.5) != (p_tf > 0.5))),
            }
        variants[variant] = res
        print(f"{variant}: features max|d| {res['features_max_abs_delta']:.3g} mean {res['features_mean_abs_delta']:.3g}, "
              f"identical clips {res['clips_bit_identical']}/{len(pick)}; "
              + "; ".join(f"{k} max|dp| {v['max_abs_dp']:.3g} mean {v['mean_abs_dp']:.3g} flips {v['decision_flips_at_0.5']}"
                          for k, v in res["models"].items()), flush=True)

    save_json("s7_3_feature_parity.json", {
        "spike": "S7.3 numpy Whisper features (pipecat port) vs transformers WhisperFeatureExtractor(chunk_length=8)",
        "env": environment(),
        "transformers_torch_available": is_torch_available(),
        "transformers_path": "numpy (_np_extract_fbank_features)" if not is_torch_available() else "torch",
        "clips": {"n": len(pick), "row_idx": [rows[i]["row_idx"] for i in pick], "selection": "rng(3).choice"},
        "results": variants,
    })


if __name__ == "__main__":
    main()

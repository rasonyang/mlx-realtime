"""Tune the speech-presence check of transcription commits (``transcription.min_speech_ms`` and
``transcription.speech_threshold``) on the corpus.

    uv run --locked --project spikes/transcription python spikes/transcription/gate_tune.py [--out spikes/results/transcription_gate_tune.json]

AICC's pump commits the tap audio of every leg (tap.py's rule); the real ``InputBuffer`` with the shipped
``Rolling`` settings decides what each commit holds; Silero (the gateway's model, 16 kHz, 32 ms frames) scores every
frame once. A commit is *speech* when it overlaps a reference utterance, otherwise *non-speech* (hold music, DTMF).
For each probability threshold p and each minimum of speech milliseconds m, a commit is skipped when fewer than m ms of its
frames reach p. The table: non-speech commits skipped (must be all), speech commits skipped (as few as possible), and
the skipped speech commits' texts. A speech commit skipped is a lost transcript.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

import common as c
from bufsim import overlaps, pump_commits
from mlx_realtime.audio.formats import PCM
from mlx_realtime.config import TranscriptionSessionConfig
from mlx_realtime.session.input_buffer import InputBuffer, Rolling
from mlx_realtime.vad.onnx import OnnxModel
from mlx_realtime.vad.silero import SileroStream

PROBS = [0.3, 0.4, 0.5, 0.6, 0.7]
MINS = [0, 32, 64, 96, 128, 160, 192, 256, 320]


def frame_probs(model: OnnxModel, pcm16k: bytes) -> list[float]:
    stream = SileroStream(model, reset_interval_s=None)
    return [stream.probability(pcm16k[k * 1024:(k + 1) * 1024], in_speech=False) for k in range(len(pcm16k) // 2 // 512)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    shipped = TranscriptionSessionConfig()
    model = OnnxModel(c.REPO / "models" / "silero_vad.onnx")
    commits = []
    for call in c.load_calls():
        meta = json.loads((c.DATA / "legs" / f"{call['id']}.json").read_text())
        for leg in ("agent", "customer"):
            pcm = c.read_wav(c.DATA / "legs" / f"{call['id']}_{leg}.wav")
            utts = [u for u in meta["utts"] if u["leg"] == leg]
            buf = InputBuffer(60000, input_format=PCM, rolling=Rolling(shipped.idle_keep_ms, shipped.voiced_rms))
            pos = 0
            for speech_t, commit_t, end_frame in pump_commits(pcm):
                for k in range(pos, end_frame):
                    buf.append(pcm[k * c.FRAME:(k + 1) * c.FRAME].astype("<i2").tobytes())
                pos = end_frame
                got = buf.commit(empty_ok=True)
                span = (speech_t - 1.0, commit_t)
                held = [u for u in utts if overlaps(*span, u["start"], u["end"])]
                kind = "speech" if held else ("music" if any(overlaps(*span, h["start"], h["end"]) for h in meta["holds"]) else "dtmf" if any(overlaps(*span, d["start"], d["end"]) for d in meta["dtmf"]) else "other")
                if not got.voiced:
                    continue
                commits.append({"kind": kind, "audio_s": got.audio_ms / 1000, "text": " ".join(u["text"] for u in held)[:40],
                                "probs": frame_probs(model, got.pcm)})
            print(call["id"], leg, len(commits), flush=True)
    kinds = Counter(x["kind"] for x in commits)
    print("\ncommits reaching the ASR check:", dict(kinds))
    out = {"commits": dict(kinds), "grid": []}
    nonspeech = [x for x in commits if x["kind"] != "speech"]
    speech = [x for x in commits if x["kind"] == "speech"]
    print("\nprob threshold | min speech ms | non-speech skipped | speech commits skipped (of %d) | skipped, audio under 1 s | examples" % len(speech))
    for p in PROBS:
        for m in MINS:
            if m == 0:
                continue
            ms = lambda x: 32 * int(np.sum(np.array(x["probs"]) >= p))  # noqa: E731
            ns = sum(ms(x) < m for x in nonspeech)
            sp = [x for x in speech if ms(x) < m]
            row = {"p": p, "min_ms": m, "nonspeech_skipped": ns, "nonspeech": len(nonspeech), "speech_skipped": len(sp),
                   "short": sum(x["audio_s"] < 1.0 for x in sp)}
            out["grid"].append(row)
            print(f"{p:4} | {m:4} | {ns:2}/{len(nonspeech)} | {len(sp):3} ({100 * len(sp) / len(speech):.1f} %) | {row['short']:3} | {[x['text'][:8] for x in sp[:4]]}")
    print("\nSilero speech ms at p=0.5 of the non-speech commits:", sorted(32 * int(np.sum(np.array(x["probs"]) >= 0.5)) for x in nonspeech))
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()

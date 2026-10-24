"""The transcription session's rolling input buffer on the corpus, offline (no server): AICC's
pump decides the commits (tap.py's rule, on the tap audio), the real ``InputBuffer`` with a
``Rolling`` of (idle_keep_ms, voiced_rms) decides what a commit holds, Silero (the gateway's
model, 16 kHz, 32 ms frames, threshold 0.5) measures the speech in it.

    uv run --locked --project spikes/transcription python spikes/transcription/bufsim.py [--out FILE.json]

Per (voiced_rms, idle_keep_ms) setting and per kind of committed segment (speech: overlaps a reference
utterance; music, dtmf, noise: it overlaps none), it reports the commits that would reach the ASR
(``voiced``), the committed audio length, the audio ahead of the speech (noise the ASR is
fed before the utterance), and the speech lost: utterance starts that the committed audio does not
contain. ``silero_ms`` is the audio with Silero probability >= 0.5.
"""

from __future__ import annotations

import argparse
import base64  # noqa: F401
import json
from collections import defaultdict

import numpy as np

import common as c
from mlx_realtime.audio.formats import PCM
from mlx_realtime.session.input_buffer import InputBuffer, Rolling
from mlx_realtime.vad.onnx import OnnxModel
from mlx_realtime.vad.silero import SileroStream

SETTINGS = [(50, 1000), (100, 1000), (150, 1000), (200, 1000), (300, 1000), (400, 1000), (100, 500), (100, 2000)]
MAX_MS = 60000


def pump_commits(pcm: np.ndarray) -> list[tuple[float, float, int]]:
    """(speech start s, commit s, commit frame index) as AICC's pump (tap.py) sends them."""
    out, speaking, quiet, speech_t = [], False, 0, 0.0
    for i in range(len(pcm) // c.FRAME):
        frame = pcm[i * c.FRAME:(i + 1) * c.FRAME].astype(np.float64)
        if float(np.mean(frame**2)) >= 500**2:
            if not speaking:
                speech_t = i * 0.02
            speaking, quiet = True, 0
        elif speaking:
            quiet += 1
        if speaking and i % 5 == 4 and quiet >= 30:
            speaking = False
            out.append((speech_t, (i + 1) * 0.02, i + 1))
    return out


def silero_ms(model: OnnxModel, pcm16k: bytes) -> int:
    stream = SileroStream(model, reset_interval_s=None)
    n = len(pcm16k) // 2 // 512
    hits = sum(stream.probability(pcm16k[k * 1024:(k + 1) * 1024], in_speech=False) >= 0.5 for k in range(n))
    return hits * 32


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model = OnnxModel(c.REPO / "models" / "silero_vad.onnx")
    rows = []
    for call in c.load_calls():
        meta = json.loads((c.DATA / "legs" / f"{call['id']}.json").read_text())
        for leg in ("agent", "customer"):
            pcm = c.read_wav(c.DATA / "legs" / f"{call['id']}_{leg}.wav")
            commits = pump_commits(pcm)
            utts = [u for u in meta["utts"] if u["leg"] == leg]
            bufs = {s: InputBuffer(MAX_MS, input_format=PCM, rolling=Rolling(s[1], s[0])) for s in SETTINGS}
            pos = 0
            for speech_t, commit_t, end_frame in commits:
                span = (speech_t - 1.0, commit_t)
                if any(overlaps(*span, u["start"], u["end"]) for u in utts):
                    kind = "speech"
                elif any(overlaps(*span, h["start"], h["end"]) and h["audio"] == "music" for h in meta["holds"]):
                    kind = "music"
                elif any(overlaps(*span, d["start"], d["end"]) for d in meta["dtmf"]):
                    kind = "dtmf"
                else:
                    kind = "noise"
                first = [u for u in utts if overlaps(*span, u["start"], u["end"])]
                chunk = pcm[pos * c.FRAME // 1: end_frame * c.FRAME].astype("<i2").tobytes() if False else None
                for s, buf in bufs.items():
                    for k in range(pos, end_frame):
                        buf.append(pcm[k * c.FRAME:(k + 1) * c.FRAME].astype("<i2").tobytes())
                    got = buf.commit(empty_ok=True)
                    start_s = got.start_ms / 1000
                    row = {"call": call["id"], "leg": leg, "setting": list(s), "kind": kind, "voiced": got.voiced,
                           "audio_s": round(got.audio_ms / 1000, 2), "lead_s": round(max(0.0, (first[0]["start"] if first else speech_t) - start_s), 2),
                           "lost_start": bool(first) and got.voiced and start_s > first[0]["start"] + 0.05 and first[0]["start"] > span[0] + 1.0}
                    if s == (100, 1000) and got.voiced and got.pcm:
                        row["silero_ms"] = silero_ms(model, got.pcm)
                    rows.append(row)
                pos = end_frame
            print(call["id"], leg, len(commits), "commits", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))
    report(rows)


def report(rows: list[dict]) -> None:
    print("\nsetting (voiced_rms, idle_keep_ms) x kind: commits, to ASR (voiced), mean/max audio s, mean/max lead (noise before the utterance) s")
    agg: dict = defaultdict(list)
    for r in rows:
        agg[(tuple(r["setting"]), r["kind"])].append(r)
    for (setting, kind), rs in sorted(agg.items()):
        v = [r for r in rs if r["voiced"]]
        print(f"{setting} {kind:7} commits={len(rs):4} voiced={len(v):4} audio mean/max={np.mean([r['audio_s'] for r in v]) if v else 0:5.1f}/{max([r['audio_s'] for r in v], default=0):5.1f}"
              f" lead mean/max={np.mean([r['lead_s'] for r in v]) if v else 0:5.1f}/{max([r['lead_s'] for r in v], default=0):5.1f} lost_start={sum(r['lost_start'] for r in rs)}")
    print("\nSilero speech ms at (100, 1000), by kind (min / p10 / median / p90 / max)")
    for kind in ("speech", "music", "dtmf", "noise"):
        v = [r["silero_ms"] for r in rows if r.get("silero_ms") is not None and r["kind"] == kind]
        if v:
            print(kind, len(v), [int(np.percentile(v, q)) for q in (0, 10, 50, 90, 100)])


if __name__ == "__main__":
    main()

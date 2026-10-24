"""The ASR backend alone (no gateway, no timeouts): how long does a non-speech clip take, with and without
the context, and does the request stop? Hold music as the pump commits it (up to the buffer maximum, 60 s),
a DTMF burst, and for scale a 5 s utterance and a 45 s utterance.

    uv run --locked --project spikes/transcription python spikes/transcription/asr_direct.py OUT.json [--url http://127.0.0.1:8102]
"""

from __future__ import annotations

import argparse
import io
import json
import time
import wave

import httpx2
import numpy as np

import common as c
import harness
from mlx_realtime.audio.resample import Resampler

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"


def wav16k(pcm24: np.ndarray) -> bytes:
    r = Resampler(24000, 16000)
    pcm = r.process(pcm24.astype("<i2").tobytes()) + r.flush()
    b = io.BytesIO()
    with wave.open(b, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(pcm)
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8102")
    a = ap.parse_args()
    metas = {x["id"]: json.loads((c.DATA / "legs" / f"{x['id']}.json").read_text()) for x in c.load_calls()}
    cases = []
    for call in ("call06", "call09", "call12"):
        h = next(h for h in metas[call]["holds"] if h["audio"] == "music")
        cases.append(("music", call, "customer", h["start"], min(h["end"], h["start"] + 60) + 0.6))
    cases.append(("dtmf", "call05", "customer", metas["call05"]["dtmf"][0]["start"], metas["call05"]["dtmf"][0]["end"] + 0.6))
    for kind, minimum in (("speech5", 4.5), ("speech45", 40)):
        call, u = next((k, u) for k, m in metas.items() for u in m["utts"] if u["end"] - u["start"] >= minimum)
        cases.append((kind, call, u["leg"], u["start"] - 0.5, u["end"] + 0.7))
    rows = []
    with httpx2.Client(base_url=a.url, timeout=None, trust_env=False) as http:
        for prompt_name in ("none", "terms"):
            for kind, call, leg, t0, t1 in cases:
                wav = wav16k(harness.leg_audio(call, leg)[int(t0 * c.RATE): int(t1 * c.RATE)])
                form = {"model": MODEL, "response_format": "verbose_json", "language": "Chinese"}
                if prompt_name == "terms":
                    form["context"] = harness.PROMPTS["terms"]
                began = time.perf_counter()
                r = http.post("/v1/audio/transcriptions", data=form, files={"file": ("a.wav", wav, "audio/wav")})
                ms = round(1000 * (time.perf_counter() - began))
                text = r.json().get("text", "") if r.status_code == 200 else r.text[:100]
                rows.append({"kind": kind, "call": call, "prompt": prompt_name, "audio_s": round(t1 - t0, 1), "ms": ms,
                             "status": r.status_code, "chars": len(text), "text": text[:70]})
                print(rows[-1], flush=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()

"""G.711 input quality, isolated: the same 8 kHz streams decoded and upsampled to 16 kHz with the
gateway's own InputDecoder at LQ and at HQ, each sent to the ASR (language auto, as the gateway
sends a session without a transcription language), scored with the bench's rules; paired
bootstrap CI on the difference (HQ - LQ).

    uv run --locked --project spikes/realstack python spikes/realstack/g711_quality.py OUT.json

``own``: each row's own law (84 μ-law, 42 A-law streams, untouched); ``ulaw``: every row as the
bench's --codec pcmu sends it (A-law rows transcoded to μ-law through 24 kHz PCM).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import sys
import wave
from pathlib import Path

import httpx2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # bench/ is not part of the mlx_realtime package

from bench.dataset import g711_encode, load  # noqa: E402
from bench.scoring import KIND_FOR_SET, error, pooled  # noqa: E402
from mlx_realtime.audio.codec import InputDecoder  # noqa: E402
from mlx_realtime.audio.formats import PCMA, PCMU  # noqa: E402
from mlx_realtime.audio.resample import Resampler  # noqa: E402

ASR = "http://127.0.0.1:8102/v1/audio/transcriptions"
MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"


def wav16k(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(pcm)
    return buffer.getvalue()


def upsample(data: bytes, law: str, quality: str) -> bytes:
    decoder = InputDecoder(PCMU if law == "ulaw" else PCMA, quality)  # type: ignore[arg-type]
    out = b""
    for i in range(0, len(data), 160):  # 20 ms appends, as on the wire
        out += decoder.decode(data[i:i + 160])
    return out + decoder.flush()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--law", choices=("own", "ulaw"), default="ulaw")
    args = parser.parse_args()
    rows = load(REPO / "spikes" / "data" / "manifest.jsonl", variant="tel8k")
    results = []
    async with httpx2.AsyncClient(trust_env=False, timeout=60) as client:
        for row in rows:
            assert row.g711 is not None
            law, data = row.g711
            if args.law == "ulaw" and law != "ulaw":
                down = Resampler(24000, 8000)
                narrow = down.process(row.pcm) + down.flush()
                data, law = g711_encode(np.frombuffer(narrow, dtype="<i2"), "ulaw"), "ulaw"
            entry = {"id": row.id, "set": row.set, "has_number": row.has_number, "ref": row.text_norm}
            for quality in ("LQ", "HQ"):
                pcm = upsample(data, law, quality)
                response = await client.post(ASR, data={"model": MODEL, "response_format": "verbose_json"},
                                             files={"file": ("a.wav", wav16k(pcm), "audio/wav")})
                response.raise_for_status()
                entry[quality] = str(response.json().get("text", ""))
            results.append(entry)
    summary = {}
    rnd = random.Random(7)
    for name in ("zh", "en", "mix", "all"):
        chosen = [r for r in results if name == "all" or r["set"] == name]
        kind = "mer" if name == "all" else KIND_FOR_SET[name]
        per = {q: [error(r["ref"], r[q], kind) for r in chosen] for q in ("LQ", "HQ")}
        rate = {q: pooled(per[q]).rate for q in ("LQ", "HQ")}
        diffs = []
        for _ in range(2000):
            idx = [rnd.randrange(len(chosen)) for _ in chosen]
            lq = pooled(per["LQ"][i] for i in idx).rate or 0.0
            hq = pooled(per["HQ"][i] for i in idx).rate or 0.0
            diffs.append(hq - lq)
        diffs.sort()
        summary[name] = {"kind": kind, "n": len(chosen), "LQ": rate["LQ"], "HQ": rate["HQ"],
                         "diff": (rate["HQ"] or 0) - (rate["LQ"] or 0),
                         "ci95": [diffs[50], diffs[1949]],
                         "rows_differ": sum(r["LQ"] != r["HQ"] for r in chosen)}
    args.out.write_text(json.dumps({"law": args.law, "summary": summary, "rows": results}, ensure_ascii=False, indent=1))
    for name, value in summary.items():
        print(name, json.dumps(value))


asyncio.run(main())

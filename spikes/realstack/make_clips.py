"""Synthesize the caller clips for aiccflow.py with the running TTS server (uncle_fu, a male voice
the bot does not use), 24 kHz WAV, 300 ms of silence on either side.

    uv run --locked --project spikes/realstack python spikes/realstack/make_clips.py [DIR]
"""

from __future__ import annotations

import json
import sys
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "spikes" / "data" / "realstack" / "clips"
OUT.mkdir(parents=True, exist_ok=True)
CLIPS = {
    "question": "你好，我的手机充电特别慢，这是怎么回事？",
    "rma": "我的维修单号是 R M A 幺零零幺，帮我查一下进度。",
    "transfer": "你说的我都试过了没用，给我转人工客服，把我的情况跟他们说清楚。",
}
for name, text in CLIPS.items():
    body = {"model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "input": text,
            "voice": "uncle_fu", "lang_code": "chinese", "response_format": "pcm", "stream": False}
    request = urllib.request.Request("http://127.0.0.1:8103/v1/audio/speech",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        pcm = answer.read()
    pad = bytes(24000 * 2 * 3 // 10)
    with wave.open(str(OUT / f"{name}.wav"), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(pad + pcm + pad)
    print(name, len(pcm) / 48000, "s")

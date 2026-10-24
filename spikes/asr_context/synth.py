"""Synthesize the corpus with Qwen3-TTS CustomVoice (8-bit 0.6B) over HTTP, 24 kHz mono PCM16.

    uv run --locked --project spikes/asr_context python spikes/asr_context/synth.py \
        [--url http://127.0.0.1:8131] [--only u001,u002]

The server is the backends/audio mlx_audio.server on a free port (not the gateway's 8103). Each
row's ``voice`` (vivian, serena, uncle_fu, aiden) is the CustomVoice speaker, lang_code
``chinese``, non-streaming. Output: data/tts24k/<id>.wav (git-ignored), plus data/bot/bot<N>.wav,
the long bot answers used as the far end of the echo clips (vivian, as the gateway's default).
A clip is regenerated only when missing; a runaway (duration far above the text's) is retried.
"""

from __future__ import annotations

import argparse
import time

import httpx2
import numpy as np

import common as c

MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
BOT_LINES = [
    "您好，这里是客服中心。您的宽带报修单已经受理，工程师会在明天上午九点到十一点之间上门，请保持电话畅通。",
    "好的，我帮您查询了一下，您当前的套餐每月包含两百分钟通话和三十吉流量，超出部分按标准资费计算，还有什么可以帮您的吗？",
    "请您先把路由器的电源关闭，等待三十秒后再重新打开，指示灯变成绿色以后，再试一下能不能正常上网。",
    "您的订单已经发货，预计后天送达。如果您需要修改收货地址，可以在订单页面操作，也可以直接告诉我，我来帮您处理。",
    "非常抱歉给您带来不便。我已经把这个问题记录下来，并且转交给了技术部门，他们会在二十四小时之内给您回电话。",
    "目前我们有几种不同的套餐可以选择，月费从几十元到几百元不等，我可以根据您平时的使用习惯，给您推荐最合适的一种。",
]


def tts(client: httpx2.Client, text: str, voice: str) -> np.ndarray:
    body = {"model": MODEL, "input": text, "voice": voice, "lang_code": "chinese",
            "response_format": "pcm", "stream": False}
    r = client.post("/v1/audio/speech", json=body)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype="<i2").copy()


def synth_one(client: httpx2.Client, text: str, voice: str, path, tries: int = 4) -> float:
    for k in range(tries):
        pcm = tts(client, text, voice)
        dur = len(pcm) / 24000
        # ~0.25 s per CJK character, ~0.12 s per other character, plus the model's 1 s of padding
        cap = 1.5 + sum(0.4 if ord(ch) > 0x2E80 else 0.3 for ch in text if not ch.isspace())
        if 0.5 < dur <= cap:
            c.write_wav(path, pcm, 24000)
            return dur
        print(f"  retry {path.name}: {dur:.1f} s for {len(text)} chars (cap {cap:.1f})", flush=True)
    raise RuntimeError(f"{path.name}: no sane clip in {tries} tries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8131")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    rows = [r for r in c.load_manifest() if r["set"] == "tts" and (not only or r["id"] in only)]
    t0 = time.perf_counter()
    with httpx2.Client(base_url=a.url, timeout=300, trust_env=False) as client:
        for i, line in enumerate(BOT_LINES):
            p = c.DATA / "bot" / f"bot{i}.wav"
            if not p.exists():
                print("bot", i, f"{synth_one(client, line, 'vivian', p):.1f} s", flush=True)
        for n, r in enumerate(rows):
            p = c.DATA / "tts24k" / f"{r['id']}.wav"
            if p.exists():
                continue
            dur = synth_one(client, r.get("tts", r["text"]), r["voice"], p)
            if n % 20 == 0:
                print(f"{n}/{len(rows)} {r['id']} {r['voice']} {dur:.1f} s  ({time.perf_counter() - t0:.0f} s)", flush=True)
    print("done", len(rows), f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

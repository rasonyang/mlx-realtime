"""Does the ASR backend keep generating when the client gives up? A DTMF burst without a prompt and without
a token cap (the backend's 8192-token runaway) is sent with a 2 s client timeout; right after, a 1 s speech clip
is sent: if the backend stopped with the first request it answers in ~0.2 s, else it waits for the runaway.

    uv run --locked --project spikes/transcription python spikes/transcription/giveup_probe.py
"""

from __future__ import annotations

import json
import time

import httpx2

import asr_direct
import common as c
import harness

d = json.loads((c.DATA / "legs" / "call05.json").read_text())["dtmf"][0]
dtmf = asr_direct.wav16k(harness.leg_audio("call05", "customer")[int(d["start"] * c.RATE): int((d["end"] + 0.6) * c.RATE)])
m = json.loads((c.DATA / "legs" / "call01.json").read_text())
u = next(u for u in m["utts"] if u["leg"] == "agent" and 3 <= u["end"] - u["start"] <= 5)
speech = asr_direct.wav16k(harness.leg_audio("call01", "agent")[int((u["start"] - 0.3) * c.RATE): int((u["end"] + 0.3) * c.RATE)])
form = {"model": asr_direct.MODEL, "response_format": "verbose_json", "language": "Chinese"}
with httpx2.Client(base_url="http://127.0.0.1:8102", trust_env=False) as http:
    t0 = time.perf_counter()
    try:
        http.post("/v1/audio/transcriptions", data=form, files={"file": ("a.wav", dtmf, "audio/wav")}, timeout=2)
    except httpx2.HTTPError as exc:
        print(f"client gave up after {time.perf_counter() - t0:.1f} s: {type(exc).__name__}")
    t1 = time.perf_counter()
    r = http.post("/v1/audio/transcriptions", data=form, files={"file": ("a.wav", speech, "audio/wav")}, timeout=None)
    print(f"the next request (a {u['end'] - u['start']:.1f} s clip) took {time.perf_counter() - t1:.1f} s: {r.json()['text'][:30]}")

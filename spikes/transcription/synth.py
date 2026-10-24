"""Synthesize and render the corpus: Qwen3-TTS per utterance, the two legs of every call, the
switch tap path (24 kHz -> 8 kHz G.711 mu-law -> 24 kHz PCM16).

    uv run --locked --project spikes/transcription python spikes/transcription/synth.py tts \
        [--url http://127.0.0.1:8131] [--calls call01,call02]
    uv run --locked --project spikes/transcription python spikes/transcription/synth.py render [--calls ...]

``tts``: every utterance of calls.jsonl at data/tts/<hash>.wav (24 kHz mono PCM16, by voice and
text, so repeated lines are spoken once). A long speech is synthesised sentence by sentence and
joined with 250 ms of silence (one continuous 30-60 s utterance for the ASR). The server is the
backends/audio mlx_audio.server on a free port (not the gateway's 8103), CustomVoice 8-bit.

``render``: the timeline of each call (gaps, crosstalk, holds, DTMF) into the agent and customer
legs at 24 kHz, plus line noise (a band-limited 100-3400 Hz noise at the call's ``noise_rms`` and a
50 Hz hum), then each leg through the tap: Resampler 24k -> 8k, G.711 mu-law encode and decode,
Resampler 8k -> 24k. Output data/legs/<call>_<leg>.wav and data/legs/<call>.json (the utterance
times on the leg timeline, holds, DTMF), also data/legs/<call>_<leg>_clean.wav (before the tap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

import httpx2
import numpy as np

import common as c
from mlx_realtime.audio import g711
from mlx_realtime.audio.resample import Resampler

MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
SPEECH_RMS = {"agent": 2600, "customer": 2000}  # active speech level on the line, 16-bit RMS
MUSIC_RMS = 2200
DTMF_RMS = 2400
SENTENCE_GAP_S = 0.25
DTMF = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336),
        "6": (770, 1477), "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "*": (941, 1209),
        "0": (941, 1336), "#": (941, 1477)}


def key(voice: str, text: str) -> str:
    return hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:16]


def tts_path(voice: str, text: str):
    return c.DATA / "tts" / f"{key(voice, text)}.wav"


def tts(client: httpx2.Client, text: str, voice: str) -> np.ndarray:
    body = {"model": MODEL, "input": text, "voice": voice, "lang_code": "chinese",
            "response_format": "pcm", "stream": False}
    r = client.post("/v1/audio/speech", json=body)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype="<i2").copy()


def synth_one(client: httpx2.Client, text: str, voice: str, tries: int = 4) -> np.ndarray:
    for _ in range(tries):
        pcm = tts(client, text, voice)
        dur = len(pcm) / c.RATE
        spelled = len(re.findall(r"(?<=\s)\S(?=\s|[，。？！]|$)", text)) >= 6  # codes spoken character by character
        cap = 2.5 + sum((0.8 if spelled else 0.4) if ord(ch) > 0x2E80 else 0.3 for ch in text if not ch.isspace())
        if 0.4 < dur <= cap:
            return pcm
        print(f"  retry: {dur:.1f} s for {len(text)} chars (cap {cap:.1f}) {text[:20]}", flush=True)
    raise RuntimeError(f"no sane clip in {tries} tries: {text[:30]}")


def utterance_audio(client: httpx2.Client, ev: dict) -> np.ndarray:
    text = ev.get("tts", ev["text"])
    path = tts_path(ev["voice"], text)
    if path.exists():
        return c.read_wav(path)
    if ev.get("long"):
        parts = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
        gap = np.zeros(int(SENTENCE_GAP_S * c.RATE), dtype=np.int16)
        pcm = np.concatenate([np.concatenate([synth_one(client, s, ev["voice"]), gap]) for s in parts][:])[: -len(gap)]
    else:
        pcm = synth_one(client, text, ev["voice"])
    c.write_wav(path, pcm)
    return pcm


def cmd_tts(args: argparse.Namespace) -> None:
    only = set(filter(None, args.calls.split(",")))
    calls = [x for x in c.load_calls() if not only or x["id"] in only]
    todo = [e for x in calls for e in x["events"] if e["kind"] == "utt" and not tts_path(e["voice"], e.get("tts", e["text"])).exists()]
    unique = {key(e["voice"], e.get("tts", e["text"])): e for e in todo}
    print(f"{len(unique)} clips to synthesize ({sum(len(e.get('tts', e['text'])) for e in unique.values())} characters)", flush=True)
    t0 = time.perf_counter()
    audio_s = 0.0
    with httpx2.Client(base_url=args.url, timeout=600, trust_env=False) as client:
        for n, ev in enumerate(unique.values()):
            pcm = utterance_audio(client, ev)
            audio_s += len(pcm) / c.RATE
            if n % 25 == 0:
                took = time.perf_counter() - t0
                print(f"{n}/{len(unique)} {audio_s / 60:.1f} min of audio in {took:.0f} s (RTF {took / max(audio_s, 1):.2f})", flush=True)
    print("done", f"{time.perf_counter() - t0:.0f} s")


# ----------------------------------------------------------------------------- rendering


def scaled(pcm: np.ndarray, rms: float) -> np.ndarray:
    x = pcm.astype(np.float64)
    active = np.sqrt(np.mean(x[np.abs(x) > 0.05 * np.abs(x).max()] ** 2)) if x.size else 1.0
    return x * (rms / max(active, 1.0))


def line_noise(rng: np.random.Generator, n: int, rms: float) -> np.ndarray:
    """White noise band-limited to 100-3400 Hz at ``rms``, plus a faint 50 Hz hum (24 kHz)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / c.RATE)
    spec[(freqs < 100) | (freqs > 3400)] = 0
    noise = np.fft.irfft(spec, n)
    noise *= 0.95 * rms / max(np.sqrt(np.mean(noise**2)), 1e-9)
    hum = 0.3 * rms * np.sin(2 * np.pi * 50 * np.arange(n) / c.RATE)
    return noise + hum


def music(seconds: float) -> np.ndarray:
    """A looping tone melody (C major pentatonic, 0.45 s notes, a soft 2nd harmonic)."""
    notes = [523.25, 587.33, 659.25, 783.99, 880.0, 783.99, 659.25, 587.33, 523.25, 659.25, 783.99, 659.25]
    n = int(seconds * c.RATE)
    t = np.arange(int(0.45 * c.RATE)) / c.RATE
    env = np.minimum(1, np.minimum(t / 0.02, (t[-1] - t) / 0.08))
    cycle = np.concatenate([env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) for f in notes])
    out = np.tile(cycle, int(np.ceil(n / len(cycle))))[:n]
    return out * (MUSIC_RMS / np.sqrt(np.mean(out**2)))


def dtmf(digits: str) -> np.ndarray:
    t = np.arange(int(0.12 * c.RATE)) / c.RATE
    gap = np.zeros(int(0.12 * c.RATE))
    parts = []
    for d in digits:
        lo, hi = DTMF[d]
        tone = np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t)
        tone *= DTMF_RMS / np.sqrt(np.mean(tone**2))
        parts += [tone, gap]
    return np.concatenate(parts)


def tap(leg: np.ndarray) -> np.ndarray:
    """The switch tap: 24 kHz -> 8 kHz -> G.711 mu-law -> 8 kHz -> 24 kHz PCM16."""
    pcm = np.clip(np.round(leg), -32768, 32767).astype("<i2").tobytes()
    down = Resampler(24000, 8000)
    narrow = down.process(pcm) + down.flush()
    coded = g711.encode(narrow, "ulaw")
    up = Resampler(8000, 24000)
    wide = up.process(g711.decode(coded, "ulaw")) + up.flush()
    return np.frombuffer(wide, dtype="<i2").copy()


def render_call(call: dict, client: httpx2.Client | None) -> dict:
    rng = np.random.default_rng(abs(hash(call["id"])) % 2**32 if False else int(call["id"][4:]) * 7919)
    legs: dict[str, list[tuple[int, np.ndarray]]] = {"agent": [], "customer": []}
    end_of = {"agent": 0.0, "customer": 0.0}
    prev_end = 0.0
    meta: dict = {"id": call["id"], "utts": [], "holds": [], "dtmf": []}
    for i, ev in enumerate(call["events"]):
        if ev["kind"] == "utt":
            leg = ev["leg"]
            pcm = scaled(utterance_audio(client, ev), SPEECH_RMS[leg])
            start = max(prev_end + ev["gap_s"], end_of[leg] + 0.35, 0.3)
            legs[leg].append((int(start * c.RATE), pcm))
            end = start + len(pcm) / c.RATE
            end_of[leg] = end
            prev_end = max(prev_end, end) if ev["gap_s"] < 0 else end
            meta["utts"].append({"i": i, "leg": leg, "text": ev["text"], "start": round(start, 3),
                                 "end": round(end, 3), "long": bool(ev.get("long"))})
        elif ev["kind"] == "hold":
            start = prev_end + 0.5
            end = start + ev["seconds"]
            if ev["audio"] == "music":
                legs["customer"].append((int(start * c.RATE), music(ev["seconds"])))
            meta["holds"].append({"start": round(start, 3), "end": round(end, 3), "audio": ev["audio"]})
            prev_end = end
            end_of["customer"] = max(end_of["customer"], end)
            end_of["agent"] = max(end_of["agent"], end)
        elif ev["kind"] == "dtmf":
            start = prev_end + ev["gap_s"]
            tones = dtmf(ev["digits"])
            legs["customer"].append((int(start * c.RATE), tones))
            end = start + len(tones) / c.RATE
            meta["dtmf"].append({"start": round(start, 3), "end": round(end, 3), "digits": ev["digits"]})
            end_of["customer"] = prev_end = end
    total = int((max(end_of.values()) + 3.0) * c.RATE)
    meta["seconds"] = round(total / c.RATE, 1)
    for leg, pieces in legs.items():
        x = line_noise(rng, total, call["noise_rms"][leg])
        for at, pcm in pieces:
            x[at: at + len(pcm)] += pcm[: total - at]
        clean = np.clip(np.round(x), -32768, 32767).astype("<i2")
        c.write_wav(c.DATA / "legs" / f"{call['id']}_{leg}_clean.wav", clean)
        c.write_wav(c.DATA / "legs" / f"{call['id']}_{leg}.wav", tap(x))
    (c.DATA / "legs" / f"{call['id']}.json").write_text(json.dumps(meta, ensure_ascii=False))
    return meta


def cmd_render(args: argparse.Namespace) -> None:
    only = set(filter(None, args.calls.split(",")))
    with httpx2.Client(base_url=args.url, timeout=600, trust_env=False) as client:
        for call in c.load_calls():
            if only and call["id"] not in only:
                continue
            meta = render_call(call, client)
            speech = {leg: sum(u["end"] - u["start"] for u in meta["utts"] if u["leg"] == leg) for leg in ("agent", "customer")}
            print(call["id"], f"{meta['seconds'] / 60:.1f} min", {k: round(v) for k, v in speech.items()},
                  "holds", [round(h["end"] - h["start"]) for h in meta["holds"]], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tts", "render"])
    ap.add_argument("--url", default="http://127.0.0.1:8131")
    ap.add_argument("--calls", default="")
    a = ap.parse_args()
    {"tts": cmd_tts, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()

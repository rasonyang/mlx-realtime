"""One spoken turn through the gateway, with a timeline of what the client saw.

    uv run --locked --project spikes/realstack python spikes/realstack/turnprobe.py [LABEL] \
        [--url ...] [--wait S]

Streams the zh fixture (24 kHz, real time) and then silence until the turn's response is done
or WAIT seconds pass; prints every non-audio event with its time after the speech ended, the
first audio delta, errors and the outcome (answered / failed / hung).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from websockets.asyncio.client import connect

from mlx_realtime.audio.resample import Resampler

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "speech" / "zh_dinosaurs.wav"


def speech_24k() -> bytes:
    """The zh fixture (16 kHz: 但鸟类还是有很多地方跟恐龙相似。) as 24 kHz client audio."""
    with wave.open(str(FIXTURE)) as file:
        pcm = file.readframes(file.getnframes())
    resampler = Resampler(16000, 24000)
    return resampler.process(pcm) + resampler.flush()


def readyz(url: str) -> str:
    base = url.replace("ws://", "http://").split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/readyz", timeout=3) as answer:
            return f"{answer.status} {answer.read().decode()}"
    except urllib.error.HTTPError as error:
        return f"{error.code} {error.read().decode()}"
    except OSError as error:
        return f"unreachable: {error}"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default="turn")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    parser.add_argument("--wait", type=float, default=20.0)
    args = parser.parse_args()
    print(f"[{args.label}] readyz before: {readyz(args.url)}", flush=True)
    speech = speech_24k()
    t0 = time.monotonic()
    outcome = "hung"
    async with connect(args.url, max_size=None, proxy=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "instructions": "你是电话客服。请用一句话简短回答。",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                "turn_detection": {"type": "server_vad", "silence_duration_ms": 500}},
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}}}}}))
        frames = [speech[i:i + 960] for i in range(0, len(speech), 960)]
        speech_end = None
        events: list[tuple[float, str]] = []
        done = asyncio.Event()
        first_audio = None

        async def read() -> None:
            nonlocal first_audio, outcome
            async for raw in ws:
                event = json.loads(raw)
                kind = event["type"]
                now = time.monotonic()
                if kind == "response.output_audio.delta":
                    if first_audio is None:
                        first_audio = now
                        events.append((now, "first audio delta"))
                    continue
                if kind.endswith(".delta"):
                    continue
                detail = ""
                if kind == "error":
                    detail = json.dumps(event["error"], ensure_ascii=False)
                elif kind == "conversation.item.input_audio_transcription.completed":
                    detail = event["transcript"]
                elif kind == "conversation.item.input_audio_transcription.failed":
                    detail = json.dumps(event.get("error"), ensure_ascii=False)
                elif kind == "response.done":
                    response = event["response"]
                    detail = f"{response['status']} {json.dumps(response.get('status_details'), ensure_ascii=False)}"
                    texts = [c.get("transcript") for o in response.get("output", []) for c in o.get("content", [])]
                    detail += f" {texts}"
                    outcome = "answered" if response["status"] == "completed" and first_audio else response["status"]
                    done.set()
                elif kind in ("session.created", "session.updated", "conversation.item.added",
                              "conversation.item.done", "response.output_item.added",
                              "response.output_item.done", "response.content_part.added",
                              "response.content_part.done", "response.output_audio.done",
                              "response.output_audio_transcript.done", "rate_limits.updated"):
                    continue
                events.append((now, f"{kind} {detail}"))

        reader = asyncio.create_task(read())
        start = time.monotonic()
        index = 0
        while True:
            frame = frames[index] if index < len(frames) else bytes(960)
            if index == len(frames):
                speech_end = time.monotonic()
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(frame).decode()}))
            index += 1
            if done.is_set() and index > len(frames) + 25:
                break
            if speech_end is not None and time.monotonic() - speech_end > args.wait:
                break
            delay = start + index * 0.02 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        reader.cancel()
    base = speech_end or t0
    for at, text in events:
        print(f"  {1000 * (at - base):8.0f} ms  {text[:300]}", flush=True)
    print(f"[{args.label}] outcome: {outcome}; readyz after: {readyz(args.url)}", flush=True)


asyncio.run(main())

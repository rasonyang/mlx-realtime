"""AICC's call flow with spoken caller turns, through the real gateway: every AICC window.

    uv run --locked --project spikes/realstack python spikes/realstack/aiccflow.py OUT.jsonl SESSIONS \
        [--clips DIR] [--url ws://127.0.0.1:8000/v1/realtime]

The caller's clips come from make_clips.py (default ``spikes/data/realstack/clips``). AICC's
checkout is ``AICC_REPO`` (default: a sibling ``ai-native-callcenter``).

Per session (AICC's line_session_zh.json: its instructions, announce and 6 tools; server_vad
500 ms, transcription on), the caller's line streams 20 ms frames in real time throughout:

1. the announce, as AICC's verbatim line (response.create with the line instructions);
2. a caller question (spoken reply);
3. "我的维修单号是…" -> repair_status; as soon as its arguments are done the client sends the
   phase update and the function_call_output, and response.create once response.done arrives
   (AICC's SendToolResult and owed turn): R2, the spoken result;
4. "你说的我都试过了没用，给我转人工客服…" -> transfer_to_agent, normally tool-only: R3.

For every response: created -> first audio, the longest delta gap, last delta -> done, and
created -> done; the transcript; the calls. One JSON line per response.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from websockets.asyncio.client import connect

from mlx_realtime.audio import g711
from mlx_realtime.audio.resample import Resampler

REPO = Path(__file__).resolve().parents[2]
AICC = Path(os.environ.get("AICC_REPO", REPO.parent / "ai-native-callcenter"))
FIXTURE = AICC / "internal" / "provider" / "testdata" / "line_session_zh.json"
CLIPS = REPO / "spikes" / "data" / "realstack" / "clips"
LINE_PREFIX = "请一字不差地说出下面这句话，不要添加任何其它内容：\n"
PHASE_MARK = "当前环节："
PHASE_RESULT = (
    "告知结果:维修进度已经查到。按工具返回如实告诉用户维修进度,然后问用户还有没有其它问题。"
    "用户要求转人工时立即调用 transfer_to_agent。"
)
LOOKUP_RESULT = '{"found":1,"status":"维修中","hint":"告诉用户维修进度：维修中，预计三天内完成。"}'


def load_aicc() -> dict[str, Any]:
    data = json.loads(FIXTURE.read_text())
    tools = []
    for tool in data["tools"]:
        spec: dict[str, Any] = {"type": "function", "name": tool["Name"]}
        if tool.get("Description"):
            spec["description"] = tool["Description"]
        if tool.get("Parameters"):
            spec["parameters"] = tool["Parameters"]
        tools.append(spec)
    return {"instructions": data["instructions"], "announce": data["announce"], "tools": tools}


A = load_aicc()


def read_clip(path: Path) -> bytes:
    """A WAV as 24 kHz mono s16le."""
    with wave.open(str(path)) as file:
        rate, pcm = file.getframerate(), file.readframes(file.getnframes())
    if rate == 24000:
        return pcm
    resampler = Resampler(rate, 24000)
    return resampler.process(pcm) + resampler.flush()


class Response:
    def __init__(self, rid: str, created: float, label: str) -> None:
        self.id, self.created, self.label = rid, created, label
        self.first_audio: float | None = None
        self.last_audio: float | None = None
        self.max_gap = 0.0
        self.done: float | None = None
        self.status: str | None = None
        self.text = ""
        self.calls: list[dict[str, Any]] = []
        self.deltas = 0
        self.audio_bytes = 0

    def row(self, session: str, codec: str) -> dict[str, Any]:
        def ms(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else round((b - a) * 1000, 1)

        return {
            "session": session, "codec": codec, "label": self.label, "response_id": self.id,
            "status": self.status, "first_audio_ms": ms(self.created, self.first_audio),
            "max_gap_ms": round(self.max_gap * 1000, 1) if self.deltas > 1 else None,
            "tail_ms": ms(self.last_audio, self.done), "done_ms": ms(self.created, self.done),
            "tool_only": bool(self.calls) and self.first_audio is None,
            "calls": [(c["name"], c["arguments"]) for c in self.calls], "text": self.text[:200],
        }


class Call:
    def __init__(self, ws, codec: str, out: Path) -> None:  # noqa: ANN001
        self.ws, self.codec, self.out = ws, codec, out
        self.session_id = ""
        self.responses: dict[str, Response] = {}
        self.current: Response | None = None
        self.label = "?"
        self.done_event = asyncio.Event()
        self.queue: list[bytes] = []  # caller audio to play, 24 kHz s16le
        self.stop = asyncio.Event()
        self.instructions_next = A["instructions"].split(PHASE_MARK)[0] + PHASE_MARK + "\n" + PHASE_RESULT
        self.transcripts: list[str] = []
        self.errors: list[Any] = []
        self.speaking = False

    async def send(self, event: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event, ensure_ascii=False))

    async def line(self) -> None:
        """The caller's line: queued clips, else silence, 20 ms frames in real time."""
        start, index = time.monotonic(), 0
        silence = bytes(960)
        buffered = b""
        while not self.stop.is_set():
            if not buffered and self.queue:
                buffered = self.queue.pop(0)
            frame, buffered = (buffered[:960], buffered[960:]) if buffered else (silence, b"")
            frame = frame.ljust(960, b"\0")
            if self.codec == "pcmu":
                frame = self._ulaw(frame)
            await self.send({"type": "input_audio_buffer.append",
                             "audio": base64.b64encode(frame).decode()})
            index += 1
            delay = start + index * 0.02 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    _down: Resampler | None = None

    def _ulaw(self, frame: bytes) -> bytes:
        if self._down is None:
            self._down = Resampler(24000, 8000)
        narrow = self._down.process(frame)
        return g711.encode_samples(np.frombuffer(narrow, dtype="<i2"), "ulaw")

    async def read(self) -> None:
        async for raw in self.ws:
            now = time.monotonic()
            event = json.loads(raw)
            kind = event["type"]
            if kind == "session.created":
                self.session_id = event["session"]["id"]
            elif kind == "response.created":
                response = Response(event["response"]["id"], now, self.label)
                self.responses[response.id] = response
                self.current = response
            elif kind == "response.output_audio.delta":
                response = self.responses.get(event["response_id"])
                if response is not None:
                    if response.last_audio is not None:
                        response.max_gap = max(response.max_gap, now - response.last_audio)
                    response.first_audio = response.first_audio or now
                    response.last_audio = now
                    response.deltas += 1
                    response.audio_bytes += len(base64.b64decode(event["delta"]))
            elif kind == "response.output_audio_transcript.delta":
                response = self.responses.get(event["response_id"])
                if response is not None:
                    response.text += event["delta"]
            elif kind == "response.function_call_arguments.done":
                response = self.responses.get(event["response_id"])
                call = {"name": event["name"], "arguments": event["arguments"], "call_id": event["call_id"]}
                if response is not None:
                    response.calls.append(call)
                if call["name"] == "repair_status":
                    # AICC: the phase update and the tool result at once, the response still open.
                    await self.send({"type": "session.update", "session": {
                        "type": "realtime", "instructions": self.instructions_next}})
                    await self.send({"type": "conversation.item.create", "item": {
                        "type": "function_call_output", "call_id": call["call_id"],
                        "output": LOOKUP_RESULT}})
                    self.owed = True
            elif kind == "response.done":
                response = self.responses.get(event["response"]["id"])
                if response is not None:
                    response.done = now
                    response.status = event["response"]["status"]
                    with self.out.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(response.row(self.session_id, self.codec), ensure_ascii=False) + "\n")
                if getattr(self, "owed", False):
                    self.owed = False
                    self.label = "R2_result"
                    await self.send({"type": "response.create"})
                else:
                    self.done_event.set()
            elif kind == "conversation.item.input_audio_transcription.completed":
                self.transcripts.append(event["transcript"])
            elif kind == "input_audio_buffer.speech_started":
                self.speaking = True
            elif kind == "input_audio_buffer.speech_stopped":
                self.speaking = False
            elif kind == "error":
                self.errors.append(event.get("error"))
                print("  error:", event.get("error"), flush=True)

    async def settle(self, before: int, timeout: float = 40.0) -> None:
        """Until a response newer than the first ``before`` is done, none is open, the caller is
        quiet, and every answer would have finished playing in real time, plus 0.5 s."""
        end = time.monotonic() + timeout
        rate = 48000 if self.codec == "pcm" else 8000
        while time.monotonic() < end:
            now = time.monotonic()
            responses = list(self.responses.values())
            open_ = [r for r in responses if r.done is None]
            played = max((r.first_audio + r.audio_bytes / rate for r in responses if r.first_audio), default=0)
            if (len(responses) > before and not open_ and not self.speaking and not self.queue
                    and now > played + 0.5):
                return
            await asyncio.sleep(0.05)
        print("  settle timed out", flush=True)


async def one_call(url: str, codec: str, clips: dict[str, bytes], out: Path, index: int) -> None:
    async with connect(url, max_size=None, proxy=None) as ws:
        call = Call(ws, codec, out)
        fmt = {"type": "audio/pcm", "rate": 24000} if codec == "pcm" else {"type": "audio/pcmu"}
        reader = asyncio.create_task(call.read())
        await call.send({"type": "session.update", "session": {
            "type": "realtime", "instructions": A["instructions"], "tools": A["tools"],
            "audio": {"input": {"format": fmt, "turn_detection": {"type": "server_vad", "silence_duration_ms": 500}},
                      "output": {"format": fmt}}}})
        line = asyncio.create_task(call.line())
        call.label = "announce"
        await call.send({"type": "response.create", "response": {"instructions": LINE_PREFIX + A["announce"]}})
        await call.settle(0)
        for label, clip in (("R1_question", "question"), ("lookup", "rma"), ("R3_transfer", "transfer")):
            call.label = label
            before = len(call.responses)
            call.queue.append(clips[clip])
            await asyncio.sleep(len(clips[clip]) / 48000 + 0.2)
            await call.settle(before)
        print(f"call {index}: {call.session_id} transcripts={call.transcripts} errors={call.errors}", flush=True)
        call.stop.set()
        await line
        await ws.close()
        reader.cancel()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("sessions", type=int)
    parser.add_argument("--codec", default="pcm")
    parser.add_argument("--clips", type=Path, default=CLIPS)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    args = parser.parse_args()
    clips = {name: read_clip(args.clips / f"{name}.wav") for name in ("question", "rma", "transfer")}
    for index in range(args.sessions):
        await one_call(args.url, args.codec, clips, args.out, index)


asyncio.run(main())

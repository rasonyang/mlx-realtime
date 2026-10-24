"""A client that behaves like AICC's transcription path for one leg of a call:
``internal/transcribe/openairt.go`` (the session) and ``internal/streamin/pump.go`` (the audio).

- connects to ``ws://.../v1/realtime?intent=transcription`` with ``Authorization: Bearer <key>``;
- sends openairt.go's exact ``session.update`` (``turn_detection`` null; model, language, prompt);
- streams 24 kHz PCM16 in 20 ms ``input_audio_buffer.append`` events, silence included;
- the pump's endpointing: a frame with mean square >= 500**2 is speech; after speech, 600 ms of
  frames below that ends the utterance and sends ``input_audio_buffer.commit`` (the pump checks
  every 100 ms; here every 5 frames of audio time, so a run faster than real time keeps its
  decisions);
- a WebSocket ping every 15 s (openairt.go's keepalive);
- reads ``...transcription.delta/completed/failed`` (matched by suffix) and ``error`` as
  openairt.go does; ``failed`` and ``error`` end the session from the client's side.

``stop_on_failed=False`` keeps streaming after a failed item (the CER runs, which count the failures).
``speed`` > 1 streams faster than real time (the CER runs); latencies are only meaningful at 1.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

import numpy as np
from websockets.asyncio.client import connect

import common as c

SILENCE_RMS = 500
SILENCE_RUN_S = 0.6
PING_S = 15.0
MODEL = "gpt-live-transcribe"


@dataclass
class Segment:
    n: int
    speech_t: float  # audio time (s) of the first loud frame of the utterance
    commit_t: float  # audio time (s) at which the commit was sent
    commit_wall: float
    item_id: str = ""
    completed_wall: float | None = None
    transcript: str | None = None
    language: str | None = None
    deltas: int = 0
    failed: str | None = None  # the code of a ...transcription.failed for this item


@dataclass
class Result:
    name: str
    segments: list[Segment] = field(default_factory=list)
    ended_by: str = "audio_done"  # audio_done | failed | error | closed
    error: dict | None = None
    events: list[tuple[float, str]] = field(default_factory=list)
    session_id: str = ""
    started_wall: float = 0.0

    def latencies_ms(self) -> list[float]:
        return [1000 * (s.completed_wall - s.commit_wall) for s in self.segments if s.completed_wall is not None]


def session_update(prompt: str | None, language: str | None = "zh") -> dict:
    transcription: dict = {"model": MODEL}
    if language:
        transcription["language"] = language
    if prompt:
        transcription["prompt"] = prompt
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "transcription": transcription, "turn_detection": None}}}}


async def run_leg(url: str, pcm: np.ndarray, *, name: str = "", prompt: str | None = None, language: str | None = "zh",
                  speed: float = 1.0, key: str = "local", settle_s: float = 60.0, t0_audio: float = 0.0, stop_on_failed: bool = True) -> Result:
    """Stream ``pcm`` (24 kHz int16) through one transcription session. ``t0_audio`` offsets the
    audio times reported (for a window cut from a longer leg)."""
    res = Result(name)
    committed: list[str] = []  # item ids in commit order
    by_item: dict[str, Segment] = {}
    pending_ids: list[Segment] = []  # commits sent, committed event not yet seen
    ended = asyncio.Event()

    async with connect(url, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None, proxy=None,
                       ping_interval=None, open_timeout=5) as ws:
        res.started_wall = time.monotonic()

        async def reader() -> None:
            try:
                async for raw in ws:
                    now = time.monotonic()
                    if isinstance(raw, bytes):
                        continue
                    ev = json.loads(raw)
                    kind = ev.get("type", "")
                    res.events.append((now, kind))
                    if kind == "session.created":
                        res.session_id = ev["session"]["id"]
                    elif kind == "input_audio_buffer.committed":
                        if pending_ids:
                            seg = pending_ids.pop(0)
                            seg.item_id = ev["item_id"]
                            by_item[seg.item_id] = seg
                    elif kind.endswith("input_audio_transcription.delta"):
                        seg = by_item.get(ev.get("item_id"))
                        if seg:
                            seg.deltas += 1
                    elif kind.endswith("input_audio_transcription.completed"):
                        seg = by_item.get(ev.get("item_id"))
                        if seg:
                            seg.completed_wall, seg.transcript, seg.language = now, ev.get("transcript", ""), ev.get("language")
                    elif kind.endswith("input_audio_transcription.failed"):
                        seg = by_item.get(ev.get("item_id"))
                        if seg and not stop_on_failed:
                            seg.failed, seg.completed_wall = (ev.get("error") or {}).get("code", "?"), now
                            continue
                        res.ended_by, res.error = "failed", ev  # AICC ends the session here
                        ended.set()
                        return
                    elif kind == "error":
                        res.ended_by, res.error = "error", ev.get("error")
                        ended.set()
                        return
            except Exception as exc:  # noqa: BLE001 - the connection ended
                if res.ended_by == "audio_done":
                    res.ended_by, res.error = "closed", {"exception": repr(exc)}
            ended.set()

        async def pinger() -> None:
            while True:
                await asyncio.sleep(PING_S)
                await ws.ping()

        read_task = asyncio.create_task(reader())
        ping_task = asyncio.create_task(pinger())
        await ws.send(json.dumps(session_update(prompt, language)))

        speaking, quiet_frames, speech_t = False, 0, 0.0
        n_commits = 0
        start = time.monotonic()
        total = len(pcm) // c.FRAME
        quiet_needed = round(SILENCE_RUN_S / 0.02)
        for i in range(total):
            if ended.is_set():
                break
            frame = pcm[i * c.FRAME: (i + 1) * c.FRAME]
            loud = float(np.mean(frame.astype(np.float64) ** 2)) >= SILENCE_RMS**2
            if loud:
                if not speaking:
                    speech_t = i * 0.02
                speaking, quiet_frames = True, 0
            elif speaking:
                quiet_frames += 1
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(frame.astype("<i2").tobytes()).decode()}))
            if speaking and i % 5 == 4 and quiet_frames >= quiet_needed:
                speaking = False
                n_commits += 1
                seg = Segment(n_commits, t0_audio + speech_t, t0_audio + (i + 1) * 0.02, time.monotonic())
                res.segments.append(seg)
                pending_ids.append(seg)
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            delay = start + (i + 1) * 0.02 / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
        # audio done: wait for the outstanding finals
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline and not ended.is_set() and any(s.completed_wall is None for s in res.segments):
            await asyncio.sleep(0.05)
        ping_task.cancel()
        read_task.cancel()
    return res

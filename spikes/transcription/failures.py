"""Failure paths of a transcription session on a running stack, each with an expected result.

    F="uv run --locked --project spikes/transcription python spikes/transcription/failures.py"
    $F main    [--url WS_URL] [--log GATEWAY_LOG] --out spikes/results/transcription_failures.json
    $F silence [--url WS_URL] [--minutes 10] --out ...        # run beside ``main`` (one session)

Cases (``main``): empty_commit, double_commit, disconnect_mid_utterance (the slot is freed and the
log shows no error), fifth_session (4 open: the handshake error), malformed_base64,
response_create. ``silence``: pure digital silence with a ping every 15 s for ``--minutes``, then an
utterance: the session lives and the transcript is right. After each case the session (if it should
live) transcribes a reference utterance. ``--log`` is the gateway's output file: lines added during
the run containing ERROR, Traceback or CRITICAL fail the run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

import common as c
import tap
from harness import DEFAULT_URL, PROMPTS, leg_audio

KEY = "local"


def probe_clip() -> tuple[np.ndarray, str]:
    """A reference utterance of 4-8 s with 0.3 s of silence on each side, and its text."""
    meta = json.loads((c.DATA / "legs" / "call01.json").read_text())
    u = next(u for u in meta["utts"] if u["leg"] == "agent" and 4 <= u["end"] - u["start"] <= 8)
    pcm = leg_audio("call01", "agent")[int((u["start"] - 0.3) * c.RATE): int((u["end"] + 0.3) * c.RATE)]
    return pcm, u["text"]


class Sess:
    def __init__(self, ws) -> None:  # noqa: ANN001
        self.ws, self.events = ws, []

    @classmethod
    async def open(cls, url: str, prompt: str | None = PROMPTS["terms"]) -> "Sess":
        ws = await connect(url, additional_headers={"Authorization": f"Bearer {KEY}"}, max_size=None, proxy=None, ping_interval=None)
        s = cls(ws)
        await s.send(tap.session_update(prompt))
        await s.until(lambda e: e["type"] == "session.updated")
        return s

    async def send(self, ev: dict) -> None:
        await self.ws.send(json.dumps(ev))

    async def audio(self, pcm: np.ndarray, realtime: bool = False) -> None:
        for k in range(len(pcm) // c.FRAME):
            await self.send({"type": "input_audio_buffer.append",
                             "audio": base64.b64encode(pcm[k * c.FRAME:(k + 1) * c.FRAME].astype("<i2").tobytes()).decode()})
            await asyncio.sleep(0.02 if realtime else 0)

    async def until(self, pred, timeout: float = 30.0) -> dict:  # noqa: ANN001
        async with asyncio.timeout(timeout):
            while True:
                ev = json.loads(await self.ws.recv())
                self.events.append(ev)
                if pred(ev):
                    return ev

    async def final(self, timeout: float = 30.0) -> dict:
        return await self.until(lambda e: e["type"].endswith("input_audio_transcription.completed") or e["type"] in ("error",), timeout)

    async def transcribes(self, pcm: np.ndarray, text: str) -> tuple[bool, str]:
        await self.audio(pcm)
        await self.send({"type": "input_audio_buffer.commit"})
        ev = await self.final()
        got = ev.get("transcript", "")
        cer = c.edit_distance(c.norm(text), c.norm(got)) / max(len(c.norm(text)), 1)
        return cer < 0.25, got


def log_errors(log: Path | None, offset: int) -> list[str]:
    if log is None or not log.exists():
        return []
    new = log.read_bytes()[offset:].decode(errors="replace").splitlines()
    return [ln[:200] for ln in new if any(w in ln for w in ("ERROR", "Traceback", "CRITICAL"))]


async def main_cases(a: argparse.Namespace) -> None:
    pcm, text = probe_clip()
    log = Path(a.log) if a.log else None
    offset = log.stat().st_size if log and log.exists() else 0
    results: dict[str, dict] = {}

    async def case(name: str, expected: str, fn) -> None:  # noqa: ANN001
        try:
            ok, detail = await fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"exception {exc!r}"
        results[name] = {"expected": expected, "pass": bool(ok), "detail": detail}
        print(("PASS" if ok else "FAIL"), name, "-", detail, flush=True)

    async def empty_commit():
        s = await Sess.open(a.url)
        await s.send({"type": "input_audio_buffer.commit"})
        ev = await s.final(10)
        errors = [e for e in s.events if e["type"] == "error"]
        alive, got = await s.transcribes(pcm, text)
        await s.ws.close()
        return (not errors and ev.get("transcript") == "" and alive), f"first final {ev['type']} transcript={ev.get('transcript')!r}; then {got!r}"

    async def double_commit():
        s = await Sess.open(a.url)
        await s.audio(pcm)
        await s.send({"type": "input_audio_buffer.commit"})
        await s.send({"type": "input_audio_buffer.commit"})
        finals = [await s.final(), await s.final(10)]  # the empty one may overtake the transcript
        texts = sorted((f.get("transcript") or "" for f in finals), key=len)
        errors = [e for e in s.events if e["type"] == "error"]
        await s.ws.close()
        return (not errors and texts[0] == "" and texts[1] != ""), f"finals={texts!r} errors={errors}"

    async def disconnect():
        s = await Sess.open(a.url)
        await s.audio(pcm[: 3 * c.RATE], realtime=False)
        s.ws.transport.abort()  # no close handshake: the client vanishes
        await asyncio.sleep(1.0)
        # all four slots must be free: open four sessions
        opened = [await Sess.open(a.url) for _ in range(4)]
        for o in opened:
            await o.ws.close()
        return True, "4 sessions opened right after the abort (slot freed)"

    async def fifth():
        opened = [await Sess.open(a.url) for _ in range(4)]
        try:
            ws = await connect(a.url, additional_headers={"Authorization": f"Bearer {KEY}"}, max_size=None, proxy=None, ping_interval=None)
            ev = json.loads(await asyncio.wait_for(ws.recv(), 5))
            try:
                await asyncio.wait_for(ws.recv(), 5)
                closed = None
            except Exception as exc:  # noqa: BLE001
                closed = getattr(getattr(exc, "rcvd", None), "code", repr(exc))
            ok = ev["type"] == "error" and ev["error"]["code"] == "session_limit_reached"
            detail = f"{ev['type']} {ev.get('error')} close={closed}"
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"handshake rejected at HTTP level or failed: {exc!r}"
        for o in opened:
            await o.ws.close()
        return ok, detail

    async def malformed():
        s = await Sess.open(a.url)
        await s.send({"type": "input_audio_buffer.append", "audio": "!!!not base64!!!"})
        err = await s.until(lambda e: e["type"] == "error", 5)
        alive, got = await s.transcribes(pcm, text)
        await s.ws.close()
        return alive, f"error {err['error']['code']}/{err['error'].get('param')}; then {got!r}"

    async def response_create():
        s = await Sess.open(a.url)
        await s.send({"type": "response.create"})
        err = await s.until(lambda e: e["type"] == "error", 5)
        alive, got = await s.transcribes(pcm, text)
        await s.ws.close()
        return (alive and "response.create" in json.dumps(err)), f"error {err['error']['code']}: {err['error']['message'][:80]}; then {got!r}"

    await case("empty_commit", "no error; completed with an empty transcript; the session still transcribes", empty_commit)
    await case("double_commit", "first transcript, second empty, no error", double_commit)
    await case("response_create", "error naming response.create; the session stays open and transcribes", response_create)
    await case("malformed_base64", "an error event; the session stays open and transcribes", malformed)
    await case("fifth_session", "error session_limit_reached at the handshake (4 open)", fifth)
    await case("disconnect_mid_utterance", "the slot is freed (4 sessions open right after)", disconnect)
    await asyncio.sleep(1)
    errors = log_errors(log, offset)
    results["gateway_log"] = {"expected": "no ERROR/Traceback lines during the run", "pass": not errors, "detail": errors[:5] if log else "no --log given"}
    print("PASS" if not errors else "FAIL", "gateway_log -", errors[:5] if log else "no --log given")
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))


async def silence_case(a: argparse.Namespace) -> None:
    pcm, text = probe_clip()
    s = await Sess.open(a.url)
    began = time.monotonic()
    zeros = np.zeros(c.FRAME, dtype=np.int16)
    n = int(a.minutes * 60 / 0.02)
    last_ping = began
    for k in range(n):
        await s.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(zeros.tobytes()).decode()})
        if time.monotonic() - last_ping >= 15:
            await s.ws.ping()
            last_ping = time.monotonic()
        await asyncio.sleep(max(0.0, began + (k + 1) * 0.02 - time.monotonic()))
    ok, got = await s.transcribes(pcm, text)
    errors = [e for e in s.events if e["type"] == "error"]
    await s.ws.close()
    res = {"minutes": a.minutes, "pass": bool(ok and not errors), "transcript": got, "reference": text, "errors": errors}
    print(res)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "silence"])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--log", default="")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    asyncio.run(main_cases(a) if a.cmd == "main" else silence_case(a))


if __name__ == "__main__":
    main()

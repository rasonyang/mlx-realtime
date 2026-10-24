"""Measurements on a running stack (issue #2). Every subcommand takes ``--url`` (default
ws://127.0.0.1:8000/v1/realtime?intent=transcription) and writes JSON to ``--out``.

    H="uv run --locked --project spikes/transcription python spikes/transcription/harness.py"
    $H cer     --out spikes/results/transcription_cer_terms.json --prompt terms [--speed 4] [--calls call01,...]
    $H latency --out spikes/results/transcription_latency_4.json --sessions 4 --calls call04 --seconds 300
    $H sweep   --out spikes/results/transcription_sweep.json [--load 3]
    $H nonspeech --out spikes/results/transcription_nonspeech.json   # hold music and DTMF commits, alone

``cer``: every call's two legs (agent, customer) through the tap path and the pump's endpointing,
two calls at a time (4 sessions, the limit), at ``--speed`` times real time. ``--prompt terms``
sends the term list as openairt.go does (hints joined with ", "), ``none`` sends no prompt.

``latency``: real time, the first ``--seconds`` of each leg of the chosen calls. ``--sessions 1``:
the agent leg of the first call; ``2``: both legs of it; ``4``: both legs plus two realtime bot
sessions (spikes/realstack/aiccflow.py, two processes started together, so the bots speak at the same
moment); ``bots2``: only the two bot sessions (the baseline for the bots' first audio).

``sweep``: ASR time against utterance length: reference utterances of 1 to 60 s, each sent as its own
commit (audio as fast as the server takes it), alone or with ``--load`` other sessions streaming speech.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import common as c
import tap

sys.path.insert(0, str(c.HERE.parent / "asr_context"))
import terms as T  # noqa: E402

DEFAULT_URL = "ws://127.0.0.1:8000/v1/realtime?intent=transcription"
PROMPTS = {"terms": T.CONTEXT_TERMS, "none": None}
AICC_REPO = os.environ.get("AICC_REPO", str(Path.home() / "workspaces/github/ai-native-callcenter"))


def leg_audio(call: str, leg: str, seconds: float | None = None) -> np.ndarray:
    pcm = c.read_wav(c.DATA / "legs" / f"{call}_{leg}.wav")
    return pcm if seconds is None else pcm[: int(seconds * c.RATE)]


def summarize(res: tap.Result) -> dict:
    return {"name": res.name, "ended_by": res.ended_by, "error": res.error, "session_id": res.session_id,
            "segments": [{"n": s.n, "speech_t": round(s.speech_t, 2), "commit_t": round(s.commit_t, 2),
                          "transcript": s.transcript,
                          "latency_ms": None if s.completed_wall is None or s.failed else round(1000 * (s.completed_wall - s.commit_wall), 1),
                          "failed": s.failed} for s in res.segments]}


async def cmd_cer(a: argparse.Namespace) -> None:
    prompt = PROMPTS[a.prompt]
    calls = [x["id"] for x in c.load_calls() if not a.calls or x["id"] in a.calls.split(",")]
    out: dict = {"prompt": a.prompt, "speed": a.speed, "legs": {}}
    # Resumable: every finished leg is appended to OUT.partial.jsonl; a rerun skips the calls whose two legs are in it.
    partial = Path(a.out).with_suffix(".partial.jsonl")
    if partial.exists():
        for line in partial.read_text().splitlines():
            row = json.loads(line)
            out["legs"][row["name"]] = row
    calls = [x for x in calls if not all(f"{x}_{leg}" in out["legs"] for leg in ("agent", "customer"))]
    print(f"{len(calls)} calls to run, {len(out['legs'])} legs resumed", flush=True)
    t0 = time.monotonic()

    async def one(call: str) -> None:
        results = await asyncio.gather(*(tap.run_leg(a.url, leg_audio(call, leg), name=f"{call}_{leg}", prompt=prompt,
                                                     speed=a.speed, stop_on_failed=False) for leg in ("agent", "customer")))
        for leg, res in zip(("agent", "customer"), results):
            out["legs"][f"{call}_{leg}"] = summarize(res)
            with partial.open("a") as f:
                f.write(json.dumps(out["legs"][f"{call}_{leg}"], ensure_ascii=False) + "\n")
            ok = sum(s.transcript is not None for s in res.segments)
            failed = sum(bool(s.failed) for s in res.segments)
            lost = len(res.segments) - ok - failed  # no completed and no failed: must be 0
            print(f"{call}_{leg}: {len(res.segments)} commits, {ok} finals, {failed} failed, {lost} unanswered, ended_by={res.ended_by} "
                  f"({time.monotonic() - t0:.0f} s)", flush=True)

    queue = asyncio.Queue()
    for call in calls:
        queue.put_nowait(call)

    async def worker() -> None:
        while not queue.empty():
            await one(queue.get_nowait())

    await asyncio.gather(*(worker() for _ in range(a.parallel)))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


def run_bots(url: str, rows: Path, n_calls: int, clips: Path) -> list[subprocess.Popen]:
    """Two aiccflow processes (one bot session each) started together."""
    env = dict(os.environ, AICC_REPO=AICC_REPO)
    rows.unlink(missing_ok=True)
    cmd = ["uv", "run", "--locked", "--project", str(c.REPO / "spikes/realstack"), "python",
           str(c.REPO / "spikes/realstack/aiccflow.py"), str(rows), str(n_calls), "--url", url, "--clips", str(clips)]
    return [subprocess.Popen(cmd, env=env, cwd=c.REPO, stdout=open(c.DATA / "logs" / f"bot{k}.log", "w"),
                             stderr=subprocess.STDOUT) for k in range(2)]


async def cmd_latency(a: argparse.Namespace) -> None:
    prompt = PROMPTS[a.prompt]
    calls = a.calls.split(",")
    out: dict = {"sessions": a.sessions, "seconds": a.seconds, "legs": {}, "bot_rows": None}
    bots: list[subprocess.Popen] = []
    base = a.url.split("/v1/realtime")[0] + "/v1/realtime"
    rows = Path(a.out).with_suffix(".bots.jsonl")
    if a.sessions in ("4", "bots2"):
        bots = run_bots(base, rows, a.bot_calls, Path(a.clips))
        await asyncio.sleep(a.bot_lead)  # both bots connected and speaking their announce
    tasks = []
    if a.sessions != "bots2":
        legs = ["agent"] if a.sessions == "1" else ["agent", "customer"]
        for call in calls:
            for leg in legs:
                tasks.append(tap.run_leg(a.url, leg_audio(call, leg, a.seconds), name=f"{call}_{leg}", prompt=prompt))
        # one call at a time per condition: 1 -> 1 session, 2 and 4 -> two legs of a call
        n = len(legs)
        for k in range(0, len(tasks), n):
            for res in await asyncio.gather(*tasks[k: k + n]):
                out["legs"][res.name] = summarize(res)
                lat = res.latencies_ms()
                print(res.name, f"{len(res.segments)} commits", f"p50={statistics.median(lat):.0f}" if lat else "", flush=True)
    for b in bots:
        b.wait(timeout=a.bot_wait)
    out["bot_rows"] = str(rows) if bots else None
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))


async def cmd_sweep(a: argparse.Namespace) -> None:
    """One commit per reference utterance, alone (or beside ``--load`` streaming sessions)."""
    metas = {x["id"]: json.loads((c.DATA / "legs" / f"{x['id']}.json").read_text()) for x in c.load_calls()}
    pool = []
    for call, meta in metas.items():
        for u in meta["utts"]:
            pool.append((u["end"] - u["start"], call, u))
    pool.sort(key=lambda x: x[0])
    picks, seen = [], set()
    for target in [1, 2, 3, 5, 8, 12, 20, 30, 40, 50, 60]:
        best = [p for p in pool if p[0] >= target - 1 and (p[1], p[2]["i"]) not in seen][:a.per_length]
        for p in best:
            seen.add((p[1], p[2]["i"]))
            picks.append((target, *p))
    stop = asyncio.Event()
    loaders = []
    for k in range(a.load):
        call = ["call04", "call09", "call12"][k % 3]
        loaders.append(asyncio.create_task(loop_leg(a.url, call, "agent" if k % 2 == 0 else "customer", stop)))
    await asyncio.sleep(2 if a.load else 0)
    rows = []
    async with asyncio.timeout(3600):
        from websockets.asyncio.client import connect
        import base64
        async with connect(a.url, additional_headers={"Authorization": "Bearer local"}, max_size=None, proxy=None) as ws:
            await ws.send(json.dumps(tap.session_update(PROMPTS[a.prompt])))
            for target, dur, call, u in picks:
                pcm = leg_audio(call, u["leg"])[int((u["start"] - 0.5) * c.RATE): int((u["end"] + 0.7) * c.RATE)]
                for k in range(0, len(pcm) // c.FRAME):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(
                        pcm[k * c.FRAME:(k + 1) * c.FRAME].astype("<i2").tobytes()).decode()}))
                    if k % 50 == 49:
                        await asyncio.sleep(0)
                t_commit = time.monotonic()
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                text, err = None, None
                while True:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
                    if ev["type"].endswith("input_audio_transcription.completed"):
                        text = ev["transcript"]
                        break
                    if ev["type"] in ("error",) or ev["type"].endswith("transcription.failed"):
                        err = ev
                        break
                rows.append({"target_s": target, "audio_s": round(len(pcm) / c.RATE, 2), "latency_ms": round(1000 * (time.monotonic() - t_commit), 1),
                             "call": call, "i": u["i"], "ref_chars": len(c.norm(u["text"])), "hyp_chars": len(c.norm(text or "")),
                             "cer": (c.edit_distance(c.norm(u["text"]), c.norm(text or "")) / max(len(c.norm(u["text"])), 1)), "error": err})
                print(rows[-1], flush=True)
    stop.set()
    await asyncio.gather(*loaders, return_exceptions=True)
    Path(a.out).write_text(json.dumps({"load": a.load, "rows": rows}, ensure_ascii=False))


async def loop_leg(url: str, call: str, leg: str, stop: asyncio.Event) -> None:
    while not stop.is_set():
        t = asyncio.create_task(tap.run_leg(url, leg_audio(call, leg, 120), name="load", prompt=T.CONTEXT_TERMS))
        while not t.done() and not stop.is_set():
            await asyncio.sleep(0.5)
        if stop.is_set():
            t.cancel()
            return
        await t


async def cmd_nonspeech(a: argparse.Namespace) -> None:
    """Hold music (as the pump commits it: from the first loud frame to 600 ms after the last), DTMF and
    a stretch of line noise, each as one commit, with and without the prompt, alone: the ASR's time, its
    answer, and whether the item failed."""
    import base64

    from websockets.asyncio.client import connect
    metas = {x["id"]: json.loads((c.DATA / "legs" / f"{x['id']}.json").read_text()) for x in c.load_calls()}
    cases = []
    for call, meta in metas.items():
        for h in meta["holds"]:
            if h["audio"] == "music":
                cases.append(("music", call, "customer", h["start"], min(h["end"], h["start"] + 60) + 0.6))
        for d in meta["dtmf"]:
            cases.append(("dtmf", call, "customer", d["start"], d["end"] + 0.6))
    cases = [x for x in cases if not a.calls or x[1] in a.calls.split(",")][: a.limit]
    rows = []
    for prompt_name in ("none", "terms"):
        for kind, call, leg, t0, t1 in cases:
            pcm = leg_audio(call, leg)[int(t0 * c.RATE): int(t1 * c.RATE)]
            async with connect(a.url, additional_headers={"Authorization": "Bearer local"}, max_size=None, proxy=None) as ws:
                await ws.send(json.dumps(tap.session_update(PROMPTS[prompt_name])))
                for k in range(len(pcm) // c.FRAME):
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(
                        pcm[k * c.FRAME:(k + 1) * c.FRAME].astype("<i2").tobytes()).decode()}))
                    if k % 100 == 99:
                        await asyncio.sleep(0)
                t_commit = time.monotonic()
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                text, failed = None, None
                while True:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), 120))
                    if ev["type"].endswith("input_audio_transcription.completed"):
                        text = ev["transcript"]
                        break
                    if ev["type"].endswith("transcription.failed") or ev["type"] == "error":
                        failed = ev.get("error", {}).get("code")
                        break
            rows.append({"kind": kind, "call": call, "prompt": prompt_name, "audio_s": round(len(pcm) / c.RATE, 1),
                         "latency_ms": round(1000 * (time.monotonic() - t_commit)), "failed": failed, "text": (text or "")[:80]})
            print(rows[-1], flush=True)
            await asyncio.sleep(a.pause)  # let a wedged ASR drain before the next case
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cer", "latency", "sweep", "nonspeech"):
        p = sub.add_parser(name)
        p.add_argument("--url", default=DEFAULT_URL)
        p.add_argument("--out", required=True)
        p.add_argument("--prompt", choices=list(PROMPTS), default="terms")
        p.add_argument("--calls", default="")
        if name == "cer":
            p.add_argument("--speed", type=float, default=4.0)
            p.add_argument("--parallel", type=int, default=2)
        if name == "latency":
            p.add_argument("--sessions", choices=["1", "2", "4", "bots2"], required=True)
            p.add_argument("--seconds", type=float, default=300)
            p.add_argument("--bot-calls", type=int, default=5)
            p.add_argument("--bot-lead", type=float, default=2.0)
            p.add_argument("--bot-wait", type=float, default=900)
            p.add_argument("--clips", default=str(Path.home() / "workspaces/github/mlx-realtime/spikes/data/realstack/clips"))
        if name == "nonspeech":
            p.add_argument("--limit", type=int, default=8)
            p.add_argument("--pause", type=float, default=15.0)
        if name == "sweep":
            p.add_argument("--load", type=int, default=0)
            p.add_argument("--per-length", type=int, default=3)
    a = ap.parse_args()
    asyncio.run({"cer": cmd_cer, "latency": cmd_latency, "sweep": cmd_sweep, "nonspeech": cmd_nonspeech}[a.cmd](a))


if __name__ == "__main__":
    main()

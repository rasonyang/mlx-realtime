"""LLM hardening (2026-10-23): AICC's latency cases on the real LLM, through the gateway.

Starts the LLM backend (``python -m mlxrt_llm.serve``: mlx_lm.server 0.32.2 with the lenient
tool-call parser, Qwen3.6-35B-A3B-4bit @38740b84, the G2 flags) on 127.0.0.1:8101 and a gateway
on 127.0.0.1:8103 (``config.mock.yaml`` with the real LLM provider; ASR and TTS stay mocks, so
the LLM is what is measured), then replays AICC's cases over ``/v1/realtime`` with AICC's
``line_session_zh.json`` (instructions, announce, 6 tools), one session at a time.

Per response it records the client's times after ``response.created`` (first transcript and
audio delta, each function call, ``response.done``), the gateway's turn marks from
``/debug/turns`` (``llm_first_token``, ``first_sentence_ready`` = the first segment boundary)
and the LLM usage (cached tokens). Writes ``spikes/results/llm_hardening_<name>.json``.

Cases (n each):
  cold             a new call whose instructions the server has never seen: update, then
                   the greeting and the caller's first turn ``--gap`` s later
  warm             the next two caller turns of the same call
  change_turn      B1's S2 case 3: an unseen instructions-only update, ``--gap`` s, then a
                   caller turn (spoken reply, or a hangup call)
  change_transfer  B1's transfer case: an unseen update, ``--gap`` s, then "转人工…"
  warm_transfer    the same "转人工…" turn without an update
  aicc_spoken      AICC's own sequence: a call (repair_status) answered by update +
                   function_call_output as soon as its arguments are done, response.create at
                   response.done (R2, spoken), then the next caller turn (R3)
  aicc_transfer    the same, with "转人工…" as the next caller turn (R3, tool-only)
  override         a per-response instructions override, new each time, then the next turn
  line             a verbatim line ("Say exactly this…"), which skips the LLM

    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm_hardening python \\
        spikes/llm_hardening/run.py --name baseline --warm off [--server-args "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import time
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "spikes" / "results"
AICC_FIXTURE = (
    Path(os.environ.get("AICC_REPO", "~/workspaces/github/ai-native-callcenter")).expanduser()
    / "internal/provider/testdata/line_session_zh.json"
)
MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
REVISION = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots"
    / REVISION
)
HOST, LLM_PORT, GATEWAY_PORT = "127.0.0.1", 8101, 8103
G2_FLAGS = [
    "--chat-template-args", '{"enable_thinking":false}', "--max-tokens", "2048",
    "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
]  # fmt: skip
LOG_DIR = Path(os.environ.get("SPIKE_LOG_DIR", "/tmp")).resolve()
LINE_PREFIX = "Say exactly this, word for word, and add nothing else:\n"

# B1's S2 material (spikes/llm/s2_perf.py, llmlib.py), so the numbers compare.
CALLER = [
    "你好，我想问一下手机怎么截图。", "我的手机充电特别慢，怎么回事？", "手机存储空间满了怎么清理？",
    "怎么把手机系统升级到最新版本？", "我手机老是自动重启，怎么办？", "怎么关闭手机的自动亮度？",
    "手机发烫正常吗？", "怎么设置来电铃声？", "手机连不上无线网怎么办？", "怎么恢复出厂设置？",
    "我的手机屏幕有点发黄，能调吗？", "怎么开启省电模式？", "蓝牙耳机连不上手机怎么办？",
    "怎么备份手机里的照片？", "手机信号不好怎么办？", "怎么查看手机还剩多少存储？",
    "手机开不了机了怎么办？", "怎么关掉应用通知？", "手机突然没声音了怎么回事？", "怎么设置指纹解锁？",
]  # fmt: skip
FOLLOW = [
    "那还有别的办法吗？", "好的，我试试。那如果还不行呢？", "明白了，那需要送修吗？", "这样会不会丢数据？",
    "嗯，那大概多久能好？", "谢谢，那我顺便再问一下怎么截图？",
]  # fmt: skip
TRANSFER_TURN = "你说的我都试过了没用，给我转人工客服，把我的情况跟他们说清楚。"
PHASE_MARK = "当前环节："
PHASE_CLOSING = (
    "结束通话:用户的问题已经解决。简短确认用户没有其它问题;用户表示没有了就道别并调用 hangup。"
    "如果用户还有新问题,按使用类问题或查维修进度的流程处理。"
)
PHASE_RESULT = (
    "告知结果:维修进度已经查到。按工具返回如实告诉用户维修进度,然后问用户还有没有其它问题。"
    "用户要求转人工时立即调用 transfer_to_agent。"
)
OVERRIDES = [
    "请用一句话亲切地问来电者还有没有其它需要帮忙的。",
    "请简短地向来电者道歉让他久等了，然后问他是否还在线。",
    "用一句话提醒来电者可以随时按0转人工，然后继续回答他的问题。",
]
LOOKUP_RESULT = '{"found":1,"status":"维修中","hint":"告诉用户维修进度：维修中，预计三天内完成。"}'


def load_aicc() -> dict[str, Any]:
    data = json.loads(AICC_FIXTURE.read_text())
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


def phase(text: str) -> str:
    """AICC's instructions with the text after 当前环节： replaced (a phase change)."""
    return A["instructions"].split(PHASE_MARK)[0] + PHASE_MARK + "\n" + text


def nonce() -> str:
    return uuid.uuid4().hex[:6]


# ------------------------------------------------------------------ processes


def port_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((HOST, port)) == 0


def http(path: str, port: int, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://{HOST}:{port}{path}", data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (localhost)
        text = response.read().decode()
    return json.loads(text) if text.strip().startswith(("{", "[")) else text


class Proc:
    def __init__(self, name: str, argv: list[str], env: dict[str, str], port: int) -> None:
        self.name, self.argv, self.port = name, argv, port
        self.log = LOG_DIR / f"llm_hardening_{name}_{int(time.time())}.log"
        if port_open(port):
            raise RuntimeError(f"port {port} is in use")
        self.proc = subprocess.Popen(  # noqa: S603
            argv,
            stdout=self.log.open("w"),
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            cwd=REPO,
            start_new_session=True,
        )

    def wait_http(self, path: str, timeout_s: float = 900) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited:\n{self.log.read_text()[-3000:]}")
            try:
                http(path, self.port, timeout=2)
                return
            except OSError:
                time.sleep(0.3)
        raise RuntimeError(f"{self.name}: {path} did not answer")

    def pid_listening(self) -> int | None:
        out = subprocess.run(  # noqa: S603
            ["lsof", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        pids = out.stdout.split()
        return int(pids[0]) if pids else None

    def footprint_gb(self) -> float | None:
        pid = self.pid_listening()
        if pid is None:
            return None
        out = subprocess.run(  # noqa: S603
            ["footprint", "-p", str(pid), "--noCategories", "-f", "bytes"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"Footprint:\s*(\d+)\s*B", out.stdout)
        return round(int(match.group(1)) / 1e9, 2) if match else None

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        for _ in range(50):
            if not port_open(self.port):
                return
            pid = self.pid_listening()
            if pid:
                os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)


def start_backend(extra: list[str]) -> Proc:
    argv = [
        "uv", "run", "--locked", "--project", "backends/llm", "python", "-m", "mlxrt_llm.serve",
        "--model", str(SNAPSHOT), "--host", HOST, "--port", str(LLM_PORT), *G2_FLAGS, *extra,
    ]  # fmt: skip
    backend = Proc("llm", argv, {"HF_HUB_OFFLINE": "1"}, LLM_PORT)
    backend.wait_http("/health")
    # /health answers before the model is loaded: a real request is the readiness check.
    http(
        "/v1/chat/completions",
        LLM_PORT,
        {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 1},
    )
    return backend


def start_gateway(*, warm: bool) -> Proc:
    llm = (
        f"{{type: mlx_lm_chat, base_url: 'http://{HOST}:{LLM_PORT}', "
        f"warm_on_update: {str(warm).lower()}}}"
    )
    env = {
        "MLXRT__GATEWAY__PORT": str(GATEWAY_PORT),
        "MLXRT__PROVIDERS__LLM": llm,
        "MLXRT__PROVIDERS__ASR__DELAY_MS": "0",
        "MLXRT__PROVIDERS__TTS__FIRST_CHUNK_DELAY_MS": "0",
        "MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS": "true",
        "MLXRT__OBSERVABILITY__LOG_LEVEL": "info",
    }
    argv = ["uv", "run", "--locked", "python", "-m", "mlx_realtime", "--config"]
    gateway = Proc("gateway", [*argv, "config.mock.yaml", "serve"], env, GATEWAY_PORT)
    gateway.wait_http("/readyz", timeout_s=120)
    return gateway


# ------------------------------------------------------------------ the client


class Session:
    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.session_id = ""

    @classmethod
    async def open(cls) -> Session:
        ws = await connect(
            f"ws://{HOST}:{GATEWAY_PORT}/v1/realtime?model=mlx-realtime", max_size=None, proxy=None
        )
        session = cls(ws)
        created = await session.until("session.created")
        session.session_id = created[-1][1]["session"]["id"]
        await session.until("conversation.created")
        return session

    async def close(self) -> None:
        await self.ws.close()

    async def send(self, event: dict[str, Any]) -> float:
        await self.ws.send(json.dumps(event, ensure_ascii=False))
        return time.perf_counter()

    async def recv(self, timeout_s: float = 60) -> tuple[float, dict[str, Any]]:
        async with asyncio.timeout(timeout_s):
            raw = await self.ws.recv()
        return time.perf_counter(), json.loads(raw)

    async def until(self, kind: str, timeout_s: float = 60) -> list[tuple[float, dict[str, Any]]]:
        seen = []
        while True:
            at, event = await self.recv(timeout_s)
            seen.append((at, event))
            if event["type"] == "error":
                print("  error:", event.get("error"), flush=True)
            if event["type"] == kind:
                return seen

    async def update(self, **session: Any) -> float:
        sent = await self.send({"type": "session.update", "session": {"type": "realtime", **session}})
        await self.until("session.updated")
        return sent

    async def item(self, role: str, text: str) -> None:
        part = "output_text" if role == "assistant" else "input_text"
        item = {"type": "message", "role": role, "content": [{"type": part, "text": text}]}
        await self.send({"type": "conversation.item.create", "item": item})
        await self.until("conversation.item.done")

    async def output(self, call_id: str, output: str, *, wait: bool = True) -> None:
        item = {"type": "function_call_output", "call_id": call_id, "output": output}
        await self.send({"type": "conversation.item.create", "item": item})
        if wait:
            await self.until("conversation.item.done")

    async def respond(
        self,
        response: dict[str, Any] | None = None,
        on_call: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {"type": "response.create"}
        if response:
            event["response"] = response
        sent = await self.send(event)
        out: dict[str, Any] = {"t_sent": sent, "calls": [], "text": "", "errors": []}
        while True:
            at, ev = await self.recv()
            kind = ev["type"]
            if kind == "response.created":
                out["t_created"] = at
                out["response_id"] = ev["response"]["id"]
            elif kind == "response.output_audio_transcript.delta":
                out.setdefault("t_first_transcript", at)
                out["text"] += ev["delta"]
            elif kind == "response.output_audio.delta":
                out.setdefault("t_first_audio", at)
            elif kind == "response.function_call_arguments.done":
                out.setdefault("t_first_call", at)
                call = {"name": ev["name"], "arguments": ev["arguments"], "call_id": ev["call_id"]}
                out["calls"].append(call)
                if on_call is not None:
                    await on_call(call)
            elif kind == "error":
                out["errors"].append(ev.get("error"))
            elif kind == "response.done":
                out["t_done"] = at
                done = ev["response"]
                out["status"] = done["status"]
                out["status_details"] = done.get("status_details")
                out["usage"] = done.get("usage")
                return out

    def debug_turns(self) -> list[dict[str, Any]]:
        answer = http(f"/debug/turns?session_id={self.session_id}", GATEWAY_PORT, timeout=10)
        return answer["turns"]


def row(case: str, r: dict[str, Any], **extra: Any) -> dict[str, Any]:
    created = r.get("t_created")

    def rel(key: str) -> float | None:
        return None if created is None or key not in r else round(r[key] - created, 4)

    usage = r.get("usage") or {}
    cached = (usage.get("input_token_details") or {}).get("cached_tokens")
    return {
        "case": case,
        "response_id": r.get("response_id"),
        "status": r.get("status"),
        "created_after_send": None if created is None else round(created - r["t_sent"], 4),
        "first_transcript": rel("t_first_transcript"),
        "first_audio": rel("t_first_audio"),
        "first_call": rel("t_first_call"),
        "done": rel("t_done"),
        "tool_only": bool(r["calls"]) and not r["text"],
        "calls": [(c["name"], c["arguments"]) for c in r["calls"]],
        "text": r["text"][:160],
        "input_tokens": usage.get("input_tokens"),
        "cached_tokens": cached,
        "output_tokens": usage.get("output_tokens"),
        "errors": r["errors"],
        "load1": round(os.getloadavg()[0], 1),
        **extra,
    }


def attach_marks(rows: list[dict[str, Any]], session: Session) -> None:
    """The gateway's marks for each response, in s after its response_created mark."""
    records = {t["response_id"]: t for t in session.debug_turns() if t.get("response_id")}
    for r in rows:
        record = records.get(r["response_id"])
        if record is None:
            continue
        marks = record["marks_ms"]
        base = marks.get("response_created")
        if base is None:
            continue
        for name in ("llm_first_token", "first_sentence_ready", "tts_first_chunk", "first_audio_sent"):
            if name in marks:
                r[f"srv_{name}"] = round((marks[name] - base) / 1000, 4)
        r["srv_values"] = record.get("values")


# ------------------------------------------------------------------ cases

SESSION_BASE = {"audio": {"input": {"turn_detection": None}}, "tools": A["tools"]}


async def answer_calls(session: Session, r: dict[str, Any]) -> None:
    """Keep the history valid when a setup turn called a tool: answer it, ask for nothing."""
    for call in r["calls"]:
        await session.output(call["call_id"], '{"ok":1}')


async def setup_call(session: Session, rnd: random.Random, i: int, turns: int) -> list[dict]:
    """A call with the session's instructions (cached across calls), the greeting, ``turns``
    caller turns answered."""
    await session.update(instructions=A["instructions"], **SESSION_BASE)
    await session.item("assistant", A["announce"])
    rows = []
    for j in range(turns):
        await session.item("user", CALLER[i % 20] if j == 0 else rnd.choice(FOLLOW))
        r = await session.respond()
        rows.append(row("setup", r))
        await answer_calls(session, r)
    return rows


async def case_cold(i: int, rnd: random.Random, gap: float) -> list[dict]:
    session = await Session.open()
    rows = []
    try:
        instructions = f"(通话 {uuid.uuid4().hex[:8]})\n" + A["instructions"]
        await session.update(instructions=instructions, **SESSION_BASE)
        await asyncio.sleep(gap)  # the greeting line plays, the caller speaks
        await session.item("assistant", A["announce"])
        await session.item("user", CALLER[i % 20])
        r = await session.respond()
        rows.append(row("cold", r, gap=gap))
        await answer_calls(session, r)
        for _ in range(2):
            await session.item("user", rnd.choice(FOLLOW))
            r = await session.respond()
            rows.append(row("warm", r))
            await answer_calls(session, r)
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


async def case_change_turn(i: int, rnd: random.Random, gap: float) -> list[dict]:
    session = await Session.open()
    try:
        rows = await setup_call(session, rnd, i, 2)
        await session.update(instructions=phase(PHASE_CLOSING + f"(变更 {nonce()})"))
        await asyncio.sleep(gap)
        await session.item("user", "好的，没有别的问题了。" if i % 2 else "那我再问一下，手机怎么截图？")
        r = await session.respond()
        rows.append(row("change_turn", r, gap=gap))
        await answer_calls(session, r)
        await session.item("user", rnd.choice(FOLLOW))
        rows.append(row("change_turn_next", await session.respond()))
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


async def case_transfer(i: int, rnd: random.Random, gap: float, *, change: bool) -> list[dict]:
    session = await Session.open()
    try:
        rows = await setup_call(session, rnd, i, 2)
        if change:
            await session.update(instructions=phase(PHASE_CLOSING + f"(变更 {nonce()})"))
            await asyncio.sleep(gap)
        await session.item("user", TRANSFER_TURN)
        r = await session.respond()
        rows.append(row("change_transfer" if change else "warm_transfer", r, gap=gap if change else None))
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


async def case_aicc(i: int, rnd: random.Random, *, transfer: bool) -> list[dict]:
    """AICC's sequence: the update and the call's output go out as soon as the call's arguments
    are done (the response still open); response.create once response.done arrives."""
    session = await Session.open()
    try:
        rows = await setup_call(session, rnd, i, 1)
        instructions = phase(PHASE_RESULT + f"(变更 {nonce()})")

        async def on_call(call: dict[str, Any]) -> None:
            if call["name"] != "repair_status":
                await session.output(call["call_id"], '{"ok":1}', wait=False)
                return
            await session.send(
                {"type": "session.update", "session": {"type": "realtime", "instructions": instructions}}
            )
            await session.output(call["call_id"], LOOKUP_RESULT, wait=False)

        await session.item("user", f"我的维修单号是RMA{rnd.randint(1000, 9999)}，帮我查一下进度。")
        r1 = await session.respond(on_call=on_call)
        rows.append(row("aicc_lookup", r1))
        if not any(call["name"] == "repair_status" for call in r1["calls"]):
            rows[-1]["note"] = "no repair_status call: the sequence did not happen"
            attach_marks(rows, session)
            return rows
        r2 = await session.respond()
        rows.append(row("aicc_result", r2))
        await answer_calls(session, r2)
        await session.item("user", TRANSFER_TURN if transfer else rnd.choice(FOLLOW))
        r3 = await session.respond()
        rows.append(row("aicc_next_transfer" if transfer else "aicc_next_spoken", r3))
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


async def case_override(i: int, rnd: random.Random) -> list[dict]:
    session = await Session.open()
    try:
        rows = await setup_call(session, rnd, i, 2)
        await session.item("user", "嗯。")
        override = OVERRIDES[i % 3] + f"(提示 {nonce()})"
        r = await session.respond({"instructions": override})
        rows.append(row("override", r))
        await answer_calls(session, r)
        await session.item("user", rnd.choice(FOLLOW))
        rows.append(row("override_next", await session.respond()))
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


async def case_interleaved(i: int, rnd: random.Random, calls: int, rounds: int) -> list[dict]:
    """``calls`` calls of different flows (their own instructions) taking turns, one caller turn
    each per round: how much of each turn the prompt cache still holds."""
    sessions = [await Session.open() for _ in range(calls)]
    rows: list[dict] = []
    try:
        for k, session in enumerate(sessions):
            instructions = f"(流程 {k} {nonce()})\n" + A["instructions"]
            await session.update(instructions=instructions, **SESSION_BASE)
            await session.item("assistant", A["announce"])
        await asyncio.sleep(3.0)  # the greetings play (and the warm-ups run)
        for j in range(rounds):
            for k, session in enumerate(sessions):
                await session.item("user", CALLER[(i + k + j) % 20] if j == 0 else rnd.choice(FOLLOW))
                r = await session.respond()
                rows.append(row(f"interleaved_{calls}", r, round=j, call=k))
                await answer_calls(session, r)
        for session in sessions:
            attach_marks([r for r in rows], session)
    finally:
        for session in sessions:
            await session.close()
    return rows


STORY = {"instructions": "你是一个讲故事的人，故事要讲得很长很细。", "audio": {"input": {"turn_detection": None}}}


async def case_transfer_contended(i: int, rnd: random.Random, others: int) -> list[dict]:
    """The transfer turn while ``others`` other calls are decoding long answers (the LLM
    batches them: every stream decodes slower)."""
    fg = await Session.open()
    backgrounds = [await Session.open() for _ in range(others)]
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        rows = await setup_call(fg, rnd, i, 2)
        for bg in backgrounds:
            await bg.update(**STORY)
            await bg.item("user", "讲一个很长的关于灯塔守护人的故事。")
            tasks.append(asyncio.create_task(bg.respond()))
        await asyncio.sleep(1.0)  # every background stream is past its prefill, decoding
        await fg.item("user", TRANSFER_TURN)
        r = await fg.respond()
        rows.append(row(f"transfer_contended_{others}", r))
        for task in tasks:
            await task
        attach_marks(rows, fg)
    finally:
        for task in tasks:
            task.cancel()
        for session in (fg, *backgrounds):
            await session.close()
    return rows


async def case_line(i: int) -> list[dict]:
    session = await Session.open()
    try:
        await session.update(instructions=A["instructions"], **SESSION_BASE)
        r = await session.respond({"instructions": LINE_PREFIX + A["announce"]})
        rows = [row("line", r)]
        attach_marks(rows, session)
    finally:
        await session.close()
    return rows


# ------------------------------------------------------------------ stats


def pq(values: list[float | None]) -> dict[str, Any]:
    """n, p50, p95 (nearest rank), max, min, mean; as B1's llmlib.pq."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    k = max(0, min(len(v) - 1, int(round(0.95 * len(v) + 0.5)) - 1))
    return {"n": len(v), "p50": round(statistics.median(v), 3), "p95": round(v[k], 3),
            "max": round(v[-1], 3), "min": round(v[0], 3), "mean": round(statistics.fmean(v), 3)}  # fmt: skip


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in sorted({r["case"] for r in rows} - {"setup"}):
        rs = [r for r in rows if r["case"] == case]
        spoken = [r for r in rs if not r["tool_only"]]
        tool_only = [r for r in rs if r["tool_only"]]
        out[case] = {
            "n": len(rs),
            "spoken": len(spoken),
            "tool_only": len(tool_only),
            "failed": sum(r["status"] != "completed" for r in rs),
            "boundary": pq([r.get("srv_first_sentence_ready") for r in spoken]),
            "first_audio_client": pq([r["first_audio"] for r in spoken]),
            "llm_first_token": pq([r.get("srv_llm_first_token") for r in rs]),
            "tool_only_done": pq([r["done"] for r in tool_only]),
            "uncached_tokens": pq(
                [(r["input_tokens"] or 0) - (r["cached_tokens"] or 0) for r in rs if r["input_tokens"]]
            ),
            "calls": sorted({c[0] for r in rs for c in r["calls"]}),
        }
    return out


def warmup_log(gateway: Proc) -> dict[str, Any]:
    warmed, waited, failed = [], [], 0
    for line in gateway.log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("msg") == "prefix warmed":
            warmed.append(record.get("warmup_ms"))
        elif record.get("msg") == "waited for a prefix warm-up":
            waited.append(record.get("waited_ms"))
        elif record.get("msg") == "prefix warm-up failed":
            failed += 1
    return {"warmed_ms": pq(warmed), "waited_ms": pq(waited), "failed": failed}


def cache_log(backend: Proc) -> dict[str, Any]:
    sizes = [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"Prompt Cache: (\d+) sequences, ([\d.]+) GB", backend.log.read_text())
    ]
    return {"max_sequences": max((s for s, _ in sizes), default=None),
            "max_gb": max((g for _, g in sizes), default=None), "samples": len(sizes)}  # fmt: skip


# ------------------------------------------------------------------ main


async def run_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    rnd = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    cases = args.cases.split(",")
    for case in cases:
        t0 = time.time()
        for i in range(args.n):
            match case:
                case "cold":
                    rows += await case_cold(i, rnd, args.gap)
                case "change_turn":
                    rows += await case_change_turn(i, rnd, args.gap)
                case "change_transfer":
                    rows += await case_transfer(i, rnd, args.gap, change=True)
                case "warm_transfer":
                    rows += await case_transfer(i, rnd, args.gap, change=False)
                case "aicc_spoken":
                    rows += await case_aicc(i, rnd, transfer=False)
                case "aicc_transfer":
                    rows += await case_aicc(i, rnd, transfer=True)
                case "override":
                    rows += await case_override(i, rnd)
                case "line":
                    rows += await case_line(i)
                case "interleaved_4":
                    rows += await case_interleaved(i, rnd, 4, 5)
                case "transfer_contended_1" | "transfer_contended_2":
                    rows += await case_transfer_contended(i, rnd, int(case[-1]))
                case _:
                    raise SystemExit(f"unknown case {case}")
        print(f"[{time.strftime('%H:%M:%S')}] {case}: {time.time() - t0:.0f} s", flush=True)
        stats = summarize([r for r in rows if r["case"].startswith(case.split("_")[0])])
        for name, s in stats.items():
            print(f"   {name}: boundary {s['boundary'].get('p50')}/{s['boundary'].get('p95')} "
                  f"tool-only done {s['tool_only_done'].get('p50')}/{s['tool_only_done'].get('p95')} "
                  f"uncached {s['uncached_tokens'].get('p50')}", flush=True)  # fmt: skip
    return rows


def env_info() -> dict[str, Any]:
    def run(*argv: str) -> str:
        return subprocess.run(argv, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603

    return {
        "pmset_powermode": next(
            (ln.strip() for ln in run("pmset", "-g").splitlines() if "powermode" in ln), None
        ),
        "power_source": run("pmset", "-g", "batt").splitlines()[0] if run("pmset", "-g", "batt") else None,
        "loadavg": os.getloadavg(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--warm", choices=["on", "off"], required=True)
    parser.add_argument("--server-args", default="")
    parser.add_argument(
        "--cases",
        default="line,cold,change_turn,change_transfer,warm_transfer,aicc_spoken,aicc_transfer,override",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--gap", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    extra = args.server_args.split()
    before = env_info()
    backend = start_backend(extra)
    gateway: Proc | None = None
    try:
        footprint_start = backend.footprint_gb()
        gateway = start_gateway(warm=args.warm == "on")
        t0 = time.time()
        rows = asyncio.run(run_cases(args))
        wall = time.time() - t0
        footprint_end = backend.footprint_gb()
        result = {
            "spike": "LLM hardening: AICC's cases on the real LLM through the gateway",
            "name": args.name,
            "conditions": {
                "model": MODEL, "revision": REVISION, "server": " ".join(backend.argv),
                "server_flags": G2_FLAGS + extra, "warm_on_update": args.warm == "on",
                "gateway": "config.mock.yaml + real LLM provider (MLXRT__PROVIDERS__LLM), mock ASR/TTS at 0 delay",
                "concurrency": 1, "n": args.n, "gap_s": args.gap, "seed": args.seed,
                "env_before": before, "env_after": env_info(), "wall_s": round(wall, 1),
                "footprint_gb_start": footprint_start, "footprint_gb_end": footprint_end,
                "prompt_cache": cache_log(backend), "logs": [str(backend.log), str(gateway.log)],
            },  # fmt: skip
            "stats": summarize(rows),
            "warmups": warmup_log(gateway),
            "rows": rows,
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"llm_hardening_{args.name}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"-> {path}", flush=True)
    finally:
        if gateway is not None:
            gateway.stop()
        backend.stop()


if __name__ == "__main__":
    main()

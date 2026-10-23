"""Shared helpers for the M0 LLM spikes S1/S2 (mlx_lm.server 0.32.2, Qwen3.x-35B-A3B-4bit).

- Server lifecycle: start `mlx_lm.server` on 127.0.0.1:8101 with HF_HUB_OFFLINE=1 and `--model <snapshot
  dir>`, wait for /health, then warm up with a real request (/health is 200 before the model loads).
- A streaming client on http.client that timestamps every SSE line with time.perf_counter_ns on the
  client side: headers, first token, first content, first segment boundary, first tool call, [DONE].
- Gateway-shaped request bodies (src/mlx_realtime/providers/chat_wire.py): no `model`, system first,
  later system items folded into user notes, a synthetic cue when there is no user message,
  chat_template_kwargs enable_thinking false, stream + include_usage, max_completion_tokens, no seed,
  no tools for tool_choice none.
- The AICC fixture (line_session_zh.json, read-only) as system prompt + OpenAI function tools.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOST, PORT = "127.0.0.1", 8101
SPIKE_DIR = Path(__file__).resolve().parent
REPO = SPIKE_DIR.parent.parent
RESULTS = SPIKE_DIR.parent / "results"
AICC_FIXTURE = (
    Path(os.environ.get("AICC_REPO", "~/workspaces/github/ai-native-callcenter")).expanduser()
    / "internal/provider/testdata/line_session_zh.json"
)
LOG_DIR = Path(os.environ.get("SPIKE_LOG_DIR", "/tmp")).resolve()

MODELS = {
    "qwen3.6": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "qwen3.5": "mlx-community/Qwen3.5-35B-A3B-4bit",
}

# The source card's (Qwen/Qwen3.6-35B-A3B README) non-thinking sampling:
# temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0.
# Only temp/top_p/top_k/min_p have server flags (--help); presence/repetition penalty are per request.
CARD = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
        "repetition_penalty": 1.0}
BASE_FLAGS = [
    "--chat-template-args", '{"enable_thinking":false}',
    "--max-tokens", "2048",
    "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
]
# What the gateway sends per request (config.py MlxLmChatConfig defaults).
GATEWAY_SAMPLING = {"temperature": 0.7, "top_p": 0.8}
GATEWAY_MAX_COMPLETION = 512
BOUNDARY = re.compile(r"[，。！？,.!?]")
THINK = re.compile(r"</?think>")


def snapshot(model: str) -> str:
    models = json.loads((RESULTS / "models.json").read_text())["models"]
    for m in models:
        if m["repo"] == model:
            return m["snapshot"]
    raise KeyError(model)


def revision(model: str) -> str:
    models = json.loads((RESULTS / "models.json").read_text())["models"]
    return next(m["sha"] for m in models if m["repo"] == model)


def loadavg() -> float:
    return os.getloadavg()[0]


def pq(values: list[float]) -> dict[str, Any]:
    """n, p50, p95, max (nearest-rank p95), mean."""
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0}
    k = max(0, min(len(v) - 1, int(round(0.95 * len(v) + 0.5)) - 1))
    return {"n": len(v), "p50": round(statistics.median(v), 4), "p95": round(v[k], 4),
            "max": round(v[-1], 4), "min": round(v[0], 4), "mean": round(statistics.fmean(v), 4)}


# ---------------------------------------------------------------- server lifecycle


@dataclass
class Server:
    model: str
    extra_flags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    tag: str = "server"
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    started_ns: int = 0
    ready_s: float | None = None
    first_request_s: float | None = None

    @property
    def flags(self) -> list[str]:
        return BASE_FLAGS + self.extra_flags

    @property
    def command(self) -> list[str]:
        return ["uv", "run", "--project", str(SPIKE_DIR), "mlx_lm.server", "--model", snapshot(self.model),
                "--host", HOST, "--port", str(PORT), *self.flags]

    def start(self) -> None:
        if port_open():
            raise RuntimeError(f"port {PORT} already in use")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"mlx_lm_server_{self.tag}_{int(time.time())}.log"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", **self.env}
        self.started_ns = time.perf_counter_ns()
        self.proc = subprocess.Popen(self.command, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,
                                     env=env, cwd=REPO)
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited: {self.log_path.read_text()[-3000:]}")
            try:
                status, _ = get("/health")
                if status == 200:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        self.ready_s = (time.perf_counter_ns() - self.started_ns) / 1e9
        # /health is 200 before the model is loaded: a real request is the readiness check.
        r = stream_chat({"messages": [{"role": "system", "content": "warmup"}, {"role": "user", "content": "hi"}],
                         "stream": True, "stream_options": {"include_usage": True}, "max_completion_tokens": 1,
                         "chat_template_kwargs": {"enable_thinking": False}}, timeout=900)
        self.first_request_s = (time.perf_counter_ns() - self.started_ns) / 1e9
        if r.status != 200:
            raise RuntimeError(f"warmup failed {r.status} {r.error_body}")

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path else ""

    def log_offset(self) -> int:
        return self.log_path.stat().st_size if self.log_path else 0

    def log_since(self, offset: int) -> str:
        if not self.log_path:
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")

    def pid(self) -> int | None:
        out = subprocess.run(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True)
        pids = out.stdout.split()
        return int(pids[0]) if pids else None

    def footprint_gb(self) -> float | None:
        pid = self.pid()
        if pid is None:
            return None
        out = subprocess.run(["footprint", "-p", str(pid), "--noCategories", "-f", "bytes"],
                             capture_output=True, text=True)
        m = re.search(r"Footprint:\s*(\d+)\s*B", out.stdout)
        return round(int(m.group(1)) / 1e9, 2) if m else None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        # uv run spawns a child python; make sure the listener is gone.
        for _ in range(50):
            if not port_open():
                break
            pid = self.pid()
            if pid:
                os.kill(pid, 15)
            time.sleep(0.2)

    def conditions(self) -> dict[str, Any]:
        return {"model": self.model, "revision": revision(self.model), "snapshot": snapshot(self.model),
                "flags": self.flags, "env": self.env, "command": " ".join(self.command),
                "ready_s": self.ready_s, "first_request_done_s": self.first_request_s,
                "log": str(self.log_path)}


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def get(path: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, resp.read().decode("utf-8", "replace")


# ---------------------------------------------------------------- streaming client


@dataclass
class Result:
    status: int = 0
    error_body: str | None = None
    t_headers: float | None = None  # seconds from send
    t_first_token: float | None = None  # first delta with content/reasoning/tool_calls
    t_first_content: float | None = None
    t_first_boundary: float | None = None
    t_first_tool_call: float | None = None
    t_last_token: float | None = None
    t_done: float | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_chunks: int = 0
    finish_reason: str | None = None
    usage: dict | None = None
    usage_before_done: bool | None = None
    n_data: int = 0
    n_token_chunks: int = 0
    n_keepalive: int = 0
    raw: list[str] = field(default_factory=list)
    closed_early: bool = False
    exception: str | None = None

    @property
    def prompt_tokens(self) -> int | None:
        return self.usage.get("prompt_tokens") if self.usage else None

    @property
    def cached_tokens(self) -> int | None:
        if not self.usage:
            return None
        d = self.usage.get("prompt_tokens_details")
        return d.get("cached_tokens") if d else None

    @property
    def completion_tokens(self) -> int | None:
        return self.usage.get("completion_tokens") if self.usage else None

    @property
    def decode_tps(self) -> float | None:
        ct = self.completion_tokens
        if ct and ct > 2 and self.t_first_token is not None and self.t_last_token and \
                self.t_last_token > self.t_first_token:
            return (ct - 1) / (self.t_last_token - self.t_first_token)
        return None

    def summary(self, keep_raw: bool = False) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in (
            "status", "error_body", "t_headers", "t_first_token", "t_first_content", "t_first_boundary",
            "t_first_tool_call", "t_last_token", "t_done", "content", "reasoning", "tool_calls",
            "tool_call_chunks", "finish_reason", "usage", "usage_before_done", "n_data", "n_token_chunks",
            "n_keepalive", "closed_early", "exception")}
        d["decode_tps"] = self.decode_tps
        d["think_markup"] = bool(THINK.search(self.content))
        if keep_raw:
            d["raw"] = self.raw
        return d


def stream_chat(body: dict, *, timeout: float = 300, close_after: str | None = None,
                keep_raw: bool = False, raw_limit: int = 400) -> Result:
    """POST /v1/chat/completions and read the SSE stream.

    close_after: "first_content" | "first_token" | "first_keepalive" | "first_byte" closes the socket at
    that point (cancellation tests).
    """
    r = Result()
    payload = json.dumps(body, ensure_ascii=False).encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    t0 = time.perf_counter_ns()

    def el() -> float:
        return (time.perf_counter_ns() - t0) / 1e9

    try:
        conn.connect()
        t0 = time.perf_counter_ns()
        conn.request("POST", "/v1/chat/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        r.t_headers = el()
        r.status = resp.status
        if resp.status != 200:
            r.error_body = resp.read().decode("utf-8", "replace")
            r.t_done = el()
            return r
        if not body.get("stream"):
            data = json.loads(resp.read())
            r.t_done = el()
            msg = data["choices"][0]["message"]
            r.content = msg.get("content") or ""
            r.reasoning = msg.get("reasoning") or ""
            r.tool_calls = msg.get("tool_calls") or []
            r.finish_reason = data["choices"][0].get("finish_reason")
            r.usage = data.get("usage")
            return r
        fp = resp.fp
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            text = line.decode("utf-8", "replace")
            if keep_raw and len(r.raw) < raw_limit:
                r.raw.append(f"{el():.4f} {text}")
            if text.startswith(":"):
                r.n_keepalive += 1
                if close_after == "first_keepalive" or (close_after == "second_keepalive" and r.n_keepalive >= 2):
                    r.closed_early = True
                    break
                continue
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                r.t_done = el()
                break
            r.n_data += 1
            chunk = json.loads(data)
            if chunk.get("usage") is not None:
                r.usage = chunk["usage"]
                r.usage_before_done = True
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason") is not None:
                r.finish_reason = ch["finish_reason"]
            delta = ch.get("delta") or {}
            got = False
            now = el()
            if delta.get("content"):
                got = True
                if r.t_first_content is None:
                    r.t_first_content = now
                r.content += delta["content"]
                if r.t_first_boundary is None and BOUNDARY.search(r.content):
                    r.t_first_boundary = now
            if delta.get("reasoning"):
                got = True
                r.reasoning += delta["reasoning"]
            if delta.get("tool_calls"):
                got = True
                r.tool_call_chunks += 1
                if r.t_first_tool_call is None:
                    r.t_first_tool_call = now
                r.tool_calls.extend(delta["tool_calls"])
            if got:
                r.n_token_chunks += 1
                if r.t_first_token is None:
                    r.t_first_token = now
                r.t_last_token = now
                if close_after == "first_token" or (close_after == "first_content" and delta.get("content")):
                    r.closed_early = True
                    break
        if r.usage_before_done is None and r.usage is None:
            r.usage_before_done = False
    except Exception as e:  # noqa: BLE001 - record and continue
        r.exception = f"{type(e).__name__}: {e}"
    finally:
        try:
            if conn.sock:
                conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()
    return r


# ---------------------------------------------------------------- gateway-shaped requests

CUE_GATEWAY = "(The conversation has just started.)"
SYSTEM_NOTE_GATEWAY = "(System note: {text})"


def gateway_messages(items: list[dict], *, cue: str = CUE_GATEWAY, system_note: str = SYSTEM_NOTE_GATEWAY) -> list[dict]:
    """chat_wire.chat_messages, re-implemented on plain dicts."""
    rest = list(items)
    leading = []
    while rest and rest[0]["role"] == "system":
        leading.append(rest.pop(0)["content"])
    out: list[dict] = []
    if leading:
        out.append({"role": "system", "content": "\n\n".join(leading)})
    for m in rest:
        role = m["role"]
        if role == "system":
            out.append({"role": "user", "content": system_note.format(text=m["content"])})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            prev = out[-1] if out else None
            if prev is not None and prev["role"] == "assistant":
                texts = [t for t in (str(prev["content"]), m.get("content") or "") if t]
                prev["content"] = "\n".join(texts)
                if calls:
                    prev["tool_calls"] = [*prev.get("tool_calls", []), *calls]
                continue
            e = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                e["tool_calls"] = calls
            out.append(e)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    if not any(e["role"] == "user" for e in out):
        out.append({"role": "user", "content": cue})
    return out


def gateway_body(messages: list[dict], tools: list[dict] | None = None, *, tool_choice: str = "auto",
                 max_completion_tokens: int = GATEWAY_MAX_COMPLETION, sampling: dict | None = None,
                 extra: dict | None = None, adapt: bool = False) -> dict:
    body: dict[str, Any] = {
        "messages": gateway_messages(messages) if adapt else messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_completion_tokens,
        **(GATEWAY_SAMPLING if sampling is None else sampling),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tool_choice == "auto" and tools:
        body["tools"] = tools
    if extra:
        body.update(extra)
    return body


def aicc() -> dict:
    d = json.loads(AICC_FIXTURE.read_text())
    tools = []
    for t in d["tools"]:
        fn = {"name": t["Name"]}
        if t.get("Description"):
            fn["description"] = t["Description"]
        if t.get("Parameters"):
            fn["parameters"] = t["Parameters"]
        tools.append({"type": "function", "function": fn})
    return {"instructions": d["instructions"], "announce": d["announce"], "tools": tools}


PHASE_MARK = "当前环节："


def aicc_phase_variant(instructions: str, phase_text: str) -> str:
    """Replace the text after 当前环节： (an instructions-only change, as AICC sends on a new phase)."""
    head = instructions.split(PHASE_MARK)[0]
    return head + PHASE_MARK + "\n" + phase_text


PHASE_CLOSING = ("结束通话:用户的问题已经解决。简短确认用户没有其它问题;用户表示没有了就道别并调用 hangup。"
                 "如果用户还有新问题,按使用类问题或查维修进度的流程处理。")
PHASE_MESSAGE = ("记录留言:用户需要回电。先问清楚回电号码和事由,号码要逐位复述确认,确认无误后调用 take_message,"
                 "工具返回后告知用户稍后会有客服回电。")


def write_json(name: str, data: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return p


def env_info() -> dict[str, Any]:
    out = {}
    out["pmset_custom"] = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True).stdout
    out["pmset_powermode_now"] = next((ln.strip() for ln in subprocess.run(
        ["pmset", "-g"], capture_output=True, text=True).stdout.splitlines() if "powermode" in ln), None)
    out["power_source"] = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.strip()
    out["iogpu_wired_limit_mb"] = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True,
                                                 text=True).stdout.strip()
    out["loadavg"] = os.getloadavg()
    return out

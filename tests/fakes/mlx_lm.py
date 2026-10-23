"""A fake ``mlx_lm.server`` (mlx-lm 0.32.2), for provider and integration tests.

It runs on the real server's HTTP stack: ``http.server.ThreadingHTTPServer`` with a
``BaseHTTPRequestHandler`` speaking HTTP/1.0, one thread per request, the response body ended by
closing the connection. What it writes copies ``mlx_lm/server.py``: ``: keepalive N/M`` comments,
``data: <chunk>`` events shaped by ``generate_response`` and ``completion_usage_response``, then
``data: [DONE]``; errors are a JSON ``{"error": ...}`` body (404 when generation cannot start).

Each POST plays the next script: a list of steps. A ``Hold`` waits for the test; while holding
(and between steps) the handler watches the socket, so a client that closes the connection is
seen at once, as the real server sees it at its next write.
"""

from __future__ import annotations

import asyncio
import json
import select
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

type Json = dict[str, Any]


class Gate:
    """A point where a request waits until the test releases it."""

    def __init__(self) -> None:
        self._released = threading.Event()
        self._reached = threading.Event()

    def release(self) -> None:
        self._released.set()

    async def reached(self, timeout_s: float = 5.0) -> None:
        async with asyncio.timeout(timeout_s):
            while not self._reached.is_set():  # noqa: ASYNC110 (set by a server thread)
                await asyncio.sleep(0.002)

    def wait(self, alive: Any) -> bool:
        """Block until released; ``False`` if the client went away first."""
        self._reached.set()
        while not self._released.wait(0.005):
            if not alive():
                return False
        return True


@dataclass(frozen=True)
class Keepalive:
    processed: int = 512
    total: int = 2048


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class Calls:
    """Complete calls in one chunk. The real server sends a call with the token after its
    ``</tool_call>``, with any text (often a newline) the model wrote since the last chunk."""

    calls: list[Json]
    text: str = ""


@dataclass(frozen=True)
class Finish:
    reason: str
    text: str = ""


@dataclass(frozen=True)
class Usage:
    prompt: int = 12
    completion: int = 3
    cached: int = 4


@dataclass(frozen=True)
class Done:
    pass


@dataclass(frozen=True)
class Hold:
    gate: Gate


@dataclass(frozen=True)
class Sleep:
    seconds: float


@dataclass(frozen=True)
class Fail:
    """An error answer instead of a stream (must be the first step)."""

    status: int
    error: str = "Failed to apply the chat template"


@dataclass(frozen=True)
class Close:
    """Stop and close the connection, as a crashed generation would."""


type Step = (
    Keepalive | Text | Reasoning | Calls | Finish | Usage | Done | Hold | Sleep | Fail | Close
)


def reply(*texts: str, finish: str = "stop") -> list[Step]:
    return [*(Text(text) for text in texts), Finish(finish), Usage(), Done()]


def tool_call(name: str, arguments: Json, call_id: str = "call-1", index: int = 0) -> Json:
    """One entry of ``delta.tool_calls``, as ToolCallFormatter writes it when streaming."""
    function = {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}
    return {"function": function, "type": "function", "id": call_id, "index": index}


@dataclass
class FakeMlxLm:
    default: list[Step] = field(default_factory=lambda: reply("好的", "。"))
    health: int = 200
    requests: list[Json] = field(default_factory=list[Json])
    disconnects: int = 0
    finished: int = 0
    _scripts: deque[list[Step]] = field(default_factory=deque[list[Step]])
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def script(self, *steps: Step) -> None:
        self._scripts.append(list(steps))

    def next_script(self) -> list[Step]:
        with self._lock:
            return self._scripts.popleft() if self._scripts else list(self.default)

    def count(self, what: str) -> None:
        with self._lock:
            setattr(self, what, getattr(self, what) + 1)

    async def disconnected(self, count: int = 1, timeout_s: float = 2.0) -> None:
        async with asyncio.timeout(timeout_s):
            while self.disconnects < count:  # noqa: ASYNC110 (a counter set by server threads)
                await asyncio.sleep(0.002)

    @contextmanager
    def serving(self) -> Iterator[str]:
        """Serve on a free port of 127.0.0.1; yields the base URL."""
        fake = self

        class Handler(_Handler):
            server_fake = fake

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()


class _Handler(BaseHTTPRequestHandler):
    server_fake: FakeMlxLm

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path != "/health":
            self._answer(404, b"Not Found", "application/json")
            return
        status = self.server_fake.health
        body = json.dumps({"status": "ok" if status == 200 else "unavailable"}).encode()
        self._answer(status, body, "application/json")

    def do_POST(self) -> None:
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._answer(404, b"Not Found", "application/json")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body: Json = json.loads(self.rfile.read(length))
        self.server_fake.requests.append(body)
        self._model = body.get("model", "default_model")
        self._id = f"chatcmpl-{uuid.uuid4()}"
        self._started = False
        try:
            self._play(self.server_fake.next_script())
        except (BrokenPipeError, ConnectionResetError):
            self.server_fake.count("disconnects")

    def _play(self, steps: list[Step]) -> None:
        for step in steps:
            if not self._alive():
                self.server_fake.count("disconnects")
                return
            match step:
                case Fail(status=status, error=error):
                    self._answer(status, json.dumps({"error": error}).encode(), "application/json")
                    return
                case Hold(gate=gate):
                    if not gate.wait(self._alive):
                        self.server_fake.count("disconnects")
                        return
                case Sleep(seconds=seconds):
                    deadline = time.monotonic() + seconds
                    while time.monotonic() < deadline:
                        if not self._alive():
                            self.server_fake.count("disconnects")
                            return
                        time.sleep(0.002)
                case Close():
                    return
                case Keepalive(processed=processed, total=total):
                    self._write(f": keepalive {processed}/{total}\n\n")
                case Done():
                    self._write("data: [DONE]\n\n")
                case Usage():
                    self._event(self._usage(step))
                case _:
                    self._event(self._chunk(step))
        self.server_fake.count("finished")

    def _chunk(self, step: Text | Reasoning | Calls | Finish) -> Json:
        # generate_response(), stream=True
        finish = step.reason if isinstance(step, Finish) else None
        delta: Json = {"role": "assistant"}
        text = step.text if isinstance(step, Text | Finish | Calls) else ""
        if text:
            delta["content"] = text
        if isinstance(step, Reasoning):
            delta["reasoning"] = step.text
        if isinstance(step, Calls):
            delta["tool_calls"] = step.calls
        choice: Json = {"index": 0, "finish_reason": finish, "delta": delta}
        return self._head("chat.completion.chunk") | {"choices": [choice]}

    def _usage(self, usage: Usage) -> Json:
        # completion_usage_response()
        counts: Json = {
            "prompt_tokens": usage.prompt,
            "completion_tokens": usage.completion,
            "total_tokens": usage.prompt + usage.completion,
            "prompt_tokens_details": {"cached_tokens": usage.cached},
        }
        return self._head("chat.completion") | {"choices": [], "usage": counts}

    def _head(self, kind: str) -> Json:
        return {
            "id": self._id,
            "system_fingerprint": "0.32.2-fake",
            "object": kind,
            "model": self._model,
            "created": int(time.time()),
        }

    def _event(self, payload: Json) -> None:
        self._write(f"data: {json.dumps(payload)}\n\n")

    def _write(self, text: str) -> None:
        if not self._started:
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._started = True
        self.wfile.write(text.encode())
        self.wfile.flush()

    def _answer(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _alive(self) -> bool:
        """The request body has been read: anything readable now is the client closing."""
        connection: socket.socket = self.connection
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return True
        try:
            return connection.recv(1, socket.MSG_PEEK) != b""
        except OSError:
            return False

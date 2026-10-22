"""A fake ``mlx_audio.server`` (mlx-audio 0.6.1) for the transcription route, on uvicorn as the
real server runs.

- ``POST /v1/audio/transcriptions`` takes the multipart form of ``stt_transcriptions`` and, for
  ``response_format=verbose_json``, answers the model's whole payload, shaped as the M0 smoke
  test recorded it (``spikes/results/a1_audio_smoke.json``): ``text``, ``segments``, ``language``
  as a list of names, and token counts. A model is "loaded" by its first request.
- ``GET /v1/models`` lists the loaded models, as ``ModelProvider.get_available_models`` does.
- Errors are FastAPI's ``{"detail": ...}``: 404 for an unknown model, 500 for a failed load.

Each transcription plays the next script; a ``Hold`` waits for the test, and a client that
disconnects meanwhile is counted.
"""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import io
import json
import socket
import time
import wave
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import uvicorn

type Json = dict[str, Any]
type Receive = Callable[[], Awaitable[Json]]
type Send = Callable[[Json], Awaitable[None]]


class Gate:
    def __init__(self) -> None:
        self._released = asyncio.Event()
        self._reached = asyncio.Event()

    def release(self) -> None:
        self._released.set()

    async def reached(self, timeout_s: float = 5.0) -> None:
        async with asyncio.timeout(timeout_s):
            await self._reached.wait()

    async def wait(self) -> None:
        self._reached.set()
        await self._released.wait()


@dataclass(frozen=True)
class Transcribe:
    text: str
    languages: tuple[str, ...] = ("Chinese",)


@dataclass(frozen=True)
class Hold:
    gate: Gate


@dataclass(frozen=True)
class Fail:
    status: int
    detail: str = "Failed to load model"


@dataclass(frozen=True)
class Garbage:
    body: bytes


type Step = Transcribe | Hold | Fail | Garbage


@dataclass(frozen=True)
class Upload:
    """One transcription request as the server received it."""

    fields: dict[str, str]
    wav: bytes

    def audio(self) -> tuple[int, int, int, bytes]:
        """(rate, channels, sample width, frames) of the uploaded WAV."""
        with wave.open(io.BytesIO(self.wav)) as file:
            frames = file.readframes(file.getnframes())
            return file.getframerate(), file.getnchannels(), file.getsampwidth(), frames


@dataclass
class FakeMlxAudio:
    default: list[Step] = field(default_factory=lambda: [Transcribe("你好。")])
    uploads: list[Upload] = field(default_factory=list[Upload])
    loaded: list[str] = field(default_factory=list[str])
    disconnects: int = 0
    _scripts: deque[list[Step]] = field(default_factory=deque[list[Step]])

    def script(self, *steps: Step) -> None:
        self._scripts.append(list(steps))

    async def __call__(self, scope: Json, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            return
        path, method = scope["path"], scope["method"]
        if (method, path) == ("GET", "/v1/models"):
            created = int(time.time())
            data = [
                {"id": m, "object": "model", "created": created, "owned_by": "system"}
                for m in self.loaded
            ]
            await _json(send, 200, {"object": "list", "data": data})
        elif (method, path) == ("POST", "/v1/audio/transcriptions"):
            await self._transcribe(scope, receive, send)
        else:
            await _json(send, 404, {"detail": "Not Found"})

    async def _transcribe(self, scope: Json, receive: Receive, send: Send) -> None:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
        upload = _parse_form(headers["content-type"], body)
        self.uploads.append(upload)
        script = self._scripts.popleft() if self._scripts else list(self.default)
        disconnected = asyncio.ensure_future(_disconnect(receive))
        try:
            for step in script:
                match step:
                    case Hold(gate=gate):
                        waiting = asyncio.ensure_future(gate.wait())
                        await asyncio.wait({waiting, disconnected}, return_when="FIRST_COMPLETED")
                        if disconnected.done():
                            waiting.cancel()
                            self.disconnects += 1
                            return
                    case Fail(status=status, detail=detail):
                        await _json(send, status, {"detail": detail})
                        return
                    case Garbage(body=garbage):
                        await _respond(send, 200, garbage, b"application/json")
                        return
                    case Transcribe():
                        await self._answer(send, upload, step)
                        return
        finally:
            disconnected.cancel()

    async def _answer(self, send: Send, upload: Upload, step: Transcribe) -> None:
        model = upload.fields["model"]
        if model not in self.loaded:
            self.loaded.append(model)
        rate, _, _, frames = upload.audio()
        end = round(len(frames) / 2 / rate, 2)
        language = step.languages[0] if step.languages else None
        payload = {
            "text": step.text,
            "segments": [{"text": step.text, "language": language, "start": 0.0, "end": end}],
            "language": list(step.languages),
            "prompt_tokens": 96,
            "generation_tokens": 16,
            "total_tokens": 112,
            "prompt_tps": 435.2,
            "generation_tps": 72.5,
            "total_time": 0.22,
        }
        await _json(send, 200, payload)


async def _disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _parse_form(content_type: str, body: bytes) -> Upload:
    head = f"Content-Type: {content_type}\r\n\r\n".encode()
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + body)
    fields: dict[str, str] = {}
    wav = b""
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        assert isinstance(payload, bytes)
        if name == "file":
            wav = payload
        else:
            fields[str(name)] = payload.decode()
    return Upload(fields, wav)


async def _json(send: Send, status: int, payload: Json) -> None:
    await _respond(send, status, json.dumps(payload).encode(), b"application/json")


async def _respond(send: Send, status: int, body: bytes, content_type: bytes) -> None:
    headers = [(b"content-type", content_type), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def serve_asgi(app: Callable[..., Awaitable[None]]) -> AsyncIterator[str]:
    """Serve ``app`` with uvicorn on a free port of 127.0.0.1; yields the base URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:  # noqa: ASYNC110 (uvicorn exposes a flag)
                await asyncio.sleep(0.005)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()

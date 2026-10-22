"""A fake ``mlx_audio.server`` (mlx-audio 0.6.1) for the transcription and speech routes, on
uvicorn as the real server runs.

- ``POST /v1/audio/transcriptions`` takes the multipart form of ``stt_transcriptions`` and, for
  ``response_format=verbose_json``, answers the model's whole payload, shaped as the M0 smoke
  test recorded it (``spikes/results/a1_audio_smoke.json``): ``text``, ``segments``, ``language``
  as a list of names, and token counts. A model is "loaded" by its first request.
- ``POST /v1/audio/speech`` takes ``tts_speech``'s JSON. The model is checked first (the real
  server's preflight load: 404 for an unknown model), then a 200 ``audio/pcm`` response streams
  the speech as chunked s16le at 24 kHz: ``ms_per_char`` of tone per non-space input character,
  in chunks of ``max(1, int(streaming_interval × 12.5))`` codec frames of 80 ms, as Qwen3-TTS
  streams; a Kokoro model sends everything in one chunk (it streams per sentence). A voice the
  model cannot use fails inside generation, which on the real server cuts the chunked body
  after the 200 status (M0 A1): a Qwen3-TTS speaker outside its list, or a Kokoro voice that is
  not a ``.safetensors`` path.
- ``GET /v1/models`` lists the loaded models, as ``ModelProvider.get_available_models`` does.
- Errors are FastAPI's ``{"detail": ...}``: 404 for an unknown model, 500 for a failed load.

Each request plays the next script of its route; a ``Hold`` waits for the test, and a client that
disconnects meanwhile is counted, as is one that disconnects between speech chunks (the real
server checks for that after every chunk).
"""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import io
import json
import math
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
class Start:
    """Send the 200 headers: the model preflight is done, generation begins. Steps before it
    (a ``Hold``) keep the client waiting for headers; data steps start the response
    themselves."""


@dataclass(frozen=True)
class Chunks:
    """Stream the next ``count`` chunks of the request's speech."""

    count: int


@dataclass(frozen=True)
class Rest:
    """Stream every remaining chunk, then end the response."""


@dataclass(frozen=True)
class Cut:
    """Fail inside generation: the connection closes without ending the chunked body."""


@dataclass(frozen=True)
class Raw:
    """Send these bytes as one chunk of the body."""

    data: bytes


@dataclass(frozen=True)
class Pause:
    seconds: float


type SpeechStep = Start | Hold | Fail | Chunks | Rest | Cut | Raw | Pause

QWEN3_SPEAKERS = frozenset(
    {"vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"}
)
SPEECH_RATE = 24_000
FRAME_MS = 80


class _Cut(Exception):
    """Raised inside the app to make uvicorn drop the connection mid-body."""


@dataclass
class Speech:
    """One speech request as the server received it, and how it ended."""

    body: dict[str, Any]
    received_at: float
    chunks: list[tuple[float, int]] = field(default_factory=list[tuple[float, int]])
    ended: str = ""  # "done", "disconnect", "cut", "error"

    @property
    def audio_bytes(self) -> int:
        return sum(size for _, size in self.chunks)


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
    # Speech: the models the server can load, and the pace of its chunks.
    models: set[str] = field(
        default_factory=lambda: {
            "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
            "mlx-community/Kokoro-82M-bf16",
            "mlx-community/Qwen3-ASR-1.7B-8bit",
        }
    )
    ms_per_char: int = 60
    first_chunk_s: float = 0.0
    chunk_s: float = 0.0
    speeches: list[Speech] = field(default_factory=list[Speech])
    _scripts: deque[list[Step]] = field(default_factory=deque[list[Step]])
    _speech_scripts: deque[list[SpeechStep]] = field(default_factory=deque[list[SpeechStep]])

    def script(self, *steps: Step) -> None:
        self._scripts.append(list(steps))

    def script_speech(self, *steps: SpeechStep) -> None:
        """The steps of the next speech request (default: ``Start(), Rest()``)."""
        self._speech_scripts.append(list(steps))

    async def disconnected(self, count: int = 1, timeout_s: float = 2.0) -> None:
        async with asyncio.timeout(timeout_s):
            while self.disconnects < count:  # noqa: ASYNC110 (a counter the app sets)
                await asyncio.sleep(0.002)

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
        elif (method, path) == ("POST", "/v1/audio/speech"):
            await self._speak(receive, send)
        else:
            await _json(send, 404, {"detail": "Not Found"})

    async def _speak(self, receive: Receive, send: Send) -> None:
        raw = b""
        while True:
            message = await receive()
            raw += message.get("body", b"")
            if not message.get("more_body"):
                break
        speech = Speech(json.loads(raw), time.monotonic())
        self.speeches.append(speech)
        body = speech.body
        script = self._speech_scripts.popleft() if self._speech_scripts else [Start(), Rest()]
        disconnected = asyncio.ensure_future(_disconnect(receive))
        try:
            await self._stream_speech(speech, body, script, send, disconnected)
        finally:
            disconnected.cancel()

    async def _stream_speech(
        self,
        speech: Speech,
        body: dict[str, Any],
        script: list[SpeechStep],
        send: Send,
        disconnected: asyncio.Future[None],
    ) -> None:
        model = str(body.get("model"))
        if model not in self.models:
            speech.ended = "error"
            detail = f"Model not found: {model!r} is not a known HuggingFace repo or local path"
            await _json(send, 404, {"detail": detail})
            return
        chunks = _speech_chunks(body, self.ms_per_char)
        started = False

        async def start() -> None:
            nonlocal started
            if not started:
                started = True
                headers = [(b"content-type", b"audio/pcm")]
                await send({"type": "http.response.start", "status": 200, "headers": headers})

        async def wait(seconds: float) -> bool:
            """Sleep, watching the client; False once it is gone."""
            if seconds > 0:
                await asyncio.wait({disconnected}, timeout=seconds)
            if disconnected.done():
                speech.ended = "disconnect"
                self.disconnects += 1
                return False
            return True

        async def chunk(data: bytes) -> bool:
            if not await wait(self.first_chunk_s if not speech.chunks else self.chunk_s):
                return False
            await send({"type": "http.response.body", "body": data, "more_body": True})
            speech.chunks.append((time.monotonic(), len(data)))
            if model not in self.loaded:
                self.loaded.append(model)
            return True

        if not _voice_ok(body):
            script = [Cut()]
        for step in script:
            match step:
                case Start():
                    await start()
                case Fail(status=status, detail=detail):
                    speech.ended = "error"
                    await _json(send, status, {"detail": detail})
                    return
                case Hold(gate=gate):
                    waiting = asyncio.ensure_future(gate.wait())
                    await asyncio.wait({waiting, disconnected}, return_when="FIRST_COMPLETED")
                    if disconnected.done():
                        waiting.cancel()
                        speech.ended = "disconnect"
                        self.disconnects += 1
                        return
                case Pause(seconds=seconds):
                    if not await wait(seconds):
                        return
                case Raw(data=data):
                    await start()
                    await send({"type": "http.response.body", "body": data, "more_body": True})
                    speech.chunks.append((time.monotonic(), len(data)))
                case Cut():
                    await start()
                    speech.ended = "cut"
                    raise _Cut("generation failed")
                case Chunks() | Rest():
                    await start()
                    count = step.count if isinstance(step, Chunks) else len(chunks)
                    for _ in range(count):
                        if not chunks:
                            break
                        if not await chunk(chunks.popleft()):
                            return
        await start()
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        speech.ended = speech.ended or "done"

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


def _speech_chunks(body: dict[str, Any], ms_per_char: int) -> deque[bytes]:
    """The request's speech: a tone as long as its text, in the model's chunks."""
    text = str(body.get("input", ""))
    samples = sum(1 for char in text if not char.isspace()) * ms_per_char * SPEECH_RATE // 1000
    tone = b"".join(
        int(6000 * math.sin(2 * math.pi * 220 * n / SPEECH_RATE)).to_bytes(2, "little", signed=True)
        for n in range(samples)
    )
    if "kokoro" in str(body.get("model", "")).lower():
        return deque([tone] if tone else [])
    interval = float(body.get("streaming_interval", 2.0))
    frames = max(1, int(interval * 12.5))
    size = frames * FRAME_MS * SPEECH_RATE // 1000 * 2
    return deque(tone[start : start + size] for start in range(0, len(tone), size))


def _voice_ok(body: dict[str, Any]) -> bool:
    voice = str(body.get("voice") or "")
    if "kokoro" in str(body.get("model", "")).lower():
        return voice.endswith(".safetensors")
    return voice.lower() in QWEN3_SPEAKERS


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

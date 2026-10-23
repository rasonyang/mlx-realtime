"""Shared test tooling: a gateway with gated mocks, an in-process server, a raw client, and the
golden-trace normalization."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import socket
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, get_args

import numpy as np
from openai import BaseModel as SDKModel
from openai.types.realtime import RealtimeServerEvent, RealtimeSessionCreateRequest
from websockets.asyncio.client import ClientConnection, connect

from mlx_realtime.audio import g711
from mlx_realtime.audio.g711 import Law
from mlx_realtime.config import validate_config
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR, SineTTS
from mlx_realtime.providers.registry import Providers
from mlx_realtime.server.app import GatewayServer
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.session.clock import Clock
from mlx_realtime.vad.base import TurnEndDetector
from mlx_realtime.vad.scripted import ScriptedSpeech

type Event = dict[str, Any]

TIMEOUT_S = 5.0
WIRE_RATE = 24_000


_SDK_EVENTS: dict[str, type[SDKModel]] = {
    get_args(cls.model_fields["type"].annotation)[0]: cls
    for cls in get_args(get_args(RealtimeServerEvent)[0])
}


def sdk_parse(event: Event) -> SDKModel:
    """Validate a server event with the SDK's own model for its type: every required field must
    be present and typed right, and no field may be unknown to the SDK (the session's ``id`` and
    ``object`` excepted, which the SDK reads as extras)."""
    model = _SDK_EVENTS[event["type"]].model_validate(event)
    unknown = list(_unknown_fields(model, event["type"]))
    assert not unknown, f"fields unknown to the SDK: {unknown}"
    return model


def _unknown_fields(model: SDKModel, path: str) -> list[str]:
    allowed = {"id", "object"} if isinstance(model, RealtimeSessionCreateRequest) else set[str]()
    found = [f"{path}.{key}" for key in model.model_extra or {} if key not in allowed]
    for name in type(model).model_fields:
        value = getattr(model, name)
        children = value if isinstance(value, list) else [value]
        for index, item in enumerate(children):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            if isinstance(item, SDKModel):
                suffix = f"[{index}]" if isinstance(value, list) else ""
                found += _unknown_fields(item, f"{path}.{name}{suffix}")
    return found


@dataclass
class Mocks:
    llm: EchoLLM
    asr: ScriptedASR
    tts: SineTTS


class ManualClock:
    """The session clock in virtual time: it moves only when a test advances it, and a sleep
    on it ends once it has been advanced far enough. Turn timelines, pacing (D6) and the D5
    fallback read it; asyncio timeouts still run on the loop."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    @property
    def sleeping(self) -> int:
        """How many tasks are waiting for virtual time to pass (the pacer at its cap)."""
        return sum(1 for _, future in self._sleepers if not future.done())

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
        for target, future in self._sleepers:
            if target <= self.t and not future.done():
                future.set_result(None)
        self._sleepers = [
            (target, future) for target, future in self._sleepers if not future.done()
        ]

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.t + seconds, future))
        await future


ECHO_OFF: dict[str, dict[str, object]] = {
    "aec": {"enabled": False},
    "barge_in": {
        "echo_guard_ms": 0,
        "min_speech_ms_while_playing": None,
        "echo_turn_suppression": False,
    },
}
"""The echo canceller and the barge-in rules off: scripted detectors and audio sent faster
than real time leave no client playout to model (the M4 echo tests turn them on)."""


def with_echo_off(config: Mapping[str, object] | None) -> dict[str, object]:
    """``config`` over ``ECHO_OFF``, section by section."""
    merged: dict[str, object] = dict(config or {})
    for section, defaults in ECHO_OFF.items():
        given = merged.get(section)
        overrides = dict(given) if isinstance(given, Mapping) else {}  # pyright: ignore[reportUnknownArgumentType]
        merged[section] = {**defaults, **overrides}
    return merged


def build_gateway(
    config: Mapping[str, object] | None = None,
    *,
    transcripts: Sequence[str] = ("你好。",),
    speech: Sequence[tuple[int, int]] = (),
    turn_end: TurnEndDetector | None = None,
    ms_per_char: int = 20,
    chunk_ms: int = 40,
    clock: Clock | None = None,
) -> tuple[Gateway, Mocks]:
    """A gateway over zero-delay mocks; ``speech`` scripts the speech detector of every session.
    The echo canceller and barge-in rules are off unless ``config`` turns them on."""
    mocks = Mocks(
        llm=EchoLLM(),
        asr=ScriptedASR(transcripts),
        tts=SineTTS(ms_per_char=ms_per_char, chunk_ms=chunk_ms),
    )
    gateway = Gateway(
        validate_config(with_echo_off(config)),
        Providers(llm=mocks.llm, asr=mocks.asr, tts=mocks.tts),
        speech_detector=lambda: ScriptedSpeech(speech),
        turn_end_detector=lambda: turn_end,
        clock=clock,
    )
    return gateway, mocks


@asynccontextmanager
async def running(gateway: Gateway, *, wait_ready: bool = True) -> AsyncIterator[str]:
    """Serve ``gateway`` on a free port of 127.0.0.1; yields ``host:port``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = GatewayServer(gateway)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(TIMEOUT_S):
            # uvicorn and the readiness monitor expose flags, not events: poll them.
            while not server.started or (wait_ready and not gateway.readiness.ever_ready):  # noqa: ASYNC110
                await asyncio.sleep(0.005)
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


class Client:
    """A raw Realtime client that records both directions of the conversation, and when each
    event was sent or received (``timeline``, loop time in seconds)."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.trace: list[tuple[str, Event]] = []
        self.timeline: list[tuple[float, str, Event]] = []

    @classmethod
    @asynccontextmanager
    async def connect(
        cls,
        address: str,
        *,
        model: str | None = "test-model",
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Client]:
        query = "" if model is None else f"?model={model}"
        url = f"ws://{address}/v1/realtime{query}"
        async with connect(url, additional_headers=headers, max_size=None, proxy=None) as ws:
            yield cls(ws)

    @property
    def events(self) -> list[Event]:
        return [event for direction, event in self.trace if direction == "server"]

    async def send(self, event: Event) -> None:
        self.trace.append(("client", event))
        self.timeline.append((asyncio.get_running_loop().time(), "client", event))
        await self.ws.send(json.dumps(event))

    async def send_raw(self, frame: str | bytes) -> None:
        """Send a frame as is (malformed JSON, binary data); the trace records its shape."""
        shape = f"<{len(frame)} bytes>" if isinstance(frame, bytes) else frame
        self.trace.append(("client", {"raw": shape}))
        await self.ws.send(frame)

    async def recv(self, wait_s: float = TIMEOUT_S) -> Event:
        """The next server event, checked against the SDK's model for it."""
        async with asyncio.timeout(wait_s):
            event: Event = json.loads(await self.ws.recv())
        self.trace.append(("server", event))
        self.timeline.append((asyncio.get_running_loop().time(), "server", event))
        sdk_parse(event)
        return event

    async def until(self, event_type: str, wait_s: float = TIMEOUT_S) -> list[Event]:
        """Receive events up to and including the first of ``event_type``."""
        received: list[Event] = []
        async with asyncio.timeout(wait_s):
            while True:
                event = await self.recv()
                received.append(event)
                if event["type"] == event_type:
                    return received

    async def quiet(self, seconds: float = 0.1) -> None:
        """Assert that nothing arrives for ``seconds``."""
        try:
            async with asyncio.timeout(seconds):
                event = await self.recv(wait_s=seconds * 2)
        except TimeoutError:
            return
        raise AssertionError(f"unexpected event {event['type']}: {event}")

    async def start(self, session: Event | None = None) -> None:
        """Consume the connect events, then apply ``session`` with a session.update."""
        await self.until("conversation.created")
        if session is not None:
            await self.send({"type": "session.update", "session": {"type": "realtime", **session}})
            await self.until("session.updated")

    async def say(self, text: str, role: str = "user") -> None:
        part = "output_text" if role == "assistant" else "input_text"
        item = {"type": "message", "role": role, "content": [{"type": part, "text": text}]}
        await self.send({"type": "conversation.item.create", "item": item})
        await self.until("conversation.item.done")

    async def append(self, pcm: bytes, chunk_ms: int = 100, *, bytes_per_ms: int = 48) -> None:
        """``pcm`` in appends of ``chunk_ms``: 24 kHz PCM by default, ``bytes_per_ms=8`` for
        G.711."""
        step = bytes_per_ms * chunk_ms
        for start in range(0, len(pcm), step):
            audio = base64.b64encode(pcm[start : start + step]).decode()
            await self.send({"type": "input_audio_buffer.append", "audio": audio})


async def http_get(address: str, path: str) -> tuple[int, Event]:
    """A minimal HTTP/1.1 GET returning the status and the JSON body."""
    host, port = address.split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(f"GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n".encode())
    async with asyncio.timeout(TIMEOUT_S):
        response = await reader.read()
    writer.close()
    head, _, body = response.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body)


def g711_tone(ms: int, law: Law = "ulaw", *, frequency: float = 220.0) -> bytes:
    """``ms`` of an 8 kHz G.711 tone (client audio in an ``audio/pcmu`` or ``pcma`` session)."""
    samples = 8 * ms
    values = np.rint(8000 * np.sin(2 * np.pi * frequency * np.arange(samples) / 8000))
    return g711.encode_samples(values.astype(np.int16), law)


G711_SESSION: dict[str, object] = {
    "audio": {
        "input": {"format": {"type": "audio/pcmu"}},
        "output": {"format": {"type": "audio/pcmu"}},
    }
}
"""A session in G.711 μ-law both ways, as AICC's G.711 path asks for it (no rate)."""


def tone(ms: int, *, frequency: float = 220.0) -> bytes:
    """``ms`` of a 24 kHz PCM16 tone (client audio; the scripted detector ignores its content)."""
    samples = WIRE_RATE * ms // 1000
    values = (int(8000 * math.sin(2 * math.pi * frequency * n / WIRE_RATE)) for n in range(samples))
    return b"".join(value.to_bytes(2, "little", signed=True) for value in values)


_ID = re.compile(r"\b(?:sess|conv|item|resp|call)_[a-z0-9]{16}\b")


def normalize(trace: Sequence[tuple[str, Event]]) -> list[list[object]]:
    """Golden form: ids become ``<item:1>``-style placeholders numbered by first appearance,
    base64 audio becomes its format and byte count (``<pcm:4800>``, ``<pcmu:160>``), and server
    event ids must count up from where they left off."""
    names: dict[str, str] = {}
    counts: dict[str, int] = {}
    last_event = 0

    def rename(match: re.Match[str]) -> str:
        value = match.group()
        if value not in names:
            kind = value.split("_", 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
            names[value] = f"<{kind}>" if kind in ("sess", "conv") else f"<{kind}:{counts[kind]}>"
        return names[value]

    def walk(value: object) -> object:
        if isinstance(value, dict):
            return {name: walk(item) for name, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [walk(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, str):
            return _ID.sub(rename, value)  # ids also appear inside error messages
        return value

    normalized: list[list[object]] = []
    input_label = "pcm"  # the session's input format, from its latest session event
    output_labels: dict[str, str] = {}  # each response's output format
    for direction, event in trace:
        kind = event.get("type")
        if direction == "server":
            number = int(event["event_id"].removeprefix("event_"))
            assert number == last_event + 1, f"event ids out of order at {event}"
            last_event = number
            event = {key: item for key, item in event.items() if key != "event_id"}
            if kind in ("session.created", "session.updated"):
                input_label = _label(event["session"]["audio"]["input"]["format"])
            elif kind == "response.created":
                response = event["response"]
                output_labels[response["id"]] = _label(response["audio"]["output"]["format"])
        if kind == "response.output_audio.delta" and direction == "server":
            label = output_labels.get(event["response_id"], "pcm")
            event = {**event, "delta": f"<{label}:{len(base64.b64decode(event['delta']))}>"}
        elif kind == "input_audio_buffer.append" and direction == "client":
            event = {**event, "audio": f"<{input_label}:{len(base64.b64decode(event['audio']))}>"}
        normalized.append([direction, walk(event)])
    return normalized


def _label(audio_format: Event) -> str:
    """``pcm``, ``pcmu`` or ``pcma``: how a golden trace shows audio (its byte count follows)."""
    return str(audio_format["type"]).removeprefix("audio/")

"""The latency and accuracy benchmark (ARCHITECTURE §9, §11; docs/LATENCY.md).

It talks to ``/v1/realtime`` as a telephone client does: ``server_vad``, each utterance replayed at
real-time speed in 20 ms ``input_audio_buffer.append`` frames, then silence, streamed until the
turn is answered (the line never goes quiet). Per utterance it records:

- the client's end to end: from sending the end of the speech (trailing silence excluded) to the
  first ``response.output_audio.delta`` of the response that answered it;
- the server's stage durations, from ``GET /debug/turns?session_id=…`` (the gateway must run with
  ``observability.debug_endpoints: true``; ``--serve`` turns it on);
- the transcript, from ``conversation.item.input_audio_transcription.completed``.

The report (``bench/report.py``) has per-stage and end-to-end P50/P95 and ASR CER/WER/MER, as JSON
and as markdown tables.

    uv run --locked python -m bench.bench --url ws://127.0.0.1:8000/v1/realtime \\
        --input spikes/data/manifest.jsonl --variant tel8k --concurrency 1 --out bench/results/run
    uv run --locked python -m bench.bench --serve config.mock.yaml --limit 4    # no backends

``--codec pcmu|pcma`` runs the session in 8 kHz G.711 both ways (M7) instead of 24 kHz PCM: the
utterances go out as 160-byte frames (the original 8 kHz stream when its law matches), and a
response in any other format stops the run.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import socket
import sys
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx2
import numpy as np
from websockets.asyncio.client import ClientConnection, connect

from bench.dataset import (
    FRAME_MS,
    WIRE_RATE,
    Utterance,
    Variant,
    g711_encode,
    load,
)
from bench.report import TurnResult, markdown, summarize
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.config import load_config
from mlx_realtime.server.app import GatewayServer
from mlx_realtime.server.gateway import Gateway

type Codec = Literal["pcm", "pcmu", "pcma"]
type Event = dict[str, object]

DEFAULT_INPUT = Path("spikes/data/manifest.jsonl")
DEFAULT_URL = "ws://127.0.0.1:8000/v1/realtime"
_SETTLE_S = 0.3  # quiet time after a turn is complete, before the next utterance


@dataclass(frozen=True)
class Options:
    url: str = DEFAULT_URL
    concurrency: int = 1
    turns_per_session: int = 10
    silence_duration_ms: int = 500  # AICC's server_vad setting
    min_silence_ms: int = 1000  # silence streamed after every utterance, at least
    no_turn_after_s: float = 6.0  # silence after which an undetected utterance is given up
    turn_timeout_s: float = 60.0
    codec: Codec = "pcm"
    language: str | None = None
    instructions: str | None = "你是电话客服。请用一句话简短回答。"
    api_key: str | None = None
    model: str | None = None


@dataclass
class _Response:
    id: str
    first_audio_at: float | None = None
    status: str | None = None


@dataclass
class _Utterance:
    """What the reader collected while one utterance played."""

    items: list[str] = field(default_factory=list[str])
    transcripts: dict[str, str] = field(default_factory=dict[str, str])  # item -> text
    responses: list[_Response] = field(default_factory=list[_Response])
    speaking: bool = False  # speech_started without its speech_stopped yet

    @property
    def complete(self) -> bool:
        """At least one turn, every turn transcribed, no speech or response in progress."""
        return (
            bool(self.items)
            and all(item in self.transcripts for item in self.items)
            and not self.speaking
            and all(response.status is not None for response in self.responses)
        )


class _Session:
    """One Realtime session: the sender is the caller's line, the reader collects events."""

    def __init__(self, ws: ClientConnection, options: Options) -> None:
        self.ws = ws
        self.options = options
        self.session_id = ""
        self.current = _Utterance()
        self.changed = asyncio.Event()
        self.failure: str | None = None
        self._responses: dict[str, _Response] = {}

    async def read(self) -> None:
        async for frame in self.ws:
            event = cast(Event, json.loads(frame))
            self._handle(event, asyncio.get_running_loop().time())
            self.changed.set()

    def _handle(self, event: Event, now: float) -> None:
        kind = event["type"]
        current = self.current
        match kind:
            case "session.created":
                self.session_id = str(_field(event, "session", "id"))
            case "input_audio_buffer.speech_started":
                current.speaking = True
            case "input_audio_buffer.speech_stopped":
                current.speaking = False
            case "input_audio_buffer.committed":
                current.items.append(str(event["item_id"]))
            case "conversation.item.input_audio_transcription.completed":
                current.transcripts[str(event["item_id"])] = str(event["transcript"])
            case "conversation.item.input_audio_transcription.failed":
                current.transcripts[str(event["item_id"])] = ""
            case "response.created":
                found = _field(event, "response", "audio", "output", "format", "type")
                if found != f"audio/{self.options.codec}":
                    self.failure = f"a response in {found}, not audio/{self.options.codec}"
                response = _Response(str(_field(event, "response", "id")))
                self._responses[response.id] = response
                current.responses.append(response)
            case "response.output_audio.delta":
                response = self._responses.get(str(event["response_id"]))
                if response is not None and response.first_audio_at is None:
                    response.first_audio_at = now
            case "response.done":
                response = self._responses.get(str(_field(event, "response", "id")))
                if response is not None:
                    response.status = str(_field(event, "response", "status"))
            case "error":
                code = _field(event, "error", "code")
                if code in ("invalid_value", "unknown_parameter", "unsupported_value"):
                    self.failure = f"the gateway refused the session: {event['error']}"
            case _:
                pass

    async def send_audio(self, data: bytes) -> None:
        audio = base64.b64encode(data).decode()
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio}))

    async def configure(self) -> None:
        options = self.options
        transcription: dict[str, object] = {"model": "qwen3-asr"}
        if options.language:
            transcription["language"] = options.language
        audio_format: dict[str, object] = (
            {"type": "audio/pcm", "rate": WIRE_RATE}
            if options.codec == "pcm"
            else {"type": f"audio/{options.codec}"}
        )
        session: dict[str, object] = {
            "type": "realtime",
            "audio": {
                "output": {"format": audio_format},
                "input": {
                    "format": audio_format,
                    "transcription": transcription,
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": options.silence_duration_ms,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
            },
        }
        if options.instructions is not None:
            session["instructions"] = options.instructions
        await self.ws.send(json.dumps({"type": "session.update", "session": session}))


def _field(event: Event, *path: str) -> object:
    value: object = event
    for key in path:
        if not isinstance(value, dict):
            return None
        value = cast(dict[str, object], value).get(key)  # pyright: ignore[reportUnnecessaryCast]
    return value


async def run(utterances: Sequence[Utterance], options: Options) -> list[TurnResult]:
    """Play every utterance, ``concurrency`` sessions at once (utterance ``i`` goes to worker
    ``i mod concurrency``; each worker opens a new session every ``turns_per_session``
    utterances). Results come back in input order."""
    workers = max(options.concurrency, 1)
    results: list[TurnResult | None] = [None] * len(utterances)
    http = httpx2.AsyncClient(base_url=_http_base(options.url), trust_env=False)
    async with http, asyncio.TaskGroup() as tasks:
        for worker in range(workers):
            indices = list(range(worker, len(utterances), workers))
            tasks.create_task(_worker(indices, utterances, results, options, http))
    return [result for result in results if result is not None]


async def _worker(
    indices: Sequence[int],
    utterances: Sequence[Utterance],
    results: list[TurnResult | None],
    options: Options,
    http: httpx2.AsyncClient,
) -> None:
    size = max(options.turns_per_session, 1)
    for first in range(0, len(indices), size):
        batch = indices[first : first + size]
        played = await _play_session([utterances[index] for index in batch], options)
        await _add_server_stages(http, played)
        for index, result in zip(batch, played, strict=True):
            results[index] = result


@contextlib.asynccontextmanager
async def _session(options: Options) -> AsyncGenerator[_Session]:
    url = options.url if options.model is None else f"{options.url}?model={options.model}"
    headers = {"Authorization": f"Bearer {options.api_key}"} if options.api_key else None
    async with connect(url, additional_headers=headers, max_size=None, proxy=None) as ws:
        session = _Session(ws, options)
        reader = asyncio.create_task(session.read())
        try:
            yield session
        finally:
            await ws.close()
            with contextlib.suppress(Exception):
                await reader


async def _play_session(utterances: Sequence[Utterance], options: Options) -> list[TurnResult]:
    results: list[TurnResult] = []
    async with _session(options) as session:
        await session.configure()
        await _until(session, lambda: bool(session.session_id), 10.0)
        for utterance in utterances:
            results.append(await _play(session, utterance))
            if session.failure is not None:
                raise SystemExit(session.failure)
    return results


async def _play(session: _Session, utterance: Utterance) -> TurnResult:
    """Replay one utterance and stream silence until its turn is answered (or given up)."""
    options = session.options
    session.current = collected = _Utterance()
    frame_bytes, audio = _frames(utterance, options.codec)
    silence = bytes(frame_bytes) if options.codec == "pcm" else _silence_g711(options.codec)
    loop = asyncio.get_running_loop()
    start = loop.time()
    frames = (len(audio) + frame_bytes - 1) // frame_bytes
    for index in range(frames):
        await _sleep_until(start + index * FRAME_MS / 1000)
        await session.send_audio(audio[index * frame_bytes : (index + 1) * frame_bytes])
    speech_sent_at = start + max(utterance.speech_end_ms - FRAME_MS, 0) / 1000
    audio_end = start + frames * FRAME_MS / 1000
    outcome: Literal["answered", "no_audio", "no_turn", "timeout"] = "timeout"
    index = frames
    settled_at: float | None = None
    while True:
        now = await _sleep_until(start + index * FRAME_MS / 1000)
        index += 1
        await session.send_audio(silence)
        silent_s = now - audio_end
        if silent_s * 1000 < options.min_silence_ms:
            continue
        if collected.complete:
            settled_at = settled_at or now
            if now - settled_at >= _SETTLE_S:
                outcome = "answered"
                break
        else:
            settled_at = None
        if not collected.items and not collected.speaking and silent_s > options.no_turn_after_s:
            outcome = "no_turn"
            break
        if now - start > options.turn_timeout_s + utterance.duration_ms / 1000:
            break
    result = TurnResult(
        utterance_id=utterance.id,
        set=utterance.set,
        has_number=utterance.has_number,
        text_norm=utterance.text_norm,
        outcome=outcome,
        session_id=session.session_id,
        items=list(collected.items),
        transcript="".join(collected.transcripts.get(item, "") for item in collected.items),
        duration_ms=utterance.duration_ms,
    )
    final = collected.responses[-1] if collected.responses else None
    if final is not None:
        result.response_status = final.status
        if final.first_audio_at is not None:
            result.client_e2e_ms = round((final.first_audio_at - speech_sent_at) * 1000, 1)
    if outcome == "answered" and (final is None or final.first_audio_at is None):
        result.outcome = "no_audio"
    return result


def _frames(utterance: Utterance, codec: Codec) -> tuple[int, bytes]:
    """(bytes per 20 ms frame, the audio in the session's input format)."""
    if codec == "pcm":
        return WIRE_RATE * FRAME_MS // 1000 * 2, utterance.pcm
    law: Literal["ulaw", "alaw"] = "ulaw" if codec == "pcmu" else "alaw"
    if utterance.g711 is not None and utterance.g711[0] == law:
        return 8000 * FRAME_MS // 1000, utterance.g711[1]  # the original stream, untouched
    resampler = Resampler(WIRE_RATE, 8000)
    narrow = resampler.process(utterance.pcm) + resampler.flush()
    return 8000 * FRAME_MS // 1000, g711_encode(np.frombuffer(narrow, dtype="<i2"), law)


def _silence_g711(codec: Codec) -> bytes:
    law: Literal["ulaw", "alaw"] = "ulaw" if codec == "pcmu" else "alaw"
    return g711_encode(np.zeros(8000 * FRAME_MS // 1000, dtype=np.int16), law)


async def _sleep_until(at: float) -> float:
    loop = asyncio.get_running_loop()
    delay = at - loop.time()
    if delay > 0:
        await asyncio.sleep(delay)
    return loop.time()


async def _until(session: _Session, done: Callable[[], bool], timeout_s: float) -> None:
    async with asyncio.timeout(timeout_s):
        while not done():
            session.changed.clear()
            await session.changed.wait()


async def _add_server_stages(http: httpx2.AsyncClient, results: Sequence[TurnResult]) -> None:
    """Fill each result's server stages from ``/debug/turns`` (the answered turn's record)."""
    if not results:
        return
    session_id = results[0].session_id
    try:
        response = await http.get("/debug/turns", params={"session_id": session_id})
    except httpx2.HTTPError as error:
        print(f"warning: /debug/turns failed: {error}", file=sys.stderr)
        return
    if response.status_code != 200:
        print(
            f"warning: /debug/turns answered {response.status_code}; is "
            "observability.debug_endpoints on? Server stages are left out.",
            file=sys.stderr,
        )
        return
    turns = cast(list[dict[str, object]], response.json()["turns"])
    by_item = {str(turn["item_id"]): turn for turn in turns if turn.get("response_id")}
    for result in results:
        for item in reversed(result.items):
            turn = by_item.get(item)
            if turn is None:
                continue
            stages = cast(dict[str, float], turn["stages_ms"])
            result.server_stages_ms = {name: float(value) for name, value in stages.items()}
            e2e = turn.get("e2e_ms")
            result.server_e2e_ms = float(e2e) if isinstance(e2e, int | float) else None
            break


def _http_base(url: str) -> str:
    parts = urlsplit(url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, "", "", ""))


@contextlib.asynccontextmanager
async def serve(config_path: Path) -> AsyncGenerator[str]:
    """A gateway of its own from ``config_path``, on a free port with ``/debug/turns`` on; yields
    its ``/v1/realtime`` URL once every backend is warm."""
    environ = {**os.environ, "MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS": "true"}
    gateway = Gateway(load_config(config_path, environ))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = GatewayServer(gateway)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(600):
            while not server.started or not gateway.readiness.ever_ready:  # noqa: ASYNC110
                await asyncio.sleep(0.02)
        yield f"ws://127.0.0.1:{port}/v1/realtime"
    finally:
        server.should_exit = True
        await task
        sock.close()


async def benchmark(
    utterances: Sequence[Utterance], options: Options, *, serve_config: Path | None = None
) -> dict[str, object]:
    """Run the benchmark (against ``options.url``, or a gateway of its own) and summarize it."""
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if serve_config is not None:
        async with serve(serve_config) as url:
            options = replace(options, url=url)
            turns = await run(utterances, options)
    else:
        turns = await run(utterances, options)
    meta: dict[str, object] = {
        "started": started,
        "target": str(serve_config) if serve_config is not None else options.url,
        "concurrency": options.concurrency,
        "codec": options.codec,
        "silence_duration_ms": options.silence_duration_ms,
    }
    return summarize(turns, meta=meta)


def write(report: dict[str, object], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "report.json", out / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench", description="Latency and ASR accuracy over /v1/realtime (docs/LATENCY.md)."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--url", default=DEFAULT_URL, help="the gateway's /v1/realtime URL")
    target.add_argument("--serve", type=Path, help="run a gateway of its own from this config")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="a manifest (.jsonl) or a WAV directory"
    )
    parser.add_argument("--variant", choices=("wide", "tel16k", "tel8k"), default="tel8k")
    parser.add_argument("--sets", default="", help="comma-separated: zh,en,mix (default: all)")
    parser.add_argument("--ids", help="a regular expression the utterance id must match")
    parser.add_argument("--numbers", choices=("include", "exclude", "only"), default="include")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1, help="parallel sessions")
    parser.add_argument("--turns-per-session", type=int, default=10)
    parser.add_argument("--silence-duration-ms", type=int, default=500)
    parser.add_argument("--min-silence-ms", type=int, default=1000)
    parser.add_argument("--turn-timeout-s", type=float, default=60.0)
    parser.add_argument("--codec", choices=("pcm", "pcmu", "pcma"), default="pcm")
    parser.add_argument("--language", help="transcription language (default: detected)")
    parser.add_argument("--instructions", default=Options.instructions)
    parser.add_argument("--api-key", default=os.environ.get("MLXRT_API_KEY"))
    parser.add_argument("--out", type=Path, help="write report.json and report.md here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    variant = cast(Variant, arguments.variant)
    utterances = load(
        arguments.input,
        variant=variant,
        sets=[name for name in str(arguments.sets).split(",") if name],
        ids=arguments.ids,
        numbers=arguments.numbers,
        limit=arguments.limit,
    )
    if not utterances:
        raise SystemExit(f"no utterances selected from {arguments.input}")
    options = Options(
        url=arguments.url,
        concurrency=arguments.concurrency,
        turns_per_session=arguments.turns_per_session,
        silence_duration_ms=arguments.silence_duration_ms,
        min_silence_ms=arguments.min_silence_ms,
        turn_timeout_s=arguments.turn_timeout_s,
        codec=arguments.codec,
        language=arguments.language,
        instructions=arguments.instructions or None,
        api_key=arguments.api_key,
    )
    print(
        f"{len(utterances)} utterances ({variant}), {options.concurrency} session(s)",
        file=sys.stderr,
    )
    report = asyncio.run(benchmark(utterances, options, serve_config=arguments.serve))
    report["meta"] = {**_meta(report), "input": str(arguments.input), "variant": variant}
    if arguments.out is not None:
        paths = write(report, arguments.out)
        print(f"wrote {paths[0]} and {paths[1]}", file=sys.stderr)
    print(markdown(report))


def _meta(report: dict[str, object]) -> dict[str, object]:
    meta = report["meta"]
    assert isinstance(meta, dict)
    return cast(dict[str, object], meta)


if __name__ == "__main__":
    main()

"""The official openai SDK against the gateway, with only ``base_url`` changed (mocks in M1)."""

from __future__ import annotations

import asyncio
import base64
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import RealtimeServerEvent, RealtimeSessionCreateRequest

from tests.support import Mocks, build_gateway, http_get, running

pytestmark = pytest.mark.e2e

REPO = Path(__file__).parents[2]
GA_SESSION = {
    "type": "realtime",
    "instructions": "你是一个客服助手。",
    "output_modalities": ["audio"],
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "qwen3-asr", "language": "zh"},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"},
    },
}
QUESTION = "你好，我想查询一下账户余额。谢谢！"


@asynccontextmanager
async def realtime(address: str) -> AsyncIterator[AsyncRealtimeConnection]:
    client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="sk-local")
    async with client.realtime.connect(model="gpt-realtime") as connection:
        yield connection


async def until(connection: AsyncRealtimeConnection, event_type: str) -> list[RealtimeServerEvent]:
    events: list[RealtimeServerEvent] = []
    async with asyncio.timeout(5):
        async for event in connection:
            events.append(event)
            if event.type == event_type:
                return events
    raise AssertionError(f"connection closed before {event_type}")


async def ask(connection: AsyncRealtimeConnection, text: str) -> None:
    await connection.conversation.item.create(
        item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    )


@asynccontextmanager
async def gateway_with_mocks() -> AsyncIterator[tuple[str, Mocks]]:
    gateway, mocks = build_gateway()
    async with running(gateway) as address:
        yield address, mocks


async def test_spoken_reply_to_a_chinese_message() -> None:
    async with gateway_with_mocks() as (address, mocks), realtime(address) as connection:
        created = (await until(connection, "session.created"))[-1]
        assert created.type == "session.created"
        assert getattr(created.session, "id", "").startswith("sess_")
        await connection.session.update(session=GA_SESSION)  # pyright: ignore[reportArgumentType]
        updated = (await until(connection, "session.updated"))[-1]
        assert updated.type == "session.updated"
        assert isinstance(updated.session, RealtimeSessionCreateRequest)
        assert updated.session.audio is not None
        assert updated.session.audio.output is not None
        assert updated.session.audio.output.voice == "marin"
        await ask(connection, QUESTION)
        await connection.response.create()
        events = await until(connection, "response.done")
    audio = b"".join(
        base64.b64decode(e.delta) for e in events if e.type == "response.output_audio.delta"
    )
    transcript = "".join(
        e.delta for e in events if e.type == "response.output_audio_transcript.delta"
    )
    done = events[-1]
    assert done.type == "response.done"
    assert done.response.status == "completed"
    assert transcript == QUESTION  # the echo LLM repeats the user's words
    assert len(audio) == sum(mocks.tts.samples_for(r.text) for r in mocks.tts.requests) * 2 > 0
    assert len(audio) % 2 == 0


async def test_text_turn() -> None:
    async with gateway_with_mocks() as (address, mocks), realtime(address) as connection:
        await connection.session.update(session={"type": "realtime", "output_modalities": ["text"]})
        await until(connection, "session.updated")
        await ask(connection, "Hello there, how are you?")
        await connection.response.create()
        events = await until(connection, "response.done")
    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    done = [e for e in events if e.type == "response.output_text.done"]
    assert text == done[0].text == "Hello there, how are you?"
    assert not [e for e in events if e.type.startswith("response.output_audio")]
    assert mocks.tts.requests == []


async def test_cancel_mid_audio() -> None:
    async with gateway_with_mocks() as (address, mocks), realtime(address) as connection:
        await connection.session.update(session=GA_SESSION)  # pyright: ignore[reportArgumentType]
        await until(connection, "session.updated")
        await ask(connection, QUESTION)
        mocks.tts.next_chunk.hold()
        await connection.response.create()
        await until(connection, "response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        await connection.response.cancel()
        events = await until(connection, "response.done")
        mocks.tts.next_chunk.release()
    done = events[-1]
    assert done.type == "response.done"
    assert done.response.status == "cancelled"
    assert done.response.status_details is not None
    assert done.response.status_details.reason == "client_cancelled"
    assert not [e for e in events if e.type == "response.output_audio.delta"]


async def test_dev_entry_point_with_the_mock_config() -> None:
    """``mlxrt dev`` as ``make dev`` runs it, with config.mock.yaml (no model backends: config.yaml
    would start the real ones), on a free port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environ = {**os.environ, "MLXRT__GATEWAY__PORT": str(port)}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "mlx_realtime",
        "--config",
        str(REPO / "config.mock.yaml"),
        "dev",
        cwd=REPO,
        env=environ,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    address = f"127.0.0.1:{port}"
    try:
        async with asyncio.timeout(15):
            while True:
                try:
                    status, body = await http_get(address, "/readyz")
                except OSError:
                    await asyncio.sleep(0.1)
                    continue
                if status == 200 and body["ready"]:
                    break
                await asyncio.sleep(0.1)
        async with realtime(address) as connection:
            await connection.session.update(session=GA_SESSION)  # pyright: ignore[reportArgumentType]
            await ask(connection, QUESTION)
            await connection.response.create()
            events = await until(connection, "response.done")
        assert events[-1].type == "response.done"
        assert events[-1].response.status == "completed"
        assert any(e.type == "response.output_audio.delta" for e in events)
    finally:
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(10):
            # uvicorn shuts down gracefully, then re-raises the signal it caught.
            assert await process.wait() in (0, -signal.SIGINT)

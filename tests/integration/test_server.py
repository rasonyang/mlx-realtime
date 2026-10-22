"""Transport and session lifecycle: health, readiness, auth, admission, limits, shutdown."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from mlx_realtime.server.app import GatewayServer
from mlx_realtime.session.actor import SessionActor
from tests.support import Client, build_gateway, http_get, running, tone

TEXT_MANUAL = {"output_modalities": ["text"], "audio": {"input": {"turn_detection": None}}}


async def test_health_and_readiness() -> None:
    gateway, mocks = build_gateway()
    mocks.llm.warming.hold()
    async with running(gateway, wait_ready=False) as address:
        assert await http_get(address, "/healthz") == (200, {"status": "ok"})
        status, body = await http_get(address, "/readyz")
        assert (status, body["ready"], body["backends"]["llm"]["warm"]) == (503, False, False)
        async with Client.connect(address) as client:
            error = await client.recv()
            await client.ws.wait_closed()
        assert (error["type"], error["error"]["code"]) == ("error", "not_ready")
        assert error["error"]["type"] == "server_error"
        assert client.ws.close_code == 1013
        mocks.llm.warming.release()
        async with asyncio.timeout(2):
            while not gateway.readiness.ever_ready:  # noqa: ASYNC110 (a flag, not an event)
                await asyncio.sleep(0.005)
        status, body = await http_get(address, "/readyz")
        assert (status, body["ready"]) == (200, True)
        assert set(body["backends"]) == {"llm", "asr", "tts"}
        async with Client.connect(address) as client:
            assert (await client.recv())["type"] == "session.created"


async def test_bearer_auth() -> None:
    gateway, _ = build_gateway({"gateway": {"api_key": "sk-local"}})
    async with running(gateway) as address:
        for headers in (None, {"Authorization": "Bearer wrong"}, {"Authorization": "sk-local"}):
            with pytest.raises(InvalidStatus) as rejected:
                async with Client.connect(address, headers=headers):
                    pass
            assert rejected.value.response.status_code == 401
        async with Client.connect(address, headers={"Authorization": "Bearer sk-local"}) as client:
            assert (await client.recv())["type"] == "session.created"


async def test_model_query_is_echoed_and_defaults_otherwise() -> None:
    gateway, _ = build_gateway({"session_defaults": {"model": "local-default"}})
    async with running(gateway) as address:
        async with Client.connect(address, model="gpt-realtime") as client:
            assert (await client.recv())["session"]["model"] == "gpt-realtime"
        async with Client.connect(address, model=None) as client:
            assert (await client.recv())["session"]["model"] == "local-default"


async def test_session_slots_are_released() -> None:
    gateway, _ = build_gateway({"gateway": {"max_sessions": 1}})
    async with running(gateway) as address:
        for _ in range(3):
            async with Client.connect(address) as client:
                assert (await client.recv())["type"] == "session.created"
            async with asyncio.timeout(2):
                while gateway.admission.active:  # noqa: ASYNC110 (a counter, not an event)
                    await asyncio.sleep(0.005)


@pytest.mark.parametrize(
    ("limits", "audio_ms", "code"),
    [
        ({"inbox_max_ms": 20}, 100, "input_queue_overflow"),
        ({"input_audio_buffer_max_ms": 500}, 700, "input_audio_buffer_overflow"),
    ],
)
async def test_input_limits_are_fatal(limits: dict[str, int], audio_ms: int, code: str) -> None:
    gateway, _ = build_gateway({"limits": limits})
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TEXT_MANUAL)
        await client.append(tone(audio_ms), chunk_ms=audio_ms)
        error = (await client.until("error"))[-1]["error"]
        await client.ws.wait_closed()
    assert (error["type"], error["code"]) == ("server_error", code)
    assert client.ws.close_code == 1011


async def test_oversized_frames_close_the_connection() -> None:
    gateway, _ = build_gateway({"limits": {"client_max_message_bytes": 1024}})
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await client.send_raw("x" * 2048)
        await client.ws.wait_closed()
    assert client.ws.close_code == 1009


async def test_a_client_that_stops_reading_is_disconnected() -> None:
    gateway, _ = build_gateway(
        {
            "limits": {"client_write_timeout_ms": 200, "max_response_audio_ms": 600_000},
            "text": {"segmenter": {"max_chars": 400}},
        },
        ms_per_char=1000,
        chunk_ms=1000,
    )
    async with running(gateway) as address:
        url = f"ws://{address}/v1/realtime"
        # No compression, as the official SDK: deflated sine audio would fit in socket buffers.
        async with connect(
            url, max_queue=1, compression=None, max_size=None, proxy=None, close_timeout=0.2
        ) as ws:
            content = [{"type": "input_text", "text": "长" * 300}]
            item = {"type": "message", "role": "user", "content": content}
            await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
            await ws.send(json.dumps({"type": "response.create"}))  # 300 s of audio, never read
            async with asyncio.timeout(10):
                while gateway.admission.active:  # noqa: ASYNC110 (a counter, not an event)
                    await asyncio.sleep(0.02)


async def test_maximum_session_duration() -> None:
    gateway, _ = build_gateway({"gateway": {"max_session_s": 1}})
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await client.ws.wait_closed()
    assert client.ws.close_code == 1000


async def test_shutdown_closes_sessions_with_going_away() -> None:
    gateway, _ = build_gateway()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{sock.getsockname()[1]}"
    server = GatewayServer(gateway)
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    async with asyncio.timeout(5):
        while not (server.started and gateway.readiness.ever_ready):  # noqa: ASYNC110
            await asyncio.sleep(0.005)
    async with Client.connect(address) as client:
        await client.start()
        server.should_exit = True
        await client.ws.wait_closed()
    await serving
    sock.close()
    assert client.ws.close_code == 1001


async def test_a_crashing_session_leaves_the_others_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crash(self: SessionActor) -> None:
        raise RuntimeError("injected fault")

    monkeypatch.setattr(SessionActor, "_clear_buffer", crash)
    gateway, _ = build_gateway()
    async with (
        running(gateway) as address,
        Client.connect(address) as victim,
        Client.connect(address) as bystander,
    ):
        await victim.start()
        await bystander.start()
        await victim.send({"type": "input_audio_buffer.clear"})
        error = (await victim.until("error"))[-1]["error"]
        await victim.ws.wait_closed()
        assert (error["type"], error["code"]) == ("server_error", "internal_error")
        assert victim.ws.close_code == 1011
        await bystander.send(
            {"type": "session.update", "session": {"type": "realtime", "instructions": "x"}}
        )
        assert (await bystander.until("session.updated"))[-1]["session"]["instructions"] == "x"

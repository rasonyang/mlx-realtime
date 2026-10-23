"""``/metrics`` and ``/debug/turns`` on a running gateway (ARCHITECTURE §9, M6): a spoken turn feeds
the stage histograms and the windowed gauges, the counters follow the session, and the debug
endpoint lists the turn's marks only when it is switched on."""

from __future__ import annotations

import asyncio
import json

import pytest
from prometheus_client.parser import text_string_to_metric_families

from mlx_realtime.providers.base import ProviderError
from mlx_realtime.session.timeline import STAGES
from tests.support import TIMEOUT_S, Client, build_gateway, running, tone

type Samples = dict[tuple[str, tuple[tuple[str, str], ...]], float]


async def get(address: str, path: str) -> tuple[int, str, bytes]:
    """A minimal HTTP/1.1 GET: the status, the content type and the body."""
    host, port = address.split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(f"GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n".encode())
    async with asyncio.timeout(TIMEOUT_S):
        response = await reader.read()
    writer.close()
    head, _, body = response.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines[1:])
    return int(lines[0].split()[1]), headers.get("content-type", ""), body


async def scrape(address: str) -> Samples:
    status, content_type, body = await get(address, "/metrics")
    assert status == 200
    assert content_type.startswith("text/plain; version=0.0.4")
    found: Samples = {}
    for family in text_string_to_metric_families(body.decode()):
        for sample in family.samples:
            found[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return found


async def spoken_turn(client: Client) -> tuple[str, str]:
    """A server_vad turn (the scripted detector hears 200–1200 ms); the item and response ids."""
    await client.start()
    await client.append(tone(1800))
    events = await client.until("response.done")
    item = next(e["item_id"] for e in events if e["type"] == "input_audio_buffer.committed")
    return item, events[-1]["response"]["id"]


async def test_a_spoken_turn_feeds_every_stage_and_the_counters() -> None:
    gateway, _ = build_gateway(speech=[(200, 1200)])
    async with running(gateway) as address, Client.connect(address) as client:
        await spoken_turn(client)
        found = await scrape(address)
    for stage in STAGES:
        assert found[("mlxrt_stage_seconds_count", (("stage", stage),))] == 1, stage
        assert ("mlxrt_stage_p50_seconds", (("stage", stage),)) in found
    assert found[("mlxrt_e2e_seconds_count", ())] == 1
    assert found[("mlxrt_e2e_p95_seconds", ())] > 0
    assert found[("mlxrt_responses_total", (("status", "completed"),))] == 1
    assert found[("mlxrt_sessions_total", ())] == 1
    assert found[("mlxrt_active_sessions", ())] == 1
    assert found[("mlxrt_tts_queue_depth", ())] == 0
    for backend in ("llm", "asr", "tts"):
        assert found[("mlxrt_backend_ready", (("backend", backend),))] == 1


async def test_provider_errors_and_refusals_are_counted() -> None:
    gateway, mocks = build_gateway({"gateway": {"max_sessions": 1}}, speech=[(200, 1200)])
    mocks.asr.result.fail(ProviderError("timeout", "asr deadline"))
    async with running(gateway) as address:
        async with Client.connect(address) as client:
            await client.start()
            await client.append(tone(1800))
            await client.until("conversation.item.input_audio_transcription.failed")
            async with Client.connect(address) as refused:
                assert (await refused.recv())["error"]["code"] == "session_limit_reached"
        found = await scrape(address)
    assert found[("mlxrt_provider_errors_total", (("backend", "asr"), ("kind", "timeout")))] == 1
    assert found[("mlxrt_sessions_refused_total", (("reason", "session_limit_reached"),))] == 1
    assert found[("mlxrt_responses_total", (("status", "failed"),))] == 1


async def test_failed_responses_and_transcripts_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider failure ends a response (or a transcript) with an error for the client and a
    count on /metrics; the log says so too, with the stage and the kind."""
    gateway, mocks = build_gateway(speech=[(200, 1200), (2200, 3200)])
    mocks.tts.queue.fail(ProviderError("overloaded", "too many requests in flight"))
    with caplog.at_level("WARNING", logger="mlx_realtime.session.actor"):
        async with running(gateway) as address, Client.connect(address) as client:
            await client.start()
            await client.append(tone(1800))
            await client.until("response.done")
            mocks.asr.result.fail(ProviderError("timeout", "asr deadline"))
            await client.append(tone(1800))
            await client.until("conversation.item.input_audio_transcription.failed")
    failed = {
        (record.getMessage(), getattr(record, "kind", None), getattr(record, "stage", None))
        for record in caplog.records
    }
    assert ("response failed", "overloaded", "tts") in failed
    assert ("transcription failed", "timeout", None) in failed


async def test_debug_turns_lists_each_turn_of_a_session() -> None:
    gateway, _ = build_gateway({"observability": {"debug_endpoints": True}}, speech=[(200, 1200)])
    async with running(gateway) as address:
        async with Client.connect(address) as client:
            item, response = await spoken_turn(client)
            session_id = client.events[0]["session"]["id"]
        status, _, body = await get(address, f"/debug/turns?session_id={session_id}")
        assert status == 200
        (turn,) = json.loads(body)["turns"]
        assert (turn["item_id"], turn["response_id"], turn["status"]) == (
            item,
            response,
            "completed",
        )
        assert set(turn["stages_ms"]) == set(STAGES)
        marks = turn["marks_ms"]
        e2e = marks["first_audio_sent"] - marks["speech_end"]
        assert turn["e2e_ms"] == pytest.approx(e2e, abs=1)  # marks are whole milliseconds
        assert turn["values"]["tts_model"] == "sine"
        status, _, body = await get(address, "/debug/turns")
        assert (status, json.loads(body)) == (200, {"sessions": [session_id]})
        status, _, _ = await get(address, "/debug/turns?session_id=sess_unknown")
        assert status == 404


async def test_debug_turns_is_off_by_default() -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address:
        status, _, _ = await get(address, "/debug/turns")
    assert status == 404

"""The cancellation matrix of ARCHITECTURE §6.5: every phase of a response, every trigger.

Each case holds a mock at an exact await point, triggers the end of the response, and checks
that: no delta follows a terminal event and every opened stream closes exactly once (the causal
invariants); the provider streams close within a bound without their gates being released, so
the cancellation, not a release, ended them; and the next request sees a consistent history.
"""

from __future__ import annotations

import asyncio

import pytest

from mlx_realtime.providers.base import LLMDone, ProviderError, ProviderTimeout, TextDelta, ToolCall
from mlx_realtime.providers.mock import AwaitPoint
from tests.integration.invariants import assert_causal
from tests.support import Client, Event, Mocks, build_gateway, running, tone

TEXT = "你好。我们开始吧，今天讲一个很长的故事。"
CLOSE_BOUND_S = 1.0

PHASES = [
    "awaiting_transcript",
    "before_first_token",
    "mid_text",
    "mid_tool_call",
    "tts_first_chunk",
    "mid_audio",
    "draining",
]
TRIGGERS = ["cancel", "barge_in", "disconnect", "provider_error", "provider_timeout"]
EXPECTED = {
    "cancel": ("cancelled", "client_cancelled", None),
    "barge_in": ("cancelled", "turn_detected", None),
    "provider_error": ("failed", None, "backend_unavailable"),
    "provider_timeout": ("failed", None, "backend_timeout"),
}


async def reach(phase: str, client: Client, mocks: Mocks) -> AwaitPoint:
    """Start a response and stop it at ``phase``; return the await point holding it."""
    if phase == "awaiting_transcript":
        mocks.asr.result.hold()
        await client.append(tone(200))
        await client.send({"type": "input_audio_buffer.commit"})
        await client.send({"type": "response.create"})
        await mocks.asr.result.held()
        return mocks.asr.result
    await client.say(TEXT)
    match phase:
        case "before_first_token":
            point = mocks.llm.first_token
            point.hold()
        case "mid_text":  # "你好。" passes and is spoken; the LLM stops before the next token
            point = mocks.llm.next_token
            point.hold(after=2)
        case "mid_tool_call":  # text spoken, the call not yet complete
            mocks.llm.script(
                TextDelta("好的，"),
                TextDelta("我查一下。"),
                ToolCall("p1", "lookup", "{}"),
                LLMDone("tool_calls", None),
            )
            point = mocks.llm.next_token
            point.hold(after=1)
        case "tts_first_chunk":
            point = mocks.tts.first_chunk
            point.hold()
        case "mid_audio":  # the LLM still generating, the TTS inside the first segment
            mocks.llm.next_token.hold(after=2)
            point = mocks.tts.next_chunk
            point.hold()
        case _:  # draining: the LLM is done, the TTS inside the second segment
            point = mocks.tts.next_chunk
            point.hold(after=1)
    await client.send({"type": "response.create"})
    await point.held()
    return point


def release_all(mocks: Mocks) -> None:
    for point in (
        mocks.llm.first_token,
        mocks.llm.next_token,
        mocks.tts.first_chunk,
        mocks.tts.next_chunk,
        mocks.asr.result,
    ):
        point.release()


async def closed_without_release(mocks: Mocks) -> None:
    async with asyncio.timeout(CLOSE_BOUND_S):
        await mocks.llm.streams.idle()
        await mocks.tts.streams.idle()


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("phase", PHASES)
async def test_cancellation(phase: str, trigger: str) -> None:
    gateway, mocks = build_gateway(speech=[(400, 1200)], transcripts=["你好。"])
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"tools": [{"type": "function", "name": "lookup"}]})
        point = await reach(phase, client, mocks)
        if trigger == "disconnect":
            await client.ws.close()
            async with asyncio.timeout(CLOSE_BOUND_S):
                await mocks.asr.calls.idle()
                while gateway.admission.active:  # noqa: ASYNC110 (a counter, not an event)
                    await asyncio.sleep(0.005)
            await closed_without_release(mocks)
            assert_causal(client.events)
            return
        match trigger:
            case "cancel":
                await client.send({"type": "response.cancel"})
            case "barge_in":
                await client.append(tone(900))  # speech from 400 ms, confirmed at ~550 ms
            case "provider_error":
                point.fail(ProviderError("unavailable", "backend down"))
                point.release()
            case _:
                point.fail(ProviderTimeout("idle"))
                point.release()
        events = await client.until("response.done")
        await closed_without_release(mocks)
        release_all(mocks)
        async with asyncio.timeout(CLOSE_BOUND_S):
            await mocks.asr.calls.idle()
        check_ending(events, trigger)
        cancelled = events[-1]["response"]
        await check_history(client, mocks, cancelled)
    assert_causal(client.events)


def check_ending(events: list[Event], trigger: str) -> None:
    status, reason, code = EXPECTED[trigger]
    response = events[-1]["response"]
    assert response["status"] == status
    details = response["status_details"]
    assert details["type"] == status
    assert details.get("reason") == reason
    types = [event["type"] for event in events]
    if code is not None:
        errors = [event["error"] for event in events if event["type"] == "error"]
        assert [error["code"] for error in errors] == [code]
        assert details["error"]["code"] == code
    else:
        assert "error" not in types
    if trigger == "barge_in":
        assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
    for output in response["output"]:
        assert output["type"] == "message", "a call cancelled before completion is never delivered"
        assert output["status"] == "incomplete"


async def check_history(client: Client, mocks: Mocks, cancelled: Event) -> None:
    """The next LLM request sees what the client heard of the ended response, and no more."""
    await client.say("继续。")
    await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
    await client.until("response.done")
    assistant = [
        message.content
        for message in mocks.llm.requests[-1].messages
        if message.role == "assistant"
    ]
    heard = [part.get("transcript") for output in cancelled["output"] for part in output["content"]]
    assert assistant == [text for text in heard if text]

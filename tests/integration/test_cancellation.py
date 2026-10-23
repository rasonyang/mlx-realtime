"""The cancellation matrix of ARCHITECTURE §6.5: every phase of a response, every trigger.

Each case holds a mock at an exact await point (or the pacer at its lead cap), triggers the end
of the response, and checks that:

- no delta follows a terminal event, and every opened stream closes exactly once (the causal
  invariants), with the terminal events in the GA order;
- the provider streams are observed closed within a bound without their gates being released,
  so the cancellation, not a release, ended them; every stream a mock opened was closed once;
- audio the pacer held back is never sent;
- a barge-in emits ``speech_started`` before ``response.done{cancelled}``, with nothing but the
  terminal events between them (one synchronous handler, AICC's ordering rule);
- the next request sees a consistent history: what the client can have heard of the ended
  response (D5 for a cancel without a truncate), and nothing more.

The session clock is virtual (``ManualClock``): pacing waits on it, and the D5 cut reads it.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from mlx_realtime.providers.base import LLMDone, ProviderError, ProviderTimeout, TextDelta, ToolCall
from mlx_realtime.providers.mock import AwaitPoint
from mlx_realtime.server.gateway import Gateway
from tests.integration.invariants import assert_causal
from tests.support import Client, Event, ManualClock, Mocks, build_gateway, running, tone

TEXT = "你好。我们开始吧，今天讲一个很长的故事。"
DIRECTION = "请一字不差地说出下面这句话，不要添加任何其它内容：\n"
LINE = DIRECTION + "您好，欢迎致电诺瓦网络。请问有什么可以帮您？"
CLOSE_BOUND_S = 1.0
PLAYED_S = 60.0  # how long the clock runs before a trigger: the client heard all it was sent

PHASES = [
    "awaiting_transcript",
    "before_first_token",
    "mid_text",
    "mid_tool_call",
    "tts_queue_wait",
    "tts_first_chunk",
    "mid_audio",
    "paced",
    "draining",
    "verbatim_line",
]
TRIGGERS = ["cancel", "barge_in", "disconnect", "provider_error", "provider_timeout"]
EXPECTED = {
    "cancel": ("cancelled", "client_cancelled", None),
    "barge_in": ("cancelled", "turn_detected", None),
    "provider_error": ("failed", None, "backend_unavailable"),
    "provider_timeout": ("failed", None, "backend_timeout"),
}
TERMINAL = {
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "conversation.item.done",
}


def build(phase: str, clock: ManualClock) -> tuple[Gateway, Mocks]:
    if phase == "paced":  # 300 ms of audio per three characters, at most 100 ms ahead
        config = {"output": {"max_lead_ms": 100, "max_delta_ms": 100}}
        return build_gateway(config, speech=[(400, 1200)], ms_per_char=100, clock=clock)
    return build_gateway(speech=[(400, 1200)], transcripts=["你好。"], clock=clock)


async def reach(phase: str, client: Client, mocks: Mocks, clock: ManualClock) -> AwaitPoint:
    """Start a response and stop it at ``phase``; return the await point holding it."""
    if phase == "awaiting_transcript":
        mocks.asr.result.hold()
        await client.append(tone(200))
        await client.send({"type": "input_audio_buffer.commit"})
        await client.send({"type": "response.create"})
        await mocks.asr.result.held()
        return mocks.asr.result
    if phase == "verbatim_line":  # a line, spoken without the LLM, inside its first segment
        point = mocks.tts.next_chunk
        point.hold()
        await client.send({"type": "response.create", "response": {"instructions": LINE}})
        await point.held()
        return point
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
        case "tts_queue_wait":  # the first segment waits for the TTS backend's stream slot
            point = mocks.tts.queue
            point.hold()
        case "tts_first_chunk":
            point = mocks.tts.first_chunk
            point.hold()
        case "mid_audio":  # the LLM still generating, the TTS inside the first segment
            mocks.llm.next_token.hold(after=2)
            point = mocks.tts.next_chunk
            point.hold()
        case "paced":  # the LLM still generating; "你好。" synthesized, mostly held by pacing
            point = mocks.llm.next_token
            point.hold(after=2)
        case _:  # draining: the LLM is done, the TTS inside the second segment
            point = mocks.tts.next_chunk
            point.hold(after=1)
    await client.send({"type": "response.create"})
    await point.held()
    if phase == "paced":
        async with asyncio.timeout(CLOSE_BOUND_S):
            while not clock.sleeping:  # noqa: ASYNC110 (the pacer waits on virtual time)
                await asyncio.sleep(0.005)
    return point


async def settle(client: Client, quiet_s: float = 0.05) -> None:
    """Receive what the server already sent (everything the held pipeline produced)."""
    while True:
        try:
            await client.recv(wait_s=quiet_s)
        except TimeoutError:
            return


def release_all(mocks: Mocks) -> None:
    for point in (
        mocks.llm.first_token,
        mocks.llm.next_token,
        mocks.tts.queue,
        mocks.tts.first_chunk,
        mocks.tts.next_chunk,
        mocks.asr.result,
    ):
        point.release()


async def closed_without_release(mocks: Mocks) -> None:
    async with asyncio.timeout(CLOSE_BOUND_S):
        await mocks.llm.streams.idle()
        await mocks.tts.streams.idle()


def closed_exactly_once(mocks: Mocks) -> None:
    for log in (mocks.llm.streams, mocks.tts.streams, mocks.asr.calls):
        assert log.closed == log.opened


def audio_ms(events: list[Event]) -> float:
    return sum(
        len(base64.b64decode(event["delta"])) / 48
        for event in events
        if event["type"] == "response.output_audio.delta"
    )


@pytest.mark.parametrize("trigger", TRIGGERS)
@pytest.mark.parametrize("phase", PHASES)
async def test_cancellation(phase: str, trigger: str) -> None:
    clock = ManualClock()
    gateway, mocks = build(phase, clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"tools": [{"type": "function", "name": "lookup"}]})
        point = await reach(phase, client, mocks, clock)
        await settle(client)
        if phase != "paced":
            clock.advance(PLAYED_S)  # D5: the client had time to play all it was sent
        if trigger == "disconnect":
            await client.ws.close()
            async with asyncio.timeout(CLOSE_BOUND_S):
                await mocks.asr.calls.idle()
                while gateway.admission.active:  # noqa: ASYNC110 (a counter, not an event)
                    await asyncio.sleep(0.005)
            await closed_without_release(mocks)
            closed_exactly_once(mocks)
            assert_causal(client.events, ended=False)
            return
        before = len(client.events)
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
        if phase == "paced":
            clock.advance(PLAYED_S)  # the pacer would release its audio now, if it still could
        release_all(mocks)
        async with asyncio.timeout(CLOSE_BOUND_S):
            await mocks.asr.calls.idle()
        await settle(client, quiet_s=0.2)
        check_ending(events, trigger)
        if phase == "paced":
            sent = audio_ms(client.events)
            assert 0 < sent < mocks.tts.samples_for("你好。") / 24  # the paced rest never went out
            assert audio_ms(client.events[before:]) == 0
        cancelled = events[-1]["response"]
        await check_history(client, mocks, cancelled, phase=phase, trigger=trigger)
        closed_exactly_once(mocks)
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
        started = types.index("input_audio_buffer.speech_started")
        assert started < types.index("response.done")
        # One synchronous handler: speech_started, the terminal events, response.done.
        assert set(types[started + 1 : -1]) <= TERMINAL, types[started:]
    for output in response["output"]:
        assert output["type"] == "message", "a call cancelled before completion is never delivered"
        assert output["status"] == "incomplete"


async def check_history(
    client: Client, mocks: Mocks, ended: Event, *, phase: str, trigger: str
) -> None:
    """The next LLM request sees what the client can have heard of the ended response, and no
    more: everything it was sent once the clock has run (D5), nothing when no time passed."""
    await client.say("继续。")
    await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
    await client.until("response.done")
    assistant = [
        message.content
        for message in mocks.llm.requests[-1].messages
        if message.role == "assistant"
    ]
    heard = [part.get("transcript") for output in ended["output"] for part in output["content"]]
    if phase == "paced" and trigger in ("cancel", "barge_in"):
        assert assistant == []  # cut where nothing had time to play (D5)
    else:
        assert assistant == [text for text in heard if text]

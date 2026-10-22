"""Replay of AICC's real client frames (spikes/aicc/traces, ARCHITECTURE §2, §11).

Frames are sent exactly as captured, except that ids minted by AICC's fake server (``item-1``,
``call-7``, ``fc_1``, ``answer-item``, ...) are replaced by the ids this server minted for the
corresponding items. The mock LLM is scripted to make the tool calls AICC's fake server made.
No frame may draw an ``error``, and each must get its documented answer.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from mlx_realtime.providers.base import LLMDone, TextDelta, ToolCall
from tests.support import Client, Event, Mocks, build_gateway, running

TRACES = Path(__file__).parents[2] / "spikes" / "aicc" / "traces"
LOOKUP = ToolCall("provider-call", "lookup_balance", '{"accountId":"42"}')


def frames(name: str) -> list[Event]:
    lines = (TRACES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@asynccontextmanager
async def aicc_session() -> AsyncIterator[tuple[Client, Mocks]]:
    gateway, mocks = build_gateway(transcripts=["喂？"])
    async with running(gateway) as address, Client.connect(address, model="gpt-realtime") as client:
        await client.until("conversation.created")
        yield client, mocks
    assert [event for event in client.events if event["type"] == "error"] == []


async def updated(client: Client, frame: Event) -> Event:
    await client.send(frame)
    return (await client.until("session.updated"))[-1]["session"]


async def respond(client: Client, frame: Event) -> Event:
    """Send a ``response.create``; return the finished response, which must be completed."""
    await client.send(frame)
    response = (await client.until("response.done"))[-1]["response"]
    assert response["status"] == "completed", response
    return response


async def answered(client: Client, frame: Event, event_type: str) -> Event:
    await client.send(frame)
    return (await client.until(event_type))[-1]


def with_id(frame: Event, **ids: str) -> Event:
    """``frame`` with AICC's ids replaced (top level or inside ``item``)."""
    if "item" in frame:
        return {**frame, "item": {**frame["item"], **ids}}
    return {**frame, **ids}


def output_of(response: Event, item_type: str) -> Event:
    return next(item for item in response["output"] if item["type"] == item_type)


async def test_call_gateway() -> None:
    update, opening, audio, truncate, closing, keypress, create, tool_result, create_again, line = (
        frames("call_gateway")
    )
    async with aicc_session() as (client, mocks):
        session = await updated(client, update)
        assert session["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 500
        assert [tool["name"] for tool in session["tools"]] == ["lookup_balance"]
        greeting = await respond(client, opening)  # a bare response.create on an empty conversation
        await client.send(audio)  # caller audio: silence, so no turn
        await client.quiet()
        message = output_of(greeting, "message")["id"]
        truncated = await answered(
            client, with_id(truncate, item_id=message), "conversation.item.truncated"
        )
        assert truncated["audio_end_ms"] == 120
        assert (await updated(client, closing))["instructions"] == "You are now closing the call."
        await answered(client, keypress, "conversation.item.done")
        mocks.llm.script(LOOKUP, LLMDone("tool_calls", None))
        call = output_of(await respond(client, create), "function_call")
        assert (call["name"], call["arguments"]) == ("lookup_balance", '{"accountId":"42"}')
        result = with_id(tool_result, call_id=call["call_id"])
        done = await answered(client, result, "conversation.item.done")
        assert done["item"]["call_id"] == call["call_id"]
        await respond(client, create_again)
        tool_message = mocks.llm.requests[-1].messages[-1]
        assert (tool_message.role, tool_message.tool_call_id) == ("tool", "provider-call")
        await respond(client, line)


async def test_opening_with_a_flow_voice() -> None:
    update, opening = frames("opening_gateway_with_voice")
    async with aicc_session() as (client, mocks):
        session = await updated(client, update)
        assert session["audio"]["output"]["voice"] == "flow-voice"  # never rejected (D19)
        response = await respond(client, opening)
        assert response["audio"]["output"]["voice"] == "flow-voice"
        system = mocks.llm.requests[-1].messages[0]
        assert (system.role, system.content) == ("system", opening["response"]["instructions"])
        assert mocks.tts.requests[0].voice == "flow-voice"


async def test_wait_line_during_a_tool_call() -> None:
    update, opening, line, tool_result, delete, owed = frames("waitline_gateway")
    async with aicc_session() as (client, mocks):
        await updated(client, update)
        mocks.llm.script(LOOKUP, LLMDone("tool_calls", None))
        first_call = output_of(await respond(client, opening), "function_call")
        mocks.llm.script(
            TextDelta("请稍等。"),
            ToolCall("p2", "lookup_balance", "{}"),
            LLMDone("tool_calls", None),
        )
        line_call = output_of(await respond(client, line), "function_call")
        await answered(
            client, with_id(tool_result, call_id=first_call["call_id"]), "conversation.item.done"
        )
        deleted = await answered(
            client, with_id(delete, item_id=line_call["id"]), "conversation.item.deleted"
        )
        assert deleted["item_id"] == line_call["id"]
        await respond(client, owed)
        history = mocks.llm.requests[-1].messages
        assert [message.tool_calls[0].call_id for message in history if message.tool_calls] == [
            "provider-call"
        ]


async def test_unheard_answer_then_keypress() -> None:
    update, opening, tool_result, create, delete, note, cancel, truncate, keypress, owed = frames(
        "unheard_keypress_gateway"
    )
    async with aicc_session() as (client, mocks):
        await updated(client, update)
        mocks.llm.script(LOOKUP, LLMDone("tool_calls", None))
        call = output_of(await respond(client, opening), "function_call")
        await answered(
            client, with_id(tool_result, call_id=call["call_id"]), "conversation.item.done"
        )
        answer = output_of(await respond(client, create), "message")
        await answered(client, with_id(delete, item_id=answer["id"]), "conversation.item.deleted")
        await answered(client, note, "conversation.item.done")
        # Adaptation: the trace's cancel interrupts a reply in progress; start that reply here.
        mocks.tts.next_chunk.hold()
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        cancelled = (await answered(client, cancel, "response.done"))["response"]
        assert cancelled["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
        mocks.tts.next_chunk.release()
        reply = output_of(cancelled, "message")["id"]
        truncated = await answered(
            client, with_id(truncate, item_id=reply), "conversation.item.truncated"
        )
        assert truncated["audio_end_ms"] <= truncate["audio_end_ms"]  # clamped to the audio sent
        await answered(client, keypress, "conversation.item.done")
        await respond(client, owed)


async def test_dropped_line_retry() -> None:
    update, opening, line, direction, retry = frames("dropped_line_retry_gateway")
    async with aicc_session() as (client, _):
        await updated(client, update)
        for frame in (opening, line):
            await respond(client, frame)
        await answered(client, direction, "conversation.item.done")
        await respond(client, retry)

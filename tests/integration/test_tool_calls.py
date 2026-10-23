"""M5 tool calling on the wire (ARCHITECTURE §6.7): golden traces of every call shape, and the
rules around them: ``tool_choice``, calls a response did not offer, the AICC tool-only deadline,
and the round trip's history.

Each call is its own item with the events of §6.7: ``response.output_item.added`` →
``conversation.item.added`` → ``response.function_call_arguments.delta`` (complete, once) →
``.done`` → ``response.output_item.done`` → ``conversation.item.done``.
"""

from __future__ import annotations

import asyncio

from mlx_realtime.providers.base import LLMDone, TextDelta, ToolCall, Usage
from tests.conftest import GoldenCheck
from tests.integration.invariants import assert_causal
from tests.support import Client, Event, build_gateway, running

MANUAL = {"audio": {"input": {"turn_detection": None}}}
LOOKUP = {
    "type": "function",
    "name": "lookup_balance",
    "description": "Read the caller's balance.",
    "parameters": {"type": "object", "properties": {"accountId": {"type": "string"}}},
}
TOOLS = {**MANUAL, "tools": [LOOKUP]}
CALL = ToolCall("provider-1", "lookup_balance", '{"accountId":"42"}')
SECOND = ToolCall("provider-2", "lookup_balance", '{"accountId":"43"}')
RESULT = '{"balance":"12.50","hint":"Tell the caller the balance, then offer to close."}'
CALL_EVENTS = [
    "response.output_item.added",
    "conversation.item.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_item.done",
    "conversation.item.done",
]


def calls_of(response: Event) -> list[Event]:
    return [item for item in response["output"] if item["type"] == "function_call"]


def call_events(events: list[Event], item_id: str) -> list[str]:
    """The types of the events about one call item, in order."""
    return [
        event["type"]
        for event in events
        if event.get("item_id") == item_id or event.get("item", {}).get("id") == item_id
    ]


async def output_for(client: Client, call_id: str, output: str = RESULT) -> None:
    item = {"type": "function_call_output", "call_id": call_id, "output": output}
    await client.send({"type": "conversation.item.create", "item": item})
    await client.until("conversation.item.done")


async def test_call_only(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(CALL, LLMDone("tool_calls", Usage(40, 9)))
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    golden("tool_call_only", client.trace)
    response = events[-1]["response"]
    (call,) = calls_of(response)
    assert [item["type"] for item in response["output"]] == ["function_call"]
    assert call_events(events, call["id"]) == CALL_EVENTS
    assert mocks.tts.requests == []
    assert_causal(client.events)


async def test_text_then_call(golden: GoldenCheck) -> None:
    """The text before the call is spoken normally; the call follows as its own item."""
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(TextDelta("好的，我查一下。"), CALL, LLMDone("tool_calls", Usage(40, 15)))
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    golden("tool_text_and_call", client.trace)
    response = events[-1]["response"]
    assert [item["type"] for item in response["output"]] == ["message", "function_call"]
    assert response["output"][0]["content"][0]["transcript"] == "好的，我查一下。"
    assert [request.text for request in mocks.tts.requests] == ["好的，我查一下。"]
    assert_causal(client.events)


async def test_round_trip(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(CALL, LLMDone("tool_calls", Usage(40, 9)))
        await client.send({"type": "response.create"})
        (call,) = calls_of((await client.until("response.done"))[-1]["response"])
        await output_for(client, call["call_id"])
        mocks.llm.script(TextDelta("您的余额是十二元五角。"), LLMDone("stop", Usage(70, 10)))
        await client.send({"type": "response.create"})
        await client.until("response.done")
    golden("tool_round_trip", client.trace)
    history = mocks.llm.requests[-1].messages
    assert [(message.role, message.content) for message in history] == [
        ("user", "查一下我的余额。"),
        ("assistant", ""),
        ("tool", RESULT),  # the hint reaches the model verbatim
    ]
    assert history[1].tool_calls == (CALL,)
    assert history[2].tool_call_id == "provider-1"  # the provider's own id, not the wire id
    assert_causal(client.events)


async def test_cancel_before_the_call_is_complete(golden: GoldenCheck) -> None:
    """The text is spoken; the call is still being generated when the client cancels: no call
    is ever delivered, before or after the cancel."""
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(TextDelta("好的，我查一下。"), CALL, LLMDone("tool_calls", None))
        mocks.llm.next_token.hold()  # the text passes; the call does not
        mocks.tts.next_chunk.hold()  # one audio delta goes out, so the trace is exact
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.llm.next_token.held()
        await mocks.tts.next_chunk.held()
        await client.send({"type": "response.cancel"})
        events = await client.until("response.done")
        mocks.llm.next_token.release()
        mocks.tts.next_chunk.release()
        await mocks.llm.streams.idle()
        await client.quiet(0.2)
    golden("tool_cancel_before_call", client.trace)
    response = events[-1]["response"]
    assert response["status"] == "cancelled"
    assert calls_of(response) == []
    assert not [event for event in client.events if "function_call" in event["type"]]
    assert_causal(client.events)


async def test_cancel_after_a_delivered_call(golden: GoldenCheck) -> None:
    """A call delivered before the cancel stays: it is in the cancelled response's output and
    in the conversation, and its output is accepted and replayed."""
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(TextDelta("好的，我查一下。"), CALL, LLMDone("tool_calls", None))
        mocks.llm.next_token.hold(after=1)  # the text and the call pass; the end does not
        mocks.tts.next_chunk.hold()
        await client.send({"type": "response.create"})
        delivered = await client.until("response.output_audio.delta")
        assert "response.function_call_arguments.done" in [event["type"] for event in delivered]
        await mocks.llm.next_token.held()
        await mocks.tts.next_chunk.held()
        await client.send({"type": "response.cancel"})
        events = await client.until("response.done")
        mocks.llm.next_token.release()
        mocks.tts.next_chunk.release()
        response = events[-1]["response"]
        (call,) = calls_of(response)
        await output_for(client, call["call_id"])
        mocks.llm.script(TextDelta("您的余额是十二元五角。"), LLMDone("stop", Usage(70, 10)))
        await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
        await client.until("response.done")
    golden("tool_cancel_after_call", client.trace)
    assert response["status"] == "cancelled"
    assert call["status"] == "completed"
    history = mocks.llm.requests[-1].messages
    assert [message.role for message in history[-2:]] == ["assistant", "tool"]
    assert history[-2].tool_calls == (CALL,)
    assert_causal(client.events)


async def test_two_calls_in_one_response(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下两个账户的余额。")
        mocks.llm.script(CALL, SECOND, LLMDone("tool_calls", Usage(40, 18)))
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
        first, second = calls_of(events[-1]["response"])
        await output_for(client, second["call_id"], '{"balance":"3.00"}')
        await output_for(client, first["call_id"])
        mocks.llm.script(TextDelta("两个账户都查到了。"), LLMDone("stop", Usage(90, 9)))
        await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
        await client.until("response.done")
    golden("tool_two_calls", client.trace)
    assert first["call_id"] != second["call_id"]
    assert first["id"] != second["id"]
    assert call_events(events, first["id"]) == CALL_EVENTS
    assert call_events(events, second["id"]) == CALL_EVENTS
    history = mocks.llm.requests[-1].messages
    assert [message.tool_call_id for message in history if message.role == "tool"] == [
        "provider-2",
        "provider-1",
    ]
    assert_causal(client.events)


async def test_tool_choice_none_sends_no_tools_and_delivers_no_call() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({**TOOLS, "tool_choice": "none"})
        await client.say("查一下我的余额。")
        mocks.llm.script(CALL, LLMDone("tool_calls", None))  # a model that calls anyway
        await client.send({"type": "response.create"})
        session_none = (await client.until("response.done"))[-1]["response"]
        mocks.llm.script(CALL, LLMDone("tool_calls", None))
        await client.send({"type": "response.create", "response": {"tool_choice": "auto"}})
        response_auto = (await client.until("response.done"))[-1]["response"]
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", "tool_choice": "auto"}}
        )
        await client.until("session.updated")
        mocks.llm.script(CALL, LLMDone("tool_calls", None))
        await client.send({"type": "response.create", "response": {"tool_choice": "none"}})
        response_none = (await client.until("response.done"))[-1]["response"]
    assert [request.tool_choice for request in mocks.llm.requests] == ["none", "auto", "none"]
    assert session_none["output"] == []
    assert session_none["status"] == "completed"
    assert len(calls_of(response_auto)) == 1
    assert response_none["output"] == []


async def test_a_call_to_an_undeclared_function_is_dropped() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("转人工。")
        mocks.llm.script(ToolCall("p9", "transfer_call", "{}"), CALL, LLMDone("tool_calls", None))
        await client.send({"type": "response.create"})
        response = (await client.until("response.done"))[-1]["response"]
    assert [call["name"] for call in calls_of(response)] == ["lookup_balance"]


async def test_required_and_forced_tool_choices_are_rejected() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        for event in (
            {"type": "session.update", "session": {"type": "realtime", "tool_choice": "required"}},
            {"type": "response.create", "response": {"tool_choice": "required"}},
            {
                "type": "response.create",
                "response": {"tool_choice": {"type": "function", "name": "lookup_balance"}},
            },
        ):
            await client.send(event)
            error = (await client.until("error"))[-1]["error"]
            assert error["code"] == "invalid_value"
            assert error["param"].endswith("tool_choice")
            assert "cannot force the model to call a tool" in error["message"]
        await client.quiet()
    assert mocks.llm.requests == []


async def test_whitespace_before_a_call_opens_no_message() -> None:
    """The real server sends the newline a model writes before ``<tool_call>`` as text."""
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.script(TextDelta("\n\n"), CALL, TextDelta("\n"), LLMDone("tool_calls", None))
        await client.send({"type": "response.create"})
        response = (await client.until("response.done"))[-1]["response"]
    assert [item["type"] for item in response["output"]] == ["function_call"]
    assert mocks.tts.requests == []


async def test_a_tool_only_response_is_done_at_once() -> None:
    """AICC abandons a tool-only response not done within 3 s of ``response.created``. Nothing
    waits after the call: with the LLM's end 50 ms after the call, done follows within 0.5 s,
    in audio modality and with the TTS slow to start."""
    gateway, mocks = build_gateway()
    mocks.tts.first_chunk.delay_s = 2.0  # a TTS that would be slow, were it asked
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TOOLS)
        await client.say("查一下我的余额。")
        mocks.llm.next_token.delay_s = 0.05
        mocks.llm.script(CALL, LLMDone("tool_calls", None))
        await client.send({"type": "response.create"})
        await client.until("response.created")
        created = asyncio.get_running_loop().time()
        await client.until("response.done")
        elapsed = asyncio.get_running_loop().time() - created
    assert elapsed < 0.5
    assert mocks.tts.requests == []

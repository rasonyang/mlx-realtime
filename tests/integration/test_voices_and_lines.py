"""Voices (D19) and verbatim lines (D16) on the wire.

- A voice name that is not configured is accepted, spoken by the default voice, logged as a
  warning, and the effective voice is echoed. Once the session has produced audio its voice is
  locked, as GA's is: changing it is the one voice error (``cannot_update_voice``).
- A response whose own instructions start with a configured direction speaks the rest as
  written, without the LLM: on an empty conversation, mid-call, and while a function call is
  unanswered (AICC's wait line). Its events are those of any audio response.
"""

from __future__ import annotations

import logging

import pytest

from mlx_realtime.config import DEFAULT_VERBATIM_PREFIXES
from mlx_realtime.providers.base import LLMDone, TextDelta, ToolCall
from tests.integration.invariants import assert_causal
from tests.support import Client, Event, build_gateway, running

MANUAL = {"audio": {"input": {"turn_detection": None}}}
SAY_EN, SAY_ZH = DEFAULT_VERBATIM_PREFIXES


def voice_update(voice: str, event_id: str | None = None) -> Event:
    event: Event = {
        "type": "session.update",
        "session": {"type": "realtime", "audio": {"output": {"voice": voice}}},
    }
    if event_id is not None:
        event["event_id"] = event_id
    return event


def kinds(events: list[Event]) -> list[str]:
    return [event["type"] for event in events]


async def test_an_unknown_voice_is_accepted_spoken_by_the_default_and_echoed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway, mocks = build_gateway()
    caplog.set_level(logging.WARNING, logger="mlx_realtime.session.actor")
    async with running(gateway) as address, Client.connect(address) as client:
        created = (await client.until("session.created"))[-1]
        assert created["session"]["audio"]["output"]["voice"] == "vivian"  # the default voice
        await client.until("conversation.created")
        await client.send(voice_update("serena"))
        echoed = (await client.until("session.updated"))[-1]["session"]
        assert echoed["audio"]["output"]["voice"] == "serena"
        await client.send(voice_update("marin"))
        echoed = (await client.until("session.updated"))[-1]["session"]
        assert echoed["audio"]["output"]["voice"] == "vivian"
        await client.say("你好。")
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]["response"]
    assert done["audio"]["output"]["voice"] == "vivian"
    assert [request.voice for request in mocks.tts.requests] == ["vivian"]
    (warning,) = [r for r in caplog.records if r.getMessage().startswith("unknown voice")]
    assert (warning.__dict__["voice"], warning.__dict__["effective"]) == ("marin", "vivian")


async def test_the_voice_locks_once_the_session_has_spoken() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.send(voice_update("uncle_fu"))
        await client.until("session.updated")
        await client.say("你好。")
        await client.send(
            {"type": "response.create", "response": {"audio": {"output": {"voice": "aiden"}}}}
        )
        done = (await client.until("response.done"))[-1]["response"]
        assert done["audio"]["output"]["voice"] == "aiden"  # before any audio: allowed
        # The session has spoken: its voice can no longer change.
        await client.send(voice_update("serena", "change"))
        error = (await client.until("error"))[-1]["error"]
        assert (error["code"], error["param"], error["event_id"]) == (
            "cannot_update_voice",
            "session.audio.output.voice",
            "change",
        )
        await client.send(voice_update("uncle_fu"))  # the same voice again: nothing changes
        assert (await client.until("session.updated"))[-1]["session"]["audio"]["output"][
            "voice"
        ] == "uncle_fu"
        await client.send(
            {
                "type": "response.create",
                "event_id": "per-response",
                "response": {"audio": {"output": {"voice": "serena"}}},
            }
        )
        error = (await client.until("error"))[-1]["error"]
        assert (error["code"], error["param"]) == (
            "cannot_update_voice",
            "response.audio.output.voice",
        )
        update = {"type": "session.update", "session": {"type": "realtime", "instructions": "x"}}
        await client.send(update)  # AICC's later updates carry no voice: never affected
        assert (await client.until("session.updated"))[-1]["session"]["instructions"] == "x"
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]["response"]
    assert done["audio"]["output"]["voice"] == "uncle_fu"
    assert [request.voice for request in mocks.tts.requests] == ["aiden", "uncle_fu"]


async def test_a_line_on_an_empty_conversation_has_the_events_of_an_audio_response() -> None:
    line = "您好，欢迎致电诺瓦网络。请问有什么可以帮您？"
    gateway, mocks = build_gateway()
    async with running(gateway) as address:
        async with Client.connect(address) as answered:
            await answered.start(MANUAL)
            await answered.say(line)  # the echo LLM answers with the same text
            await answered.send({"type": "response.create"})
            normal = await answered.until("response.done")
        llm_requests = len(mocks.llm.requests)
        async with Client.connect(address) as spoken:
            await spoken.start(MANUAL)
            await spoken.send(
                {"type": "response.create", "response": {"instructions": SAY_ZH + line}}
            )
            verbatim = await spoken.until("response.done")
    assert len(mocks.llm.requests) == llm_requests  # no LLM call for the line
    assert kinds(verbatim) == kinds(normal)
    done = verbatim[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["transcript"] == line
    assert done["usage"] is None  # no tokens were generated
    assert_causal(spoken.events)


async def test_a_line_mid_call_becomes_the_assistant_message() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("我想查一下余额。")
        await client.send({"type": "response.create"})
        await client.until("response.done")
        line = "Thanks for calling NovaNet."
        await client.send({"type": "response.create", "response": {"instructions": SAY_EN + line}})
        done = (await client.until("response.done"))[-1]["response"]
        await client.say("还有一个问题。")
        await client.send({"type": "response.create"})
        await client.until("response.done")
    assert done["output"][0]["content"][0]["transcript"] == line
    history = [(message.role, message.content) for message in mocks.llm.requests[-1].messages]
    assert ("assistant", line) in history  # the next request sees the line as said
    assert [request.text for request in mocks.tts.requests][1] == line
    assert {request.lang for request in mocks.tts.requests[1:2]} == {"en"}


async def test_the_wait_line_while_a_function_call_is_unanswered() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "tools": [{"type": "function", "name": "lookup"}]})
        await client.say("查一下余额。")
        mocks.llm.script(ToolCall("p1", "lookup", "{}"), LLMDone("tool_calls", None))
        await client.send({"type": "response.create"})
        call = (await client.until("response.done"))[-1]["response"]["output"][0]
        assert call["type"] == "function_call"
        # The call is unanswered: the line still speaks, as written, with no LLM call.
        requests = len(mocks.llm.requests)
        line = "请稍等，我查一下。"
        await client.send({"type": "response.create", "response": {"instructions": SAY_ZH + line}})
        done = (await client.until("response.done"))[-1]["response"]
        assert len(mocks.llm.requests) == requests
        assert [item["type"] for item in done["output"]] == ["message"]
        assert done["output"][0]["content"][0]["transcript"] == line
        output = {"type": "function_call_output", "call_id": call["call_id"], "output": "{}"}
        await client.send({"type": "conversation.item.create", "item": output})
        await client.until("conversation.item.done")
        mocks.llm.script(TextDelta("您的余额是十二元。"), LLMDone("stop", None))
        await client.send({"type": "response.create"})
        assert (await client.until("response.done"))[-1]["response"]["status"] == "completed"
    roles = [message.role for message in mocks.llm.requests[-1].messages]
    assert roles == ["user", "assistant", "assistant", "tool"]  # call, the line, the output


async def test_a_line_in_text_modality_and_a_line_with_its_own_voice() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        line = "Say this."
        params = {"instructions": SAY_EN + line, "output_modalities": ["text"]}
        await client.send({"type": "response.create", "response": params})
        events = await client.until("response.done")
        assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == [line]
        params = {"instructions": SAY_EN + line, "audio": {"output": {"voice": "uncle_fu"}}}
        await client.send({"type": "response.create", "response": params})
        await client.until("response.done")
    assert mocks.llm.requests == []
    assert [(request.text, request.voice) for request in mocks.tts.requests] == [(line, "uncle_fu")]


async def test_other_instructions_and_disabled_prefixes_go_to_the_llm() -> None:
    gateway, mocks = build_gateway({"lines": {"verbatim_prefixes": []}})
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("你好")
        await client.send({"type": "response.create", "response": {"instructions": SAY_EN + "Hi."}})
        await client.until("response.done")
    (request,) = mocks.llm.requests
    assert request.messages[0].content == SAY_EN + "Hi."  # the instructions override, as GA does

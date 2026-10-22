"""Causal invariants over a long session: interleaved audio and transcript, cancels, barge-in,
text responses and a tool call."""

from __future__ import annotations

import base64

from mlx_realtime.providers.base import LLMDone, TextDelta, ToolCall
from tests.integration.invariants import assert_causal
from tests.support import Client, Event, build_gateway, running, tone

STORY = (
    "从前有一座山。山里有一座庙，庙里有一个老和尚。"
    "Once upon a time, there was a mountain. The temple had an old monk, who told stories!"
    "老和尚每天都讲故事，讲的是什么呢？就是这个故事。"
)


async def test_long_session_keeps_every_invariant() -> None:
    gateway, mocks = build_gateway(speech=[(1000, 1800)], transcripts=["再说一遍。"], chunk_ms=30)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"tools": [{"type": "function", "name": "lookup"}]})
        await client.say(STORY)
        await client.send({"type": "response.create"})
        await client.until("response.done")
        mocks.llm.script(
            TextDelta("我查一下。"), ToolCall("p1", "lookup", "{}"), LLMDone("tool_calls", None)
        )
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]
        call_id = done["response"]["output"][1]["call_id"]
        output = {"type": "function_call_output", "call_id": call_id, "output": '{"ok":1}'}
        await client.send({"type": "conversation.item.create", "item": output})
        await client.until("conversation.item.done")
        await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
        await client.until("response.done")
        mocks.tts.next_chunk.hold()
        await client.say(STORY)
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        await client.send({"type": "response.cancel"})
        await client.until("response.done")
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await client.append(tone(1300))  # barge-in
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        await client.append(tone(1000))
        await client.until("response.done")
    events = client.events
    assert_causal(events)
    assert_transcript_leads_audio(events)
    statuses = [event["response"]["status"] for event in events if event["type"] == "response.done"]
    assert statuses == [
        "completed",
        "completed",
        "completed",
        "cancelled",
        "cancelled",
        "completed",
    ]


def assert_transcript_leads_audio(events: list[Event]) -> None:
    """A segment's transcript delta goes out just before its first audio, and an item's audio is
    never ahead of its transcript."""
    stream = [
        event
        for event in events
        if event["type"]
        in ("response.output_audio_transcript.delta", "response.output_audio.delta")
    ]
    seen_transcript: set[str] = set()
    for index, event in enumerate(stream):
        if event["type"] == "response.output_audio_transcript.delta":
            seen_transcript.add(event["item_id"])
            following = stream[index + 1] if index + 1 < len(stream) else None
            if following is not None and following["item_id"] == event["item_id"]:
                assert following["type"] == "response.output_audio.delta", (
                    "transcript without audio"
                )
        else:
            assert event["item_id"] in seen_transcript, "audio before any transcript"


async def test_audio_adds_up_to_the_synthesized_length() -> None:
    gateway, mocks = build_gateway(chunk_ms=30)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"audio": {"input": {"turn_detection": None}}})
        await client.say(STORY)
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    audio = sum(
        len(base64.b64decode(event["delta"]))
        for event in events
        if event["type"] == "response.output_audio.delta"
    )
    expected = sum(mocks.tts.samples_for(request.text) for request in mocks.tts.requests) * 2
    assert audio == expected
    assert len(mocks.tts.requests) > 3  # several segments, so audio and transcript interleave

"""Truncation (ARCHITECTURE §6.6, D5, D18) on the wire, with the session clock under the test's
control (``ManualClock``):

- without a client truncate, an interrupted answer is cut before the next LLM request to what
  can have played: min(the audio sent, the time since its first audio delta was sent) (D5);
- a later client truncate overrides that cut;
- ``audio_end_ms`` beyond the audio sent is clamped, with a warning, never an error (D18);
- truncating a completed item works (AICC truncates after ``response.done``);
- a truncate to 0 followed by a delete, AICC's flow for an unheard answer, works.
"""

from __future__ import annotations

import logging

import pytest

from tests.support import Client, Event, ManualClock, Mocks, build_gateway, running

MANUAL = {"audio": {"input": {"turn_detection": None}}}
ANSWER = "你好。我们开始吧。"  # the mock LLM echoes it: 300 ms, then 600 ms of audio (100 ms/char)


async def heard_by_the_llm(client: Client, mocks: Mocks) -> list[str]:
    """Ask for a text reply; return the assistant messages that request carried."""
    await client.say("继续。")
    await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
    await client.until("response.done")
    return [
        message.content
        for message in mocks.llm.requests[-1].messages
        if message.role == "assistant"
    ]


async def truncate(client: Client, item_id: str, audio_end_ms: int) -> Event:
    await client.send(
        {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms,
        }
    )
    return (await client.until("conversation.item.truncated"))[-1]


async def start_answer(client: Client, mocks: Mocks) -> str:
    """The answer's first segment and the first 100 ms of its second are sent, then the TTS
    stops (held); returns the answer's item id."""
    await client.say(ANSWER)
    mocks.tts.next_chunk.hold(after=2)
    await client.send({"type": "response.create"})
    received_ms = 0.0
    item_id = ""
    while received_ms < 400:
        event = await client.recv()
        if event["type"] == "response.output_audio.delta":
            item_id = event["item_id"]
            received_ms += len(event["delta"]) * 3 / 4 / 48
    await mocks.tts.next_chunk.held()
    return item_id


@pytest.mark.parametrize(
    ("played_s", "heard"),
    [
        (0.0, []),  # cancelled at once: nothing can have played
        (0.15, ["你"]),  # half of the first segment's 300 ms
        # The first segment, and half of the 100 ms synthesized of the second: a segment cut
        # while still being synthesized counts its text over the audio made so far.
        (0.35, ["你好。我们开"]),
        (60.0, ["你好。我们开始吧。"]),  # long after: everything sent (D5 never exceeds it)
    ],
)
async def test_without_a_truncate_the_cut_is_what_can_have_played(
    played_s: float, heard: list[str]
) -> None:
    clock = ManualClock()
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100, clock=clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await start_answer(client, mocks)
        clock.advance(played_s)
        await client.send({"type": "response.cancel"})
        done = (await client.until("response.done"))[-1]["response"]
        mocks.tts.next_chunk.release()
        # The client was told it got the text of every segment whose audio it was sent...
        assert done["output"][0]["content"][0]["transcript"] == ANSWER
        # ...but the next request sees only what can have played.
        assert await heard_by_the_llm(client, mocks) == heard


async def test_a_client_truncate_overrides_the_fallback() -> None:
    clock = ManualClock()
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100, clock=clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        item_id = await start_answer(client, mocks)
        clock.advance(0.1)
        await client.send({"type": "response.cancel"})
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        assert (await truncate(client, item_id, 350))["audio_end_ms"] == 350
        # 50 of the second segment's 100 synthesized ms: half of its six characters.
        assert await heard_by_the_llm(client, mocks) == ["你好。我们开"]


async def test_a_truncate_beyond_the_audio_sent_is_clamped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="mlx_realtime.session.actor")
    clock = ManualClock()
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100, clock=clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        item_id = await start_answer(client, mocks)
        await client.send({"type": "response.cancel"})
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        truncated = await truncate(client, item_id, 99_999)
        assert truncated["audio_end_ms"] == 400  # the audio sent
        await client.quiet()  # no error
    assert [r.getMessage() for r in caplog.records] == ["truncate beyond the audio sent; clamped"]
    assert caplog.records[0].__dict__["sent_ms"] == 400


async def test_a_completed_answer_can_be_truncated_and_an_unheard_one_deleted() -> None:
    """AICC truncates after ``response.done`` (its queue still playing), and takes an answer
    nobody heard out of the conversation: a truncate to 0, then a delete."""
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say(ANSWER)
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]["response"]
        assert done["status"] == "completed"
        item_id = done["output"][0]["id"]
        assert (await truncate(client, item_id, 300))["audio_end_ms"] == 300
        assert await heard_by_the_llm(client, mocks) == ["你好。"]
        # The next answer, unheard: truncated to 0, then deleted.
        await client.say(ANSWER)
        await client.send({"type": "response.create"})
        unheard = (await client.until("response.done"))[-1]["response"]["output"][0]["id"]
        assert (await truncate(client, unheard, 0))["audio_end_ms"] == 0
        await client.send({"type": "conversation.item.delete", "item_id": unheard})
        assert (await client.until("conversation.item.deleted"))[-1]["item_id"] == unheard
        # The first answer as heard, the text reply after it; the unheard answer is gone.
        assert await heard_by_the_llm(client, mocks) == ["你好。", "继续。"]
    assert "error" not in [event["type"] for event in client.events]

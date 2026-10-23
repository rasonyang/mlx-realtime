"""Golden traces: both directions of deterministic sessions, ids normalized (tests/golden/).

Regenerate with ``uv run pytest tests/integration/test_golden.py --update-golden`` and review
the diff: a changed trace is a protocol change.
"""

from __future__ import annotations

import base64

from tests.conftest import GoldenCheck
from tests.support import Client, ManualClock, build_gateway, running, tone

MANUAL = {"audio": {"input": {"turn_detection": None}}}
TEXT = {"output_modalities": ["text"], **MANUAL}


async def test_connect(golden: GoldenCheck) -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address, model="gpt-realtime") as client:
        await client.until("conversation.created")
        await client.send({"type": "session.update", "session": {"type": "realtime", **TEXT}})
        await client.until("session.updated")
    golden("connect", client.trace)


async def test_text_turn(golden: GoldenCheck) -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TEXT)
        await client.say("你好，世界。")
        await client.send({"type": "response.create"})
        await client.until("response.done")
    golden("text_turn", client.trace)


async def test_audio_turn_with_manual_commit(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway(transcripts=["你好。今天天气很好，我们出去走走吧。"])
    mocks.asr.result.hold()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.append(tone(300))
        await client.send({"type": "input_audio_buffer.commit"})
        await client.until("conversation.item.added")
        await client.send({"type": "response.create"})
        await client.until("response.created")
        mocks.asr.result.release()  # the transcript arrives after the response was created
        await client.until("response.done")
    golden("audio_turn", client.trace)
    assert mocks.asr.requests == [300]
    assert [request.text for request in mocks.tts.requests] == [
        "你好。",
        "今天天气很好，我们出去走走吧。",
    ]


async def test_verbatim_line(golden: GoldenCheck) -> None:
    """D16: a line is spoken as written, with the events of any audio response, and no LLM."""
    gateway, mocks = build_gateway()
    direction = "请一字不差地说出下面这句话，不要添加任何其它内容：\n"
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        line = {"instructions": direction + "您好，欢迎致电诺瓦网络。请问有什么可以帮您？"}
        await client.send({"type": "response.create", "response": line})
        await client.until("response.done")
    golden("verbatim_line", client.trace)
    assert mocks.llm.requests == []
    assert [request.text for request in mocks.tts.requests] == [
        "您好，欢迎致电诺瓦网络。",
        "请问有什么可以帮您？",
    ]


async def test_response_create_while_a_response_is_active(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway()
    mocks.llm.first_token.hold()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(TEXT)
        await client.send({"type": "response.create", "event_id": "first"})
        await client.until("response.created")
        await mocks.llm.first_token.held()
        await client.send({"type": "response.create", "event_id": "second"})
        await client.until("error")
        mocks.llm.first_token.release()
        await client.until("response.done")
    golden("response_while_active", client.trace)


async def test_response_cancel_mid_audio(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("第一句话说完了。第二句话还没有说。")
        mocks.tts.next_chunk.hold()
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        await client.send({"type": "response.cancel", "event_id": "stop"})
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        await client.send({"type": "response.cancel", "event_id": "again"})
        await client.until("error")
        await client.quiet()
    golden("response_cancel", client.trace)
    await mocks.tts.streams.idle()


async def test_barge_in(golden: GoldenCheck) -> None:
    # The scripted detector hears speech from 1000 ms to 1800 ms of input audio. The session
    # clock stands still: the interrupted answer had no time to play, so the next LLM request
    # leaves it out (D5), which its usage shows.
    gateway, mocks = build_gateway(
        speech=[(1000, 1800)], transcripts=["等一下。"], clock=ManualClock()
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await client.say("请讲一个很长很长的故事。")
        mocks.tts.next_chunk.hold()
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        await client.append(tone(1300))  # speech confirmed at 1152 ms
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        await client.append(tone(1000))  # silence from 1792 ms ends the turn at 2016 ms
        await client.until("response.done")
    golden("barge_in", client.trace)


async def test_truncate_and_delete(golden: GoldenCheck) -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("你好。这是一段会被截断的回答。")
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]
        answer = done["response"]["output"][0]["id"]
        user = client.events[3]["item"]["id"]
        for event_id, item_id, end_ms in [
            ("cut", answer, 200),
            ("overshoot", answer, 99_999),
            ("unknown", "item_missing", 0),
            ("user", user, 0),
        ]:
            await client.send(
                {
                    "type": "conversation.item.truncate",
                    "event_id": event_id,
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": end_ms,
                }
            )
            await client.recv()
        await client.send({"type": "conversation.item.delete", "event_id": "d1", "item_id": answer})
        await client.until("conversation.item.deleted")
        await client.send({"type": "conversation.item.delete", "event_id": "d2", "item_id": answer})
        await client.until("error")
    golden("truncate_and_delete", client.trace)


async def test_truncated_text_is_what_the_next_request_sees() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("你好。这是一段会被截断的回答。")
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]
        answer = done["response"]["output"][0]["id"]
        truncate = {"item_id": answer, "content_index": 0, "audio_end_ms": 120}
        await client.send({"type": "conversation.item.truncate", **truncate})
        await client.until("conversation.item.truncated")
        await client.say("继续。")
        await client.send({"type": "response.create"})
        await client.until("response.done")
    # 120 ms of audio: all of "你好。" (60 ms at 20 ms per character), then 60 of the 240 ms of
    # the next segment: a quarter of its 12 characters.
    history = mocks.llm.requests[-1].messages
    assert [(message.role, message.content) for message in history] == [
        ("user", "你好。这是一段会被截断的回答。"),
        ("assistant", "你好。这是一"),
        ("user", "继续。"),
    ]


async def test_empty_transcript_ends_the_turn_without_a_reply(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway(speech=[(500, 900)], transcripts=[""])
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await client.append(tone(1400))
        await client.until("response.done")
        await client.quiet()
    golden("empty_transcript", client.trace)
    assert mocks.llm.requests == []


async def test_protocol_errors(golden: GoldenCheck) -> None:
    gateway, _ = build_gateway()
    bad_frames: list[dict[str, object]] = [
        {"type": "response.cancel", "event_id": "e1"},
        {"type": "input_audio_buffer.commit", "event_id": "e2"},
        {"type": "conversation.item.retrieve", "event_id": "e3", "item_id": "x"},
        {"type": "scooby.dooby.doo", "event_id": "e4"},
        {"event_id": "e5"},
        {
            "type": "session.update",
            "event_id": "e6",
            "session": {"type": "realtime", "modalities": ["text"]},
        },
        {
            "type": "session.update",
            "event_id": "e7",
            "session": {
                "type": "realtime",
                "audio": {"input": {"format": {"type": "audio/pcmu", "rate": 16000}}},
            },
        },
        {
            "type": "session.update",
            "event_id": "e8",
            "session": {
                "type": "realtime",
                "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}},
            },
        },
        {
            "type": "session.update",
            "event_id": "e9",
            "session": {"type": "realtime", "audio": {"output": {"speed": 1.2}}},
        },
        {
            "type": "input_audio_buffer.append",
            "event_id": "e10",
            "audio": base64.b64encode(b"\x00").decode(),
        },
        {
            "type": "conversation.item.create",
            "event_id": "e11",
            "item": {"type": "function_call_output", "call_id": "call_missing", "output": "{}"},
        },
        {
            "type": "conversation.item.create",
            "event_id": "e12",
            "previous_item_id": "item_missing",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "x"}],
            },
        },
        {"type": "response.create", "event_id": "e13", "response": {"conversation": "none"}},
    ]
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        for frame in bad_frames:
            await client.send(frame)
            await client.until("error")
        await client.send_raw(b"\x00\x01")
        await client.until("error")
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", "instructions": "ok"}}
        )
        await client.until("session.updated")  # the session survived every error
    golden("protocol_errors", client.trace)


async def test_session_limit(golden: GoldenCheck) -> None:
    gateway, _ = build_gateway({"gateway": {"max_sessions": 1}})
    async with running(gateway) as address, Client.connect(address) as first:
        await first.until("conversation.created")
        async with Client.connect(address) as second:
            await second.until("error")
            await second.ws.wait_closed()
            assert second.ws.close_code == 1013
    golden("session_limit", second.trace)

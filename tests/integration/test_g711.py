"""G.711 sessions on the wire (M7: ``audio/pcmu``, ``audio/pcma`` at 8 kHz, D2, D3).

- A G.711 turn in and out, pinned by golden traces: μ-law in, A-law out (the two directions are
  independent); the speech events land at the same input-timeline milliseconds as with 24 kHz
  PCM; deltas are at most ``max_delta_ms`` of 8 kHz audio (800 bytes); the formats are echoed.
- Barge-in and truncate count G.711 right: the truncate clamps to the bytes sent / 8, D5 cuts
  what can have played, and pacing keeps at most ``max_lead_ms`` of 8 kHz audio ahead.
- The formats can change after audio has flowed (GA): later appends decode in the new input
  format, later responses speak the new output format, and ``response.create`` may set its own.
- Any byte count is whole G.711 samples; half a PCM sample is still refused.
"""

from __future__ import annotations

import base64
import logging

import numpy as np
import pytest

from mlx_realtime.audio import g711
from mlx_realtime.audio.g711 import Law
from tests.conftest import GoldenCheck
from tests.integration.invariants import assert_causal
from tests.support import (
    G711_SESSION,
    Client,
    Event,
    ManualClock,
    Mocks,
    build_gateway,
    g711_tone,
    running,
    tone,
)

MIXED = {
    "audio": {
        "input": {"format": {"type": "audio/pcmu"}},
        "output": {"format": {"type": "audio/pcma", "rate": 8000}},
    }
}
MANUAL_G711 = {
    "audio": {
        "input": {"format": {"type": "audio/pcmu"}, "turn_detection": None},
        "output": {"format": {"type": "audio/pcmu"}},
    }
}


def deltas(events: list[Event]) -> list[bytes]:
    return [
        base64.b64decode(event["delta"])
        for event in events
        if event["type"] == "response.output_audio.delta"
    ]


def dominant_hz(audio: bytes, law: Law) -> float:
    """The strongest frequency of G.711 audio decoded with ``law``."""
    samples = g711.decode_samples(audio, law).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.argmax(spectrum) * 8000 / len(samples))


async def speech_turn(client: Client, *, g711_input: bool) -> list[Event]:
    """3.3 s of input in 100 ms appends; the scripted detector hears speech at 1000-1800 ms."""
    if g711_input:
        await client.append(g711_tone(1300), bytes_per_ms=8)
    else:
        await client.append(tone(1300))
    await client.until("input_audio_buffer.speech_started")
    if g711_input:
        await client.append(g711_tone(2000), bytes_per_ms=8)
    else:
        await client.append(tone(2000))
    return await client.until("response.done")


def speech_marks(events: list[Event]) -> list[tuple[str, int]]:
    return [
        (event["type"], event.get("audio_start_ms", event.get("audio_end_ms", -1)))
        for event in events
        if event["type"].startswith("input_audio_buffer.speech")
    ]


async def test_a_g711_turn_in_and_out(golden: GoldenCheck) -> None:
    gateway, mocks = build_gateway(speech=[(1000, 1800)], transcripts=["你好。今天天气很好。"])
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MIXED)
        await speech_turn(client, g711_input=True)
        answer = next(e for e in client.events if e["type"] == "response.done")["response"]
        item_id = answer["output"][0]["id"]
        await client.send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": 99_999,
            }
        )
        truncated = (await client.until("conversation.item.truncated"))[-1]
    golden("g711_turn", client.trace)
    session = next(e for e in client.events if e["type"] == "session.updated")["session"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcma"}
    assert answer["audio"]["output"]["format"] == {"type": "audio/pcma"}
    audio = deltas(client.events)
    assert all(0 < len(delta) <= 800 for delta in audio)  # max_delta_ms of 8 kHz, 1 byte each
    total = sum(len(delta) for delta in audio)
    # The mock TTS speaks 20 ms per character: "你好。" and "今天天气很好。", 10 characters.
    assert total == 8 * 200
    assert truncated["audio_end_ms"] == total // 8  # clamped to the audio sent, at 8 kHz
    assert dominant_hz(b"".join(audio), "alaw") == pytest.approx(440, abs=10)
    assert mocks.asr.requests == [1100]  # the committed span, 700-1800 ms, at 16 kHz


async def test_speech_events_land_on_the_same_timeline_at_8_and_24_khz() -> None:
    marks: list[list[tuple[str, int]]] = []
    for g711_input in (False, True):
        gateway, mocks = build_gateway(speech=[(1000, 1800)])
        async with running(gateway) as address, Client.connect(address) as client:
            await client.start(G711_SESSION if g711_input else None)
            await speech_turn(client, g711_input=g711_input)
        marks.append(speech_marks(client.events))
        assert mocks.asr.requests == [1100]
    assert (
        marks[0]
        == marks[1]
        == [
            # the scripted onset at 1000 ms, on a 32 ms frame, minus 300 ms of padding
            ("input_audio_buffer.speech_started", 692),
            ("input_audio_buffer.speech_stopped", 1792),
        ]
    )


async def test_a_g711_barge_in(golden: GoldenCheck) -> None:
    """As ``barge_in.jsonl``, in μ-law both ways: the session clock stands still, so the
    interrupted answer had no time to play and the next request leaves it out (D5)."""
    # 100 ms TTS chunks: the 24 → 8 kHz resampler lets nothing out of a first 40 ms chunk.
    gateway, mocks = build_gateway(
        speech=[(1000, 1800)], transcripts=["等一下。"], clock=ManualClock(), chunk_ms=100
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(G711_SESSION)
        await client.say("请讲一个很长很长的故事。")
        mocks.tts.next_chunk.hold()
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await mocks.tts.next_chunk.held()
        await client.append(g711_tone(1300), bytes_per_ms=8)
        interrupted = (await client.until("response.done"))[-1]["response"]
        mocks.tts.next_chunk.release()
        item_id = interrupted["output"][0]["id"]
        sent = sum(len(delta) for delta in deltas(client.events))
        await client.send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": 5000,
            }
        )
        truncated = (await client.until("conversation.item.truncated"))[-1]
        await client.append(g711_tone(1000), bytes_per_ms=8)
        reply = (await client.until("response.done"))[-1]["response"]
    golden("g711_barge_in", client.trace)
    assert interrupted["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    assert truncated["audio_end_ms"] == sent // 8 < 5000
    assert reply["status"] == "completed"
    assert reply["audio"]["output"]["format"] == {"type": "audio/pcmu"}


async def answer_until_held(client: Client, mocks: Mocks, text: str) -> tuple[str, int]:
    """The answer's first segment and its second's first chunk are synthesized, then the TTS
    stops (held); returns the item and the G.711 bytes sent."""
    await client.say(text)
    mocks.tts.next_chunk.hold(after=2)
    await client.send({"type": "response.create"})
    received = await client.until("response.output_audio.delta")
    await mocks.tts.next_chunk.held()
    item_id = received[-1]["item_id"]
    sent = sum(len(delta) for delta in deltas(received))
    while True:
        try:
            event = await client.recv(wait_s=0.3)
        except TimeoutError:
            return item_id, sent
        if event["type"] == "response.output_audio.delta":
            item_id = event["item_id"]
            sent += len(base64.b64decode(event["delta"]))


@pytest.mark.parametrize(
    ("played_s", "heard"),
    [
        (0.0, []),  # cancelled at once: nothing can have played
        (0.15, ["你"]),  # half of the first segment's 300 ms (2400 bytes)
        (60.0, ["你好。我们开始吧。"]),  # everything sent; the cut segment over what was made
    ],
)
async def test_d5_cuts_g711_audio_to_what_can_have_played(
    played_s: float, heard: list[str]
) -> None:
    clock = ManualClock()
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100, clock=clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL_G711)
        first = client.trace[-1]
        assert first[1]["type"] == "session.updated"
        await answer_until_held(client, mocks, "你好。我们开始吧。")
        clock.advance(played_s)
        await client.send({"type": "response.cancel"})
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        await client.say("继续。")
        await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
        await client.until("response.done")
    assistant = [m.content for m in mocks.llm.requests[-1].messages if m.role == "assistant"]
    assert assistant == heard


async def test_a_truncate_beyond_the_g711_audio_sent_is_clamped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="mlx_realtime.session.actor")
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=100, clock=ManualClock())
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL_G711)
        item_id, sent = await answer_until_held(client, mocks, "你好。我们开始吧。")
        await client.send({"type": "response.cancel"})
        await client.until("response.done")
        mocks.tts.next_chunk.release()
        await client.send(
            {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": 99_999,
            }
        )
        truncated = (await client.until("conversation.item.truncated"))[-1]
        await client.quiet()
    # The first segment whole (300 ms), and what the 24 → 8 kHz resampler had let out of the
    # second's first 100 ms chunk when the TTS stopped (61 ms at HQ).
    assert sent == 2400 + 490
    assert truncated["audio_end_ms"] == sent // 8
    assert caplog.records[0].__dict__["sent_ms"] == sent // 8


async def test_pacing_counts_8_khz_audio() -> None:
    gateway, mocks = build_gateway(
        {"output": {"max_delta_ms": 100, "max_lead_ms": 400}}, ms_per_char=100, chunk_ms=100
    )
    text = "这是一个用来测试输出节奏的比较长的回答。"  # 20 characters: 2 s of audio
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL_G711)
        await client.say(text)
        await client.send({"type": "response.create"})
        await client.until("response.done")
    timed = [
        (at, len(base64.b64decode(event["delta"])))
        for at, direction, event in client.timeline
        if direction == "server" and event["type"] == "response.output_audio.delta"
    ]
    start = timed[0][0]
    sent_ms = 0.0
    for at, size in timed:
        assert size <= 800
        sent_ms += size / 8
        lead = sent_ms - (at - start) * 1000
        assert lead <= 400 + 100 + 50, lead  # the cap, one delta, scheduling slack
    assert sent_ms == mocks.tts.samples_for(text) / 24
    assert timed[-1][0] - start >= (2000 - 400 - 100) / 1000  # it took real time


async def test_a_barge_in_flushes_g711_audio_held_back_by_pacing() -> None:
    gateway, mocks = build_gateway(
        {"output": {"max_lead_ms": 300}}, speech=[(1000, 1800)], ms_per_char=100, chunk_ms=100
    )
    text = "请讲一个很长很长的故事，越长越好，一直讲下去不要停。"  # 26 characters: 2.6 s
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(G711_SESSION)
        await client.say(text)
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await client.append(g711_tone(1300), bytes_per_ms=8)
        interrupted = await client.until("response.done")
    assert interrupted[-1]["response"]["status"] == "cancelled"
    heard_ms = sum(len(delta) for delta in deltas(client.events)) / 8
    assert heard_ms < mocks.tts.samples_for(text) / 24 - 1000  # the paced rest was never sent
    assert_causal(client.events)


async def test_the_response_audio_limit_counts_8_khz_audio() -> None:
    gateway, _ = build_gateway({"limits": {"max_response_audio_ms": 300}}, ms_per_char=100)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL_G711)
        await client.say("这是一个很长的回答，它会超过上限。")
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]["response"]
    assert done["status_details"] == {"type": "incomplete", "reason": "max_output_tokens"}
    # Stopped once 300 ms of 8 kHz audio was made (the deltas still queued then are dropped);
    # bytes counted at 24 kHz PCM's rate would have let six times as much through.
    sent_ms = sum(len(delta) for delta in deltas(client.events)) / 8
    assert 0 < sent_ms <= 400


async def test_formats_can_change_after_audio_has_flowed() -> None:
    """GA lets a session.update change the formats at any time: later appends decode in the new
    input format (the old decoder's tail goes onto the timeline first, so nothing is lost and
    the milliseconds stay exact), later responses speak the new output format, and a
    response.create may set its own."""
    gateway, mocks = build_gateway({"aec": {"enabled": False}})
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"audio": {"input": {"turn_detection": None}}})
        await client.append(tone(500))
        await client.send(
            {
                "type": "session.update",
                "session": {"type": "realtime", **MANUAL_G711},
            }
        )
        updated = (await client.until("session.updated"))[-1]["session"]
        await client.append(g711_tone(500, "ulaw"), bytes_per_ms=8)
        await client.send({"type": "input_audio_buffer.commit"})
        await client.until("conversation.item.input_audio_transcription.completed")
        assert mocks.asr.requests == [1000]  # 500 ms of each, end to end
        assert updated["audio"]["output"]["format"] == {"type": "audio/pcmu"}

        await client.send({"type": "response.create"})
        first = await client.until("response.done")
        pcma = {"audio": {"output": {"format": {"type": "audio/pcma"}}}}
        await client.send({"type": "response.create", "response": pcma})
        second = await client.until("response.done")
        back = {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}}
        await client.send({"type": "session.update", "session": {"type": "realtime", **back}})
        await client.until("session.updated")
        await client.send({"type": "response.create"})
        third = await client.until("response.done")
    formats = [
        events[-1]["response"]["audio"]["output"]["format"] for events in (first, second, third)
    ]
    assert formats == [
        {"type": "audio/pcmu"},
        {"type": "audio/pcma"},
        {"type": "audio/pcm", "rate": 24000},
    ]
    sizes = [sum(len(delta) for delta in deltas(events)) for events in (first, second, third)]
    assert sizes[0] == sizes[1] == sizes[2] // 6  # the same answer: 8 kHz × 1 byte, 24 kHz × 2
    assert dominant_hz(b"".join(deltas(first)), "ulaw") == pytest.approx(440, abs=10)
    assert dominant_hz(b"".join(deltas(second)), "alaw") == pytest.approx(440, abs=10)
    assert "error" not in [event["type"] for event in client.events]


async def test_any_g711_byte_count_is_accepted_and_half_a_pcm_sample_is_not() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL_G711)
        for size in (1, 3, 159, 160, 801):
            audio = base64.b64encode(bytes([0xFF]) * size).decode()
            await client.send({"type": "input_audio_buffer.append", "audio": audio})
        await client.send({"type": "input_audio_buffer.commit"})
        await client.until("conversation.item.done")
        await client.quiet()
        pcm = {"audio": {"input": {"format": {"type": "audio/pcm"}}}}
        await client.send({"type": "session.update", "session": {"type": "realtime", **pcm}})
        await client.until("session.updated")
        odd = base64.b64encode(b"\x00\x00\x00").decode()
        await client.send({"type": "input_audio_buffer.append", "audio": odd, "event_id": "odd"})
        error = (await client.until("error"))[-1]["error"]
    assert (error["code"], error["param"], error["event_id"]) == ("invalid_value", "audio", "odd")
    assert mocks.asr.requests == [(1 + 3 + 159 + 160 + 801) // 8]


async def test_g711_session_defaults_serve_telephony_from_the_first_event() -> None:
    """A deployment can make G.711 the default: ``session.created`` echoes it, and appends are
    taken as μ-law before any session.update."""
    defaults = {"audio": {"input": {"format": {"type": "audio/pcmu"}, "turn_detection": None}}}
    gateway, mocks = build_gateway({"session_defaults": defaults})
    async with running(gateway) as address, Client.connect(address) as client:
        created = (await client.until("session.created"))[-1]["session"]
        await client.until("conversation.created")
        await client.append(g711_tone(400), bytes_per_ms=8)
        await client.send({"type": "input_audio_buffer.commit"})
        await client.until("conversation.item.input_audio_transcription.completed")
    assert created["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert created["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert mocks.asr.requests == [400]

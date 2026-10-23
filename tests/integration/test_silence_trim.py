"""Edge-silence trimming on the wire (M0 S3): a TTS that pads every segment with silence before
(460 ms here) and 500 ms after, as Qwen3-TTS does. The client receives each segment with at
most 60 ms of lead and 100 ms of tail, and the segment map that truncation reads counts exactly
the audio sent."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator

from mlx_realtime.providers.base import Lang, PCMChunk, SpeechTiming
from tests.support import Client, ManualClock, Mocks, build_gateway, running

MANUAL = {"audio": {"input": {"turn_detection": None}}}
ANSWER = "你好。我们开始吧。"  # echoed: 300 ms, then 600 ms of tone (100 ms per character)
RATE = 24_000


def pad_every_segment(mocks: Mocks, lead_ms: int = 460, tail_ms: int = 500) -> None:
    """Silence on the 20 ms grid, so the expected lengths are exact."""
    sine = mocks.tts.stream

    async def padded(
        text: str, voice: str, lang: Lang, *, speed: float = 1.0, timing: SpeechTiming | None = None
    ) -> AsyncGenerator[PCMChunk]:
        for _ in range(lead_ms // 20):
            yield PCMChunk(bytes(RATE // 50 * 2), RATE)  # 20 ms of silence per chunk
        async for chunk in sine(text, voice, lang, speed=speed, timing=timing):
            yield chunk
        for _ in range(tail_ms // 20):
            yield PCMChunk(bytes(RATE // 50 * 2), RATE)

    mocks.tts.stream = padded  # type: ignore[method-assign]


def audio_ms(events: list[dict[str, object]]) -> float:
    return sum(
        len(base64.b64decode(str(event["delta"]))) / 48
        for event in events
        if event["type"] == "response.output_audio.delta"
    )


async def test_segments_arrive_trimmed_and_truncation_counts_what_was_sent() -> None:
    gateway, mocks = build_gateway(ms_per_char=100, chunk_ms=50, clock=ManualClock())
    pad_every_segment(mocks)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say(ANSWER)
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
        item_id = events[-1]["response"]["output"][0]["id"]
        # two segments: (60 + 300 + 100) + (60 + 600 + 100) ms
        assert audio_ms(events) == 460 + 760
        cut = {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0}
        await client.send({**cut, "audio_end_ms": 460})  # the end of the first segment
        await client.until("conversation.item.truncated")
        await client.say("继续。")
        await client.send({"type": "response.create", "response": {"output_modalities": ["text"]}})
        await client.until("response.done")
        heard = [m.content for m in mocks.llm.requests[-1].messages if m.role == "assistant"]
        assert heard == ["你好。"]
        await client.send({**cut, "audio_end_ms": 1220})  # all of it
        truncated = (await client.until("conversation.item.truncated"))[-1]
        assert truncated["audio_end_ms"] == 1220  # not clamped: exactly what was sent


async def test_trimming_can_be_switched_off() -> None:
    config = {"output": {"trim_silence": {"enabled": False}}}
    gateway, mocks = build_gateway(config, ms_per_char=100, chunk_ms=50)
    pad_every_segment(mocks)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await client.say("你好。")
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    assert audio_ms(events) == 460 + 300 + 500

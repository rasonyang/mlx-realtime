"""Spoken turns end to end on fake backends, audio out included (M3): client audio → Silero VAD and
Smart Turn → the ASR, LLM and TTS providers (real, built from the configuration) against fake
mlx_audio and mlx_lm servers → GA events. AICC's deadlines are checked on every session
(``deadlines.py``), and pacing (D6) is measured on the wire."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedTurnEnd
from tests.fakes.mlx_audio import (
    Chunks,
    Fail,
    FakeMlxAudio,
    Gate,
    Hold,
    Rest,
    Transcribe,
    serve_asgi,
)
from tests.fakes.mlx_lm import Done, FakeMlxLm, Finish, Sleep, Text, Usage, reply
from tests.integration.deadlines import check_deadlines
from tests.integration.invariants import assert_causal
from tests.phone import SimulatedCall
from tests.speech import EN, SILERO, ZH, clip, needs_models, silence, to_wire
from tests.support import Client, Event, build_gateway, running

ZH_TEXT = "但鸟类还是有很多地方跟恐龙相似。"
KOKORO = "mlx-community/Kokoro-82M-bf16"
AICC_SESSION = {  # what AICC's gateway profile sends (spikes/aicc/traces)
    "audio": {
        "input": {
            "format": {"rate": 24000, "type": "audio/pcm"},
            "turn_detection": {"silence_duration_ms": 500, "type": "server_vad"},
        },
        "output": {"format": {"rate": 24000, "type": "audio/pcm"}},
    },
    "instructions": "You answer the phone for NovaNet.",
}


@dataclass
class Stack:
    gateway: Gateway
    address: str
    llm: FakeMlxLm
    asr: FakeMlxAudio
    tts: FakeMlxAudio


@pytest.fixture
def packs(tmp_path: Path) -> Path:
    """A local Kokoro snapshot: the voice packs the built-in voices fall back to."""
    voices = tmp_path / "voices"
    voices.mkdir()
    for name in ("zf_xiaobei", "zf_xiaoxiao", "zm_yunjian", "zm_yunxi", "zm_yunyang"):
        (voices / f"{name}.safetensors").write_bytes(b"pack")
    for name in ("af_heart", "af_bella", "am_adam", "am_michael", "am_puck"):
        (voices / f"{name}.safetensors").write_bytes(b"pack")
    return tmp_path


@asynccontextmanager
async def real_stack(
    packs: Path,
    *,
    tts: Mapping[str, object] | None = None,
    turn_end: ScriptedTurnEnd | None = None,
    output: Mapping[str, object] | None = None,
) -> AsyncIterator[Stack]:
    """The gateway as config.yaml builds it, its backends faked; the VAD is real Silero, the end
    of turn scripted (complete) unless given. The echo canceller and the barge-in rules are on,
    as configured by default (M4)."""
    llm, asr, speech = FakeMlxLm(), FakeMlxAudio(), FakeMlxAudio()
    with llm.serving() as llm_url:
        async with serve_asgi(asr) as asr_url, serve_asgi(speech) as tts_url:
            tts_config: dict[str, object] = {"type": "mlx_audio_speech", "base_url": tts_url}
            tts_config["fallback"] = {
                "model": KOKORO,
                "family": "kokoro",
                "voices_dir": str(packs / "voices"),
            }
            tts_config.update(tts or {})
            config = validate_config(
                {
                    "limits": {"inbox_max_ms": 60_000},  # audio sent faster than real time
                    "providers": {
                        "llm": {"type": "mlx_lm_chat", "base_url": llm_url},
                        "asr": {"type": "mlx_audio_transcription", "base_url": asr_url},
                        "tts": tts_config,
                    },
                    "turn": {"vad": {"type": "silero", "model": str(SILERO)}},
                    "output": dict(output or {}),
                }
            )
            verdicts = turn_end or ScriptedTurnEnd(default=0.9)
            gateway = Gateway(config, turn_end_detector=lambda: verdicts)
            async with running(gateway) as address:
                speech.first_chunk_s, speech.chunk_s = 0.11, 0.09  # M0's pace, after warmup
                yield Stack(gateway, address, llm, asr, speech)


def kinds(events: list[Event]) -> list[str]:
    return [event["type"] for event in events]


def audio_bytes(events: list[Event]) -> int:
    return sum(
        len(base64.b64decode(event["delta"]))
        for event in events
        if event["type"] == "response.output_audio.delta"
    )


@needs_models
async def test_a_spoken_chinese_turn_answered_with_speech(
    packs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mlx_realtime.session.timeline")
    async with real_stack(packs) as stack, Client.connect(stack.address) as client:
        await client.start(AICC_SESSION)
        stack.asr.script(Transcribe(ZH_TEXT))
        stack.llm.script(*reply("是的，", "鸟类是恐龙的后代。", "还有什么想了解的吗？"))
        await client.append(to_wire(silence(500) + clip(ZH) + silence(1500)), chunk_ms=20)
        events = await client.until("response.done", wait_s=15)
        await client.quiet(0.3)
    types = kinds(events)
    for kind in ("speech_started", "speech_stopped", "committed"):
        assert types.count(f"input_audio_buffer.{kind}") == 1
    assert types.count("response.created") == 1  # one reply per speech turn
    done = events[-1]["response"]
    assert done["status"] == "completed"
    transcript = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    )
    assert transcript == "是的，鸟类是恐龙的后代。还有什么想了解的吗？"
    assert done["output"][0]["content"][0]["transcript"] == transcript
    spoken = [speech.body["input"] for speech in stack.tts.speeches[-2:]]
    assert "".join(spoken) == transcript.replace(" ", "")  # one request per segment
    assert audio_bytes(events) == sum(speech.audio_bytes for speech in stack.tts.speeches[-2:])
    assert {speech.body["voice"] for speech in stack.tts.speeches[-2:]} == {"vivian"}
    assert {speech.body["lang_code"] for speech in stack.tts.speeches[-2:]} == {"chinese"}
    assert_causal(client.events)
    report = check_deadlines(client.timeline)
    assert report.audio_responses == 1
    (record,) = [r for r in caplog.records if r.getMessage() == "turn timeline"]
    marks: dict[str, int] = record.__dict__["marks_ms"]
    assert list(marks) == [
        "speech_end", "speech_stopped", "turn_confirmed", "response_created", "asr_done",
        "llm_first_token", "first_sentence_ready", "tts_first_chunk", "first_audio_sent",
    ]  # fmt: skip
    assert marks["first_audio_sent"] >= marks["tts_first_chunk"] >= marks["first_sentence_ready"]
    assert record.__dict__["tts_queue_ms"] >= 0
    assert record.__dict__["tts_model"] == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"


@needs_models
async def test_the_fallback_speaks_a_segment_the_primary_failed(
    packs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mlx_realtime.session.timeline")
    async with real_stack(packs) as stack, Client.connect(stack.address) as client:
        await client.start({"audio": {"input": {"turn_detection": None}}})
        stack.llm.script(*reply("您好，", "请问有什么可以帮您？"))
        stack.tts.script_speech(Fail(500, "Failed to load model"))
        await client.say("你好")
        await client.send({"type": "response.create"})
        events = await client.until("response.done", wait_s=10)
    assert events[-1]["response"]["status"] == "completed"
    failed, fallback = stack.tts.speeches[-2:]
    assert failed.ended == "error"
    assert fallback.body["model"] == KOKORO
    assert fallback.body["voice"] == str(packs / "voices" / "zf_xiaobei.safetensors")
    assert fallback.body["lang_code"] == "z"
    assert audio_bytes(events) == fallback.audio_bytes
    (record,) = [r for r in caplog.records if r.getMessage() == "turn timeline"]
    assert record.__dict__["tts_fallback"] is True
    check_deadlines(client.timeline)


@needs_models
async def test_sessions_queue_for_the_tts_backend_and_the_wait_is_logged(
    packs: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mlx_realtime.session.timeline")
    async with real_stack(packs) as stack:
        gate = Gate()
        stack.tts.script_speech(Chunks(1), Hold(gate), Rest())
        stack.llm.script(*reply("第一位来电者，您好。"))
        stack.llm.script(*reply("第二位来电者，您好。"))
        manual = {"audio": {"input": {"turn_detection": None}}}
        async with Client.connect(stack.address) as first, Client.connect(stack.address) as second:
            await first.start(manual)
            await second.start(manual)
            await first.say("你好")
            await second.say("你好")
            await first.send({"type": "response.create"})
            await first.until("response.output_audio.delta")
            await gate.reached()
            await second.send({"type": "response.create"})
            await second.until("response.created")
            await asyncio.sleep(0.3)  # the second session's request waits for the first one
            gate.release()
            await first.until("response.done", wait_s=10)
            await second.until("response.done", wait_s=10)
    waits = sorted(
        r.__dict__["tts_queue_ms"] for r in caplog.records if r.getMessage() == "turn timeline"
    )
    assert waits[0] < 100
    assert waits[1] >= 300


async def test_pacing_keeps_the_audio_at_most_max_lead_ms_ahead() -> None:
    gateway, mocks = build_gateway(
        {"output": {"max_delta_ms": 100, "max_lead_ms": 400}}, ms_per_char=100, chunk_ms=100
    )
    text = "这是一个用来测试输出节奏的比较长的回答。"  # 20 characters: 2 s of audio
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({"audio": {"input": {"turn_detection": None}}})
        await client.say(text)
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    deltas = [
        (at, len(base64.b64decode(event["delta"])))
        for at, direction, event in client.timeline
        if direction == "server" and event["type"] == "response.output_audio.delta"
    ]
    start = deltas[0][0]
    sent_ms = 0.0
    for at, size in deltas:
        sent_ms += size / 48
        lead = sent_ms - (at - start) * 1000
        assert lead <= 400 + 100 + 50, lead  # the cap, one delta, scheduling slack
    assert sent_ms == mocks.tts.samples_for(text) / 24
    assert deltas[-1][0] - start >= (2000 - 400 - 100) / 1000  # it took real time
    report = check_deadlines(client.timeline)
    assert report.max_gap_s < 0.5  # steady: one delta per 100 ms at the cap
    assert events[-1]["response"]["status"] == "completed"


async def test_a_barge_in_flushes_the_audio_held_back_by_pacing() -> None:
    # The scripted detector hears speech from 1000 ms to 1800 ms of input audio.
    gateway, mocks = build_gateway(
        {"output": {"max_lead_ms": 300}}, speech=[(1000, 1800)], ms_per_char=100, chunk_ms=100
    )
    text = "请讲一个很长很长的故事，越长越好，一直讲下去不要停。"  # 26 characters: 2.6 s
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await client.say(text)
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await client.append(b"\x00\x00" * 24 * 1300)  # speech confirmed at about 1150 ms
        interrupted = await client.until("response.done")
    assert interrupted[-1]["response"]["status"] == "cancelled"
    heard = audio_bytes(client.events) / 48
    synthesized = mocks.tts.samples_for(text) / 24
    assert heard < synthesized - 1000  # the paced remainder was never sent
    assert mocks.tts.streams.active == 0
    assert_causal(client.events)


ANSWER = [  # three long segments at once, then more, slowly
    "您好，我来为您详细介绍一下我们公司目前正在推出的几种宽带套餐和它们各自的特点。",
    "第一种是家庭基础套餐，每个月一百兆的带宽，适合日常上网和看视频的家庭用户使用。",
    "第二种是家庭高速套餐，每个月五百兆的带宽，还送一张副卡，适合人多的家庭一起使用。",
]
TERMINAL = {
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.content_part.done",
    "response.output_item.done",
    "conversation.item.done",
}


@needs_models
async def test_a_spoken_barge_in_runs_the_cancellation_handler_end_to_end(packs: Path) -> None:
    """ARCHITECTURE §6.5 steps 1-7 on the real TTS path, from real speech: a caller (AICC,
    simulated in real time, no echo) talks over a long answer that the LLM is still writing and
    the pacer is still holding back (1 s lead)."""
    async with (
        real_stack(packs, output={"max_lead_ms": 1000}) as stack,
        Client.connect(stack.address) as client,
    ):
        await client.start(AICC_SESSION)
        stack.tts.voice = to_wire(clip(EN))  # the TTS speaks (the canceller needs speech)
        stack.llm.script(  # the LLM is still writing (in 2 s steps, inside its idle deadline)
            *(Text(text) for text in ANSWER),
            *(step for _ in range(4) for step in (Sleep(2.0), Text("还有，"))),
            Finish("stop"),
            Usage(),
            Done(),
        )
        stack.asr.script(Transcribe("等一下，我想问问价格。"))
        stack.llm.script(*reply("好的，", "价格是每月九十九元。"))
        async with SimulatedCall(client, echo_db=None) as call:
            await asyncio.sleep(0.3)
            item = {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "介绍一下套餐。"}],
            }
            await client.send({"type": "conversation.item.create", "item": item})
            await client.send({"type": "response.create"})
            first_audio = await call.until("response.output_audio.delta")
            answer = call.events[first_audio][1]["item_id"]
            await asyncio.sleep(2.6)  # past the 2 s echo guard of the answer's playback
            said_at = asyncio.get_running_loop().time()
            call.say(clip(ZH))  # speech from 0.32 s
            started = await call.until("input_audio_buffer.speech_started", after=first_audio)
            # 400 ms while playing, the canceller's hold-back, framing: well under 2 s.
            assert call.events[started][0] - said_at < 2.0
            done = await call.until("response.done", after=started)
            committed = await call.until("input_audio_buffer.committed", after=done)
            reply_done = await call.until("response.done", after=committed, timeout_s=15)
        events = [event for _, event in call.events]
        await stack.llm.disconnected()  # step 2: the LLM request was closed mid-answer
    # Step 1: speech_started first, naming the item the commit then created.
    speech = events[started]
    assert speech["audio_start_ms"] >= 0
    assert events[committed]["item_id"] == speech["item_id"]
    # Steps 2-6 ran in one handler: only the answer's terminal events lie in between.
    between = [event["type"] for event in events[started + 1 : done]]
    assert set(between) <= TERMINAL, between
    assert "conversation.item.done" in between
    response = events[done]["response"]
    assert response["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    (message,) = response["output"]
    assert message["status"] == "incomplete"
    # Step 4: audio the pacer still held was never sent; nothing of the answer came after.
    sent = sum(
        len(base64.b64decode(event["delta"]))
        for event in events
        if event["type"] == "response.output_audio.delta" and event["item_id"] == answer
    )
    synthesized = sum(speech.audio_bytes for speech in stack.tts.speeches[-4:-1])
    assert 0 < sent < synthesized
    assert all(
        event.get("item_id") != answer
        or event["type"] in TERMINAL | {"conversation.item.truncated"}
        for event in events[done:]
    )
    # Steps 5 and 7: AICC truncated the answer to what it played; the next request saw that.
    (truncated,) = [event for event in events if event["type"] == "conversation.item.truncated"]
    assert truncated["item_id"] == answer
    played = call.log.truncates[0][1]
    assert truncated["audio_end_ms"] == min(played, sent // 48)
    history = [
        m.get("content") for m in stack.llm.requests[-1]["messages"] if m["role"] == "assistant"
    ]
    transcript = message["content"][0]["transcript"]
    assert len(history) == 1
    assert transcript.startswith(history[0])
    assert history[0] != transcript  # cut to what was played
    assert events[reply_done]["response"]["status"] == "completed"
    assert "error" not in [event["type"] for event in events]
    assert_causal(events)

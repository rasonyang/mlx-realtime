"""Speech turns end to end, on fake backends: client audio at 24 kHz → Silero VAD and Smart Turn
on their executors → the ASR and LLM providers (real, built from the configuration) against fake
mlx_audio and mlx_lm servers → GA events (ARCHITECTURE §6.2, D4, D7, D15)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedTurnEnd
from tests.fakes.mlx_audio import FakeMlxAudio, Transcribe, serve_asgi
from tests.fakes.mlx_lm import FakeMlxLm, Gate, Hold, Text, reply
from tests.integration.invariants import assert_causal
from tests.speech import EN, SILERO, SMART_TURN, ZH, clip, needs_models, silence, to_wire
from tests.support import Client, Event, running

pytestmark = needs_models

TEXT = {"output_modalities": ["text"]}
ZH_TEXT = "但鸟类还是有很多地方跟恐龙相似。"
SERVER_VAD_TURN = [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.added",
    "response.created",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.done",
    "response.output_item.added",
    "conversation.item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "conversation.item.done",
    "response.done",
]


@dataclass
class Stack:
    client: Client
    llm: FakeMlxLm
    asr: FakeMlxAudio


@asynccontextmanager
async def speech_stack(
    *, turn_end: ScriptedTurnEnd | None = None, max_silence_ms: int = 3000
) -> AsyncIterator[Stack]:
    """The gateway as config.yaml builds it, its backends faked. ``turn_end`` scripts Smart
    Turn's verdicts; without it the real model decides."""
    llm, asr = FakeMlxLm(), FakeMlxAudio()
    with llm.serving() as llm_url:
        async with serve_asgi(asr) as asr_url:
            config = validate_config(
                {
                    # The test client sends seconds of audio at once, not in real time.
                    "limits": {"inbox_max_ms": 60_000},
                    "providers": {
                        "llm": {"type": "mlx_lm_chat", "base_url": llm_url},
                        "asr": {"type": "mlx_audio_transcription", "base_url": asr_url},
                    },
                    "turn": {
                        "vad": {"type": "silero", "model": str(SILERO)},
                        "smart_turn": {
                            "enabled": True,
                            "model": str(SMART_TURN),
                            "max_silence_ms": max_silence_ms,
                        },
                    },
                    "session_defaults": {
                        "audio": {"input": {"transcription": {"model": "qwen3-asr"}}}
                    },
                }
            )
            scripted = None if turn_end is None else lambda: turn_end
            gateway = Gateway(config, turn_end_detector=scripted)
            async with running(gateway) as address, Client.connect(address) as client:
                await client.start(TEXT)
                yield Stack(client, llm, asr)


def subsequence(events: Sequence[Event], types: Sequence[str]) -> bool:
    remaining = iter(event["type"] for event in events)
    return all(wanted in remaining for wanted in types)


def kinds(events: Sequence[Event]) -> list[str]:
    return [event["type"] for event in events]


async def test_a_server_vad_turn_from_24k_audio(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="mlx_realtime.session.timeline")
    async with speech_stack() as stack:
        stack.asr.script(Transcribe(ZH_TEXT))
        stack.llm.script(*reply("鸟类", "确实是恐龙的后代。"))
        await stack.client.append(to_wire(silence(500) + clip(ZH) + silence(3500)), chunk_ms=20)
        events = await stack.client.until("response.done")
    assert subsequence(events, SERVER_VAD_TURN)
    assert kinds(events).count("input_audio_buffer.speech_started") == 1
    started, stopped = (
        next(e for e in events if e["type"] == kind) for kind in SERVER_VAD_TURN[:2]
    )
    # Speech in the fixture runs 0.32-3.66 s, after 0.5 s of silence; prefix padding is 300 ms.
    assert 480 <= started["audio_start_ms"] <= 600
    assert 4100 <= stopped["audio_end_ms"] <= 4250
    transcript = next(e for e in events if e["type"].endswith("transcription.completed"))
    assert transcript["transcript"] == ZH_TEXT  # transcription is on by default (D15)
    done = events[-1]["response"]
    assert done["status"] == "completed"
    assert done["output"][0]["content"][0]["text"] == "鸟类确实是恐龙的后代。"
    request = stack.llm.requests[-1]
    assert request["messages"][-1] == {"role": "user", "content": ZH_TEXT}
    upload = stack.asr.uploads[-1]
    rate, channels, width, frames = upload.audio()
    assert (rate, channels, width) == (16000, 1, 2)
    assert abs(len(frames) / 32 - (stopped["audio_end_ms"] - started["audio_start_ms"])) <= 1
    assert_causal(stack.client.events)
    (timeline,) = [r for r in caplog.records if r.getMessage() == "turn timeline"]
    marks: dict[str, int] = timeline.__dict__["marks_ms"]
    assert list(marks) == [
        "speech_end",
        "speech_stopped",
        "turn_confirmed",
        "response_created",
        "asr_done",
        "llm_first_token",
    ]  # fmt: skip (text modality: no TTS marks)


async def test_an_incomplete_verdict_holds_the_turn_until_speech_resumes() -> None:
    verdicts = ScriptedTurnEnd([0.1], default=0.9)
    async with speech_stack(turn_end=verdicts) as stack:
        audio = silence(500) + clip(ZH) + silence(700) + clip(EN) + silence(1500)
        await stack.client.append(to_wire(audio), chunk_ms=20)
        events = await stack.client.until("response.done")
    assert subsequence(events, SERVER_VAD_TURN)
    types = kinds(events)
    assert types.count("input_audio_buffer.speech_started") == 1  # one turn for both sentences
    assert types.count("input_audio_buffer.speech_stopped") == 1
    assert types.count("input_audio_buffer.committed") == 1
    assert len(verdicts.evaluated) == 2  # incomplete after the first sentence, complete after
    stopped = next(e for e in events if e["type"] == "input_audio_buffer.speech_stopped")
    assert stopped["audio_end_ms"] > 500 + 3960 + 700  # the end of the second sentence
    # The second evaluation saw the whole turn so far: both sentences and the pause.
    assert verdicts.evaluated[1] > verdicts.evaluated[0] + 2 * 16000 * 4
    rate, _, _, frames = stack.asr.uploads[-1].audio()
    assert len(frames) / 2 / rate > 8.5


async def test_a_turn_held_incomplete_commits_after_max_silence() -> None:
    verdicts = ScriptedTurnEnd(default=0.1)  # never complete
    async with speech_stack(turn_end=verdicts, max_silence_ms=1000) as stack:
        await stack.client.append(to_wire(silence(500) + clip(ZH) + silence(1500)), chunk_ms=20)
        events = await stack.client.until("response.done")
    assert subsequence(events, SERVER_VAD_TURN)
    assert len(verdicts.evaluated) == 1
    stopped = next(e for e in events if e["type"] == "input_audio_buffer.speech_stopped")
    assert 4100 <= stopped["audio_end_ms"] <= 4250  # still the acoustic end of speech
    assert events[-1]["response"]["status"] == "completed"


async def test_an_empty_transcript_completes_the_reply_without_the_llm() -> None:
    async with speech_stack() as stack:
        stack.asr.script(Transcribe("", ()))
        warmup_requests = len(stack.llm.requests)
        await stack.client.append(to_wire(silence(500) + clip(ZH) + silence(3500)), chunk_ms=20)
        events = await stack.client.until("response.done")
        await stack.client.quiet(0.2)
    assert kinds(events)[-4:] == [
        "response.created",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.done",
        "response.done",
    ]
    done = events[-1]["response"]
    assert (done["status"], done["output"]) == ("completed", [])
    assert len(stack.llm.requests) == warmup_requests  # no LLM call (D7)


async def test_barge_in_during_a_response_with_the_real_vad() -> None:
    gate = Gate()
    async with speech_stack() as stack:
        stack.llm.script(Text("从前有一座山，"), Hold(gate), *reply("x"))
        stack.asr.script(Transcribe(ZH_TEXT))
        await stack.client.say("讲个故事。")
        await stack.client.send({"type": "response.create"})
        await stack.client.until("response.output_text.delta")
        await gate.reached()
        await stack.client.append(to_wire(silence(500) + clip(ZH) + silence(3500)), chunk_ms=20)
        interrupted = await stack.client.until("response.done")
        await stack.llm.disconnected()  # barge-in closed the LLM request
        answered = await stack.client.until("response.done")
        gate.release()
    types = kinds(interrupted)
    assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
    cancelled = interrupted[-1]["response"]
    assert (cancelled["status"], cancelled["status_details"]["reason"]) == (
        "cancelled",
        "turn_detected",
    )
    assert subsequence(answered, SERVER_VAD_TURN[1:])
    assert answered[-1]["response"]["status"] == "completed"
    assert_causal(stack.client.events)

"""Replay of AICC's real client frames (spikes/aicc/traces, ARCHITECTURE §2, §11), on the real TTS
path: the TTS provider (Qwen3-TTS primary, Kokoro fallback) against a fake mlx_audio.server that
streams at the pace M0 measured (first chunk after about 110 ms, a 320 ms chunk every 90 ms).

Frames are sent exactly as captured, except that ids minted by AICC's fake server (``item-1``,
``call-7``, ``fc_1``, ``answer-item``, ...) are replaced by the ids this server minted for the
corresponding items. The mock LLM is scripted to make the tool calls AICC's fake server made.
No frame may draw an ``error``, each must get its documented answer, and every response must
meet AICC's deadlines (``deadlines.py``): ``response.created`` and the first audio within 3 s,
no 2 s gap between audio deltas.

Since M3 the "Say exactly this" lines are spoken verbatim without the LLM (D16), and a flow voice
that is not configured is spoken by the default voice and echoed as such (D19).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_realtime.config import MlxAudioSpeechConfig, VoicesConfig, validate_config
from mlx_realtime.providers.base import LLMDone, TextDelta, ToolCall
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR
from mlx_realtime.providers.registry import Providers, build_speech
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedSpeech
from tests.fakes.mlx_audio import Chunks, FakeMlxAudio, Gate, Hold, Rest, serve_asgi
from tests.integration.deadlines import check_deadlines
from tests.support import Client, Event, running

TRACES = Path(__file__).parents[2] / "spikes" / "aicc" / "traces"
LOOKUP = ToolCall("provider-call", "lookup_balance", '{"accountId":"42"}')
QWEN = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"


def frames(name: str) -> list[Event]:
    lines = (TRACES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@dataclass
class Session:
    client: Client
    llm: EchoLLM
    tts: FakeMlxAudio


@asynccontextmanager
async def aicc_session(packs: Path) -> AsyncIterator[Session]:
    llm, tts = EchoLLM(), FakeMlxAudio()
    async with serve_asgi(tts) as tts_url:
        speech = MlxAudioSpeechConfig.model_validate(
            {"type": "mlx_audio_speech", "base_url": tts_url}
        )
        providers = Providers(
            llm=llm,
            asr=ScriptedASR(["喂？"]),
            tts=build_speech(speech, VoiceBook(VoicesConfig()), snapshots=lambda repo: packs),
        )
        gateway = Gateway(
            validate_config({}), providers, speech_detector=lambda: ScriptedSpeech([])
        )
        async with running(gateway) as address:
            # Warm (every voice, both models) at full speed; then the pace M0 measured.
            tts.first_chunk_s, tts.chunk_s = 0.11, 0.09
            async with Client.connect(address, model="gpt-realtime") as client:
                await client.until("conversation.created")
                yield Session(client, llm, tts)
    assert [event for event in client.events if event["type"] == "error"] == []
    report = check_deadlines(client.timeline)
    assert report.responses == len(report.created_s) > 0


@pytest.fixture
def packs(tmp_path: Path) -> Path:
    """Kokoro's voice packs for the built-in voices (the fallback's warmup asks for each)."""
    voices = tmp_path / "voices"
    voices.mkdir()
    for voice in VoicesConfig().map.values():
        for name in voice.fallback.values():
            (voices / f"{name}.safetensors").write_bytes(b"pack")
    return tmp_path


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


def spoken(response: Event) -> str:
    """The transcript of a response's audio message."""
    return output_of(response, "message")["content"][0]["transcript"]


def line_of(frame: Event) -> str:
    return frame["response"]["instructions"].split("\n", 1)[1]


async def test_call_gateway(packs: Path) -> None:
    update, opening, audio, truncate, closing, keypress, create, tool_result, create_again, line = (
        frames("call_gateway")
    )
    async with aicc_session(packs) as session:
        client, llm = session.client, session.llm
        updated_session = await updated(client, update)
        assert updated_session["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 500
        assert [tool["name"] for tool in updated_session["tools"]] == ["lookup_balance"]
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
        llm.script(LOOKUP, LLMDone("tool_calls", None))
        call = output_of(await respond(client, create), "function_call")
        assert (call["name"], call["arguments"]) == ("lookup_balance", '{"accountId":"42"}')
        result = with_id(tool_result, call_id=call["call_id"])
        done = await answered(client, result, "conversation.item.done")
        assert done["item"]["call_id"] == call["call_id"]
        await respond(client, create_again)
        tool_message = llm.requests[-1].messages[-1]
        assert (tool_message.role, tool_message.tool_call_id) == ("tool", "provider-call")
        llm_requests = len(llm.requests)
        closing_line = await respond(client, line)
        assert spoken(closing_line) == line_of(line) == "Thanks for calling NovaNet."
        assert len(llm.requests) == llm_requests  # the line skipped the LLM (D16)


async def test_opening_with_a_flow_voice(packs: Path) -> None:
    update, opening = frames("opening_gateway_with_voice")
    async with aicc_session(packs) as session:
        client = session.client
        assert update["session"]["audio"]["output"]["voice"] == "flow-voice"
        accepted = await updated(client, update)  # never rejected (D19)...
        assert accepted["audio"]["output"]["voice"] == "vivian"  # ...but spoken by the default
        response = await respond(client, opening)
        assert response["audio"]["output"]["voice"] == "vivian"
        assert spoken(response) == line_of(opening) == "Good morning, NovaNet support."
        assert session.llm.requests == []  # the opening line is spoken verbatim (D16)
        bodies = [speech.body for speech in session.tts.speeches if speech.body["input"] != ""]
        assert bodies[-1]["voice"] == "vivian"
        assert bodies[-1]["model"] == QWEN
        assert bodies[-1]["input"] == "Good morning, NovaNet support."


async def test_wait_line_during_a_tool_call(packs: Path) -> None:
    update, opening, line, tool_result, delete, owed = frames("waitline_gateway")
    async with aicc_session(packs) as session:
        client, llm = session.client, session.llm
        await updated(client, update)
        llm.script(LOOKUP, LLMDone("tool_calls", None))
        first_call = output_of(await respond(client, opening), "function_call")
        # The wait line, asked for while the call is unanswered: spoken as written, without
        # the LLM, so its turn makes no second call (D16). AICC deletes only the calls a line's
        # turn made, so the trace's delete of one has nothing to name here and is not sent.
        wait_line = await respond(client, line)
        assert spoken(wait_line) == "One moment, let me check that."
        assert [item["type"] for item in wait_line["output"]] == ["message"]
        assert len(llm.requests) == 1
        assert delete["type"] == "conversation.item.delete"
        await answered(
            client, with_id(tool_result, call_id=first_call["call_id"]), "conversation.item.done"
        )
        await respond(client, owed)
        history = llm.requests[-1].messages
        assert [message.tool_calls[0].call_id for message in history if message.tool_calls] == [
            "provider-call"
        ]
        assert ("assistant", "One moment, let me check that.") in [
            (message.role, message.content) for message in history
        ]


async def test_unheard_answer_then_keypress(packs: Path) -> None:
    update, opening, tool_result, create, delete, note, cancel, truncate, keypress, owed = frames(
        "unheard_keypress_gateway"
    )
    async with aicc_session(packs) as session:
        client, llm = session.client, session.llm
        await updated(client, update)
        llm.script(LOOKUP, LLMDone("tool_calls", None))
        call = output_of(await respond(client, opening), "function_call")
        await answered(
            client, with_id(tool_result, call_id=call["call_id"]), "conversation.item.done"
        )
        answer = output_of(await respond(client, create), "message")
        await answered(client, with_id(delete, item_id=answer["id"]), "conversation.item.deleted")
        await answered(client, note, "conversation.item.done")
        # Adaptation: the trace's cancel interrupts a reply in progress; start that reply here
        # and hold its synthesis after the first chunk.
        gate = Gate()
        session.tts.script_speech(Chunks(1), Hold(gate), Rest())
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await gate.reached()
        cancelled = (await answered(client, cancel, "response.done"))["response"]
        assert cancelled["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
        await session.tts.disconnected()  # the cancel closed the TTS request
        gate.release()
        reply = output_of(cancelled, "message")["id"]
        truncated = await answered(
            client, with_id(truncate, item_id=reply), "conversation.item.truncated"
        )
        assert truncated["audio_end_ms"] <= truncate["audio_end_ms"]  # clamped to the audio sent
        await answered(client, keypress, "conversation.item.done")
        await respond(client, owed)


async def test_dropped_line_retry(packs: Path) -> None:
    update, opening, line, direction, retry = frames("dropped_line_retry_gateway")
    async with aicc_session(packs) as session:
        client = session.client
        await updated(client, update)
        await respond(client, opening)
        first = await respond(client, line)
        await answered(client, direction, "conversation.item.done")
        again = await respond(client, retry)
        assert spoken(first) == spoken(again) == "Your request is done."
        assert len(session.llm.requests) == 1  # only the opening asked the LLM


async def test_a_reply_text_streamed_over_several_segments_meets_the_deadlines(
    packs: Path,
) -> None:
    """A long LLM answer: one TTS request per segment, back to back; no gap reaches 2 s."""
    update, *_ = frames("call_gateway")
    async with aicc_session(packs) as session:
        client, llm = session.client, session.llm
        await updated(client, update)
        sentences = [
            "您好，",
            "这里是诺瓦网络客服。",
            "您的余额是十二元五角。",
            "还有什么可以帮您？",
        ]
        llm.script(*(TextDelta(text) for text in sentences), LLMDone("stop", None))
        response = await respond(client, {"type": "response.create"})
        assert spoken(response) == "".join(sentences)
        requests = [speech.body["input"] for speech in session.tts.speeches[-3:]]
        assert len(requests) == 3
    report = check_deadlines(client.timeline)
    assert report.audio_responses == 1
    assert report.max_gap_s < 1.0

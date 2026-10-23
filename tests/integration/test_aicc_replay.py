"""Replay of AICC's real client frames (spikes/aicc/traces, ARCHITECTURE §2, §11), on the real TTS
path: the TTS provider (Qwen3-TTS, with its 4-bit build as the fallback) against a fake
mlx_audio.server that streams at the pace M0 measured (S3, streaming_interval 0.16 since M5:
first chunk after about 80 ms, a 160 ms chunk every 50 ms; M3 ran 0.32 s chunks every 90 ms).

Frames are sent exactly as captured, except that ids minted by AICC's fake server (``item-1``,
``call-7``, ``fc_1``, ``answer-item``, ...) are replaced by the ids this server minted for the
corresponding items. The mock LLM is scripted to make the tool calls AICC's fake server made.
No frame may draw an ``error``, each must get its documented answer, and every response must
meet AICC's deadlines (``deadlines.py``): ``response.created`` and the first audio within 3 s,
no 2 s gap between audio deltas.

Since M3 the "Say exactly this" lines are spoken verbatim without the LLM (D16), and a flow voice
that is not configured is spoken by the default voice and echoed as such (D19).

M7 replays every trace again in AICC's G.711 session shape (``CODECS``): what a dedicated AICC
profile with ``AcceptsG711: true`` sends, ``{"type": "audio/pcmu"}`` both ways with no rate and
160-byte (20 ms) appends, and the deltas it must get back: G.711 of at most 100 ms.

M4 (AICC acceptance, part 2) adds barge-in, with AICC simulated in real time
(``tests/phone.py``: 20 ms appends, its playout model and its barge-in: on ``speech_started``
after its 800 ms guard, or on a cancelled ``response.done``, it flushes and truncates to the
frames it played): the caller talks over a long line through the real Silero VAD, the echo
canceller and the barge-in rules (all as configured by default), and the keypress flow cancels
an answer nobody heard, truncates it to 0 and deletes it.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlx_realtime.audio import g711
from mlx_realtime.audio.pcm import WIRE_RATE
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.config import MlxAudioSpeechConfig, MlxLmChatConfig, VoicesConfig, validate_config
from mlx_realtime.providers.base import LLMBackend, LLMDone, TextDelta, ToolCall
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR
from mlx_realtime.providers.registry import Providers, build_speech
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedSpeech
from tests.fakes.mlx_audio import Chunks, FakeMlxAudio, Gate, Hold, Rest, serve_asgi
from tests.fakes.mlx_lm import Calls, Done, FakeMlxLm, Finish, Text, reply, tool_call
from tests.fakes.mlx_lm import Usage as UsageChunk
from tests.integration.deadlines import check_deadlines
from tests.phone import Codec, SimulatedCall, g711_law
from tests.qwen_template import render
from tests.speech import EN, SILERO, ZH, clip, needs_models, to_wire
from tests.support import Client, Event, running

TRACES = Path(__file__).parents[2] / "spikes" / "aicc" / "traces"
LOOKUP = ToolCall("provider-call", "lookup_balance", '{"accountId":"42"}')
QWEN = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"


CODECS = ["pcm", "pcmu"]
"""The traces as captured (24 kHz PCM), and in AICC's G.711 passthrough shape (M7)."""


def frames(name: str, codec: Codec = "pcm") -> list[Event]:
    lines = (TRACES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    captured = [json.loads(line) for line in lines]
    return captured if codec == "pcm" else [g711_frame(frame, codec) for frame in captured]


def g711_frame(frame: Event, codec: Codec) -> Event:
    """``frame`` as AICC's G.711 path sends it: formats ``{"type": "audio/pcmu"}`` (no rate),
    and each 20 ms append as its 160 G.711 bytes at 8 kHz."""
    law = g711_law(codec)
    match frame["type"]:
        case "session.update" if "audio" in frame["session"]:
            audio = {
                direction: {**settings, "format": {"type": f"audio/{codec}"}}
                if "format" in settings
                else settings
                for direction, settings in frame["session"]["audio"].items()
            }
            return {**frame, "session": {**frame["session"], "audio": audio}}
        case "input_audio_buffer.append":
            resampler = Resampler(WIRE_RATE, 8000)
            pcm = base64.b64decode(frame["audio"])
            narrow = resampler.process(pcm) + resampler.flush()
            return {**frame, "audio": base64.b64encode(g711.encode(narrow, law)).decode()}
        case _:
            return frame


@dataclass
class Session:
    client: Client
    llm: EchoLLM
    tts: FakeMlxAudio


@asynccontextmanager
async def aicc_session(
    *,
    silero: bool = False,
    provider: LLMBackend | None = None,
    codec: Codec = "pcm",
) -> AsyncIterator[Session]:
    """``silero``: the real Silero VAD hears the caller (else a detector that never does).
    ``provider``: the LLM provider to use instead of the scripted mock (``Session.llm``).
    ``codec``: the session's audio format once the trace's update is applied; every response's
    audio is checked to be in it."""
    llm, tts = EchoLLM(), FakeMlxAudio()
    async with serve_asgi(tts) as tts_url:
        speech = MlxAudioSpeechConfig.model_validate(
            {"type": "mlx_audio_speech", "base_url": tts_url}
        )
        providers = Providers(
            llm=provider or llm,
            asr=ScriptedASR(["喂？"]),
            tts=build_speech(speech, VoiceBook(VoicesConfig())),
        )
        if silero:
            config = validate_config({"turn": {"vad": {"type": "silero", "model": str(SILERO)}}})
            gateway = Gateway(config, providers)
        else:
            gateway = Gateway(
                validate_config({}), providers, speech_detector=lambda: ScriptedSpeech([])
            )
        async with running(gateway) as address:
            # Warm (every voice on the primary, the default one on the fallback) at full
            # speed; then the pace M0 measured.
            tts.first_chunk_s, tts.chunk_s = 0.08, 0.05
            async with Client.connect(address, model="gpt-realtime") as client:
                await client.until("conversation.created")
                yield Session(client, llm, tts)
    assert [event for event in client.events if event["type"] == "error"] == []
    report = check_deadlines(client.timeline)
    asked = [event for _, way, event in client.timeline if way == "client"]
    requests = sum(1 for event in asked if event.get("type") == "response.create")
    assert len(report.created_s) == requests > 0  # every request answered (VAD turns add more)
    check_audio_format(client.events, codec)


def check_audio_format(events: list[Event], codec: Codec) -> None:
    """Every response speaks the session's format; deltas are at most 100 ms of it."""
    expected = (
        {"type": "audio/pcm", "rate": 24000} if codec == "pcm" else {"type": f"audio/{codec}"}
    )
    formats = [
        event["response"]["audio"]["output"]["format"]
        for event in events
        if event["type"] == "response.created"
    ]
    assert formats
    assert all(found == expected for found in formats)
    limit = 4800 if codec == "pcm" else 800
    sizes = [
        len(base64.b64decode(event["delta"]))
        for event in events
        if event["type"] == "response.output_audio.delta"
    ]
    assert all(0 < size <= limit for size in sizes)
    if codec != "pcm":
        updates = [event["session"] for event in events if event["type"] == "session.updated"]
        assert updates[0]["audio"]["input"]["format"] == expected
        assert updates[0]["audio"]["output"]["format"] == expected


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


@pytest.mark.parametrize("codec", CODECS)
async def test_call_gateway(codec: Codec) -> None:
    update, opening, audio, truncate, closing, keypress, create, tool_result, create_again, line = (
        frames("call_gateway", codec)
    )
    async with aicc_session(codec=codec) as session:
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


@pytest.mark.parametrize("codec", CODECS)
async def test_opening_with_a_flow_voice(codec: Codec) -> None:
    update, opening = frames("opening_gateway_with_voice", codec)
    async with aicc_session(codec=codec) as session:
        client = session.client
        assert update["session"]["audio"]["output"]["voice"] == "flow-voice"
        accepted = await updated(client, update)  # never rejected (D19)...
        assert accepted["audio"]["output"]["voice"] == "vivian"  # ...but spoken by the default
        response = await respond(client, opening)
        assert response["audio"]["output"]["voice"] == "vivian"
        assert spoken(response) == line_of(opening) == "Good morning, NovaNet support."
        assert session.llm.requests == []  # the opening line is spoken verbatim (D16)
        bodies = [speech.body for speech in session.tts.speeches if speech.body["input"] != ""]
        assert bodies[-1]["voice"] == "vivian"  # one voice per session
        assert bodies[-1]["model"] == QWEN
        assert bodies[-1]["input"] == "Good morning, NovaNet support."


@pytest.mark.parametrize("codec", CODECS)
async def test_wait_line_during_a_tool_call(codec: Codec) -> None:
    update, opening, line, tool_result, delete, owed = frames("waitline_gateway", codec)
    async with aicc_session(codec=codec) as session:
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


@pytest.mark.parametrize("codec", CODECS)
async def test_unheard_answer_then_keypress(codec: Codec) -> None:
    update, opening, tool_result, create, delete, note, cancel, truncate, keypress, owed = frames(
        "unheard_keypress_gateway", codec
    )
    async with aicc_session(codec=codec) as session:
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


def offered_tools(request: Event) -> list[str]:
    return [tool["function"]["name"] for tool in request.get("tools", [])]


async def test_call_gateway_through_the_llm_provider() -> None:
    """M5: the call_gateway trace with the real LLM provider on a fake mlx_lm.server that streams
    the call the way the real one does (the newline before ``<tool_call>`` in the call's chunk,
    the server's own call id). The request offers the session's tools; the result, with AICC's
    hint, follows its call in the next request, which the real Qwen template renders."""
    update, opening, audio, truncate, closing, keypress, create, tool_result, create_again, line = (
        frames("call_gateway")
    )
    fake = FakeMlxLm()
    with fake.serving() as url:
        config = MlxLmChatConfig.model_validate({"type": "mlx_lm_chat", "base_url": url})
        provider = MlxLmChatLLM(config)
        async with aicc_session(provider=provider) as session:
            fake.requests.clear()  # the warmup's
            fake.script(*reply("Hello, NovaNet support."))
            fake.script(
                Text("Let me check that."),
                Calls([tool_call("lookup_balance", {"accountId": "42"}, "6f1c-uuid")], text="\n"),
                Finish("tool_calls"),
                UsageChunk(),
                Done(),
            )
            fake.script(*reply("Your balance is 12.50.", " Anything else?"))
            client = session.client
            await updated(client, update)
            greeting = await respond(client, opening)
            await client.send(audio)
            await client.quiet()
            message = output_of(greeting, "message")["id"]
            await answered(
                client, with_id(truncate, item_id=message), "conversation.item.truncated"
            )
            await updated(client, closing)
            await answered(client, keypress, "conversation.item.done")
            asked = await respond(client, create)
            assert [item["type"] for item in asked["output"]] == ["message", "function_call"]
            assert spoken(asked) == "Let me check that."
            call = output_of(asked, "function_call")
            assert call["call_id"] != "6f1c-uuid"  # minted by the gateway
            assert call["arguments"] == '{"accountId": "42"}'
            result = with_id(tool_result, call_id=call["call_id"])
            await answered(client, result, "conversation.item.done")
            await respond(client, create_again)
            closing_line = await respond(client, line)
            assert spoken(closing_line) == "Thanks for calling NovaNet."
        await provider.aclose()
    assert len(fake.requests) == 3  # the closing line skipped the LLM
    assert all(offered_tools(request) == ["lookup_balance"] for request in fake.requests)
    messages = fake.requests[-1]["messages"]
    assert messages[0] == {"role": "system", "content": "You are now closing the call."}
    spec = {
        "id": "6f1c-uuid",
        "type": "function",
        "function": {"name": "lookup_balance", "arguments": '{"accountId": "42"}'},
    }
    assert messages[-3:] == [
        {"role": "user", "content": "(The caller pressed 2 on their keypad.)"},
        {"role": "assistant", "content": "Let me check that.", "tool_calls": [spec]},
        {
            "role": "tool",
            "tool_call_id": "6f1c-uuid",
            "content": tool_result["item"]["output"],
        },
    ]
    render(messages, fake.requests[-1]["tools"])


async def test_wait_line_through_the_llm_provider() -> None:
    """M5: AICC's wait line spoken between the call and its output. The owed request replays
    the call with the line merged into its message, and the result right after it."""
    update, opening, line, tool_result, _delete, owed = frames("waitline_gateway")
    fake = FakeMlxLm()
    with fake.serving() as url:
        config = MlxLmChatConfig.model_validate({"type": "mlx_lm_chat", "base_url": url})
        provider = MlxLmChatLLM(config)
        async with aicc_session(provider=provider) as session:
            fake.requests.clear()  # the warmup's
            fake.script(
                Calls([tool_call("lookup_balance", {"accountId": "42"}, "7a2d-uuid")]),
                Finish("tool_calls"),
                UsageChunk(),
                Done(),
            )
            fake.script(*reply("Your balance is 12.50."))
            client = session.client
            await updated(client, update)
            first = await respond(client, opening)  # a tool-only response (deadline checked)
            assert [item["type"] for item in first["output"]] == ["function_call"]
            call = output_of(first, "function_call")
            await respond(client, line)
            await answered(
                client, with_id(tool_result, call_id=call["call_id"]), "conversation.item.done"
            )
            answer = await respond(client, owed)
            assert spoken(answer) == "Your balance is 12.50."
        await provider.aclose()
    messages = fake.requests[-1]["messages"]
    assert [entry["role"] for entry in messages] == ["system", "user", "assistant", "tool"]
    assert messages[1]["content"] == config.empty_conversation_cue  # no caller turn yet
    assert messages[2]["content"] == "One moment, let me check that."
    assert [entry["id"] for entry in messages[2]["tool_calls"]] == ["7a2d-uuid"]
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "7a2d-uuid",
        "content": tool_result["item"]["output"],
    }
    render(messages, fake.requests[-1]["tools"])


@pytest.mark.parametrize("codec", CODECS)
async def test_dropped_line_retry(codec: Codec) -> None:
    update, opening, line, direction, retry = frames("dropped_line_retry_gateway", codec)
    async with aicc_session(codec=codec) as session:
        client = session.client
        await updated(client, update)
        await respond(client, opening)
        first = await respond(client, line)
        await answered(client, direction, "conversation.item.done")
        again = await respond(client, retry)
        assert spoken(first) == spoken(again) == "Your request is done."
        assert len(session.llm.requests) == 1  # only the opening asked the LLM


async def test_a_reply_text_streamed_over_several_segments_meets_the_deadlines() -> None:
    """A long LLM answer: one TTS request per segment, back to back; no gap reaches 2 s."""
    update, *_ = frames("call_gateway")
    async with aicc_session() as session:
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


SAY = "Say exactly this, word for word, and add nothing else:\n"
LONG_LINE = (
    "Thank you for calling NovaNet. Before we begin, please listen carefully, because our "
    "menu options have changed. For billing questions, stay on the line."
)  # the fake TTS speaks 150 ms per character: about 19 s, longer than the 10 s lead
TERMINAL = {
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.content_part.done",
    "response.output_item.done",
    "conversation.item.done",
}


@needs_models
@pytest.mark.parametrize("codec", CODECS)
async def test_barge_in_over_a_long_line(codec: Codec) -> None:
    """The caller talks over a long line: ``speech_started``, then ``response.done{cancelled}``
    in the same step; AICC's truncate (its played frames) is accepted; the caller's turn is
    answered after ``speech_stopped``."""
    update, *_ = frames("call_gateway", codec)
    async with aicc_session(silero=True, codec=codec) as session:
        client = session.client
        await updated(client, update)
        session.tts.voice = to_wire(clip(EN))  # the line is speech (a tone fools the canceller)
        session.tts.ms_per_char = 150
        async with SimulatedCall(client, echo_db=None, codec=codec) as call:
            await asyncio.sleep(0.3)  # AICC streams from the start of the call
            await client.send(
                {"type": "response.create", "response": {"instructions": SAY + LONG_LINE}}
            )
            first = await call.until("response.output_audio.delta")
            line = call.events[first][1]["item_id"]
            await asyncio.sleep(2.5)  # past the line's 2 s echo guard
            call.say(clip(ZH))
            started = await call.until("input_audio_buffer.speech_started", after=first)
            done = await call.until("response.done", after=started)
            truncated = await call.until("conversation.item.truncated", after=done)
            stopped = await call.until("input_audio_buffer.speech_stopped", after=started)
            reply = await call.until("response.done", after=stopped, timeout_s=15)
    events = [event for _, event in call.events]
    between = {event["type"] for event in events[started + 1 : done]}
    assert between <= TERMINAL  # one step: speech_started, the terminal events, response.done
    assert events[done]["response"]["status_details"] == {
        "type": "cancelled",
        "reason": "turn_detected",
    }
    (item, played_ms) = call.log.truncates[0]
    assert item == line
    assert events[truncated]["audio_end_ms"] == played_ms  # within the audio sent: as asked
    assert events[reply]["response"]["status"] == "completed"
    assert any(event["type"] == "response.output_audio.delta" for event in events[stopped:reply])
    assert session.llm.requests[-1].messages[-1].content == "喂？"  # the caller's turn


@needs_models
@pytest.mark.parametrize("codec", CODECS)
async def test_speech_over_the_tail_of_a_finished_line(codec: Codec) -> None:
    """AICC's most common barge-in: the line was all sent (its response done), AICC is still
    playing it, and the caller talks. There is no response to cancel: ``speech_started``, AICC
    flushes and truncates (accepted, and the client model stops playing it), and the caller's
    turn is answered."""
    update, *_ = frames("call_gateway", codec)
    async with aicc_session(silero=True, codec=codec) as session:
        client = session.client
        await updated(client, update)
        session.tts.voice = to_wire(clip(EN))
        session.tts.ms_per_char = 80  # about 7 s: all of it sent at once (10 s lead)
        async with SimulatedCall(client, echo_db=None, codec=codec) as call:
            await asyncio.sleep(0.3)
            line = {"type": "response.create", "response": {"instructions": SAY + LONG_LINE[:90]}}
            await client.send(line)
            first = await call.until("response.output_audio.delta")
            done = await call.until("response.done", after=first)
            assert call.events[done][1]["response"]["status"] == "completed"
            await asyncio.sleep(2.5)  # AICC is still playing the line
            call.say(clip(ZH))
            started = await call.until("input_audio_buffer.speech_started", after=done)
            truncated = await call.until("conversation.item.truncated", after=started)
            stopped = await call.until("input_audio_buffer.speech_stopped", after=started)
            reply = await call.until("response.done", after=stopped, timeout_s=15)
    events = [event for _, event in call.events]
    played_ms = call.log.truncates[0][1]
    assert 2500 <= played_ms < 7000
    assert events[truncated]["audio_end_ms"] == played_ms
    assert events[reply]["response"]["status"] == "completed"
    assert [
        event["response"]["status"] for event in events if event["type"] == "response.done"
    ] == [
        "completed",
        "completed",
    ]  # nothing was cancelled: the line was over on the server's side


@pytest.mark.parametrize("codec", CODECS)
async def test_keypress_cancels_an_unheard_answer_truncates_and_deletes_it(codec: Codec) -> None:
    """AICC's keypress over an answer nobody heard yet: ``response.cancel``, a truncate to 0,
    the delete of the unheard item once its response is done, the keypress note and the owed
    reply. No ``error`` at any step."""
    update, *_, keypress, owed = frames("unheard_keypress_gateway", codec)
    async with aicc_session(codec=codec) as session:
        client = session.client
        await updated(client, update)
        gate = Gate()
        session.tts.script_speech(Hold(gate), Rest())  # the answer waits for the TTS
        await client.send({"type": "response.create"})
        answer = (await client.until("response.output_item.added"))[-1]["item"]["id"]
        await gate.reached()
        await client.send({"type": "response.cancel"})
        await client.send(
            {
                "type": "conversation.item.truncate",
                "item_id": answer,
                "content_index": 0,
                "audio_end_ms": 0,
            }
        )
        cancelled = (await client.until("response.done"))[-1]["response"]
        assert cancelled["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
        assert (await client.until("conversation.item.truncated"))[-1]["audio_end_ms"] == 0
        gate.release()
        await answered(
            client,
            {"type": "conversation.item.delete", "item_id": answer},
            "conversation.item.deleted",
        )
        await answered(client, keypress, "conversation.item.done")
        reply = await respond(client, owed)
        assert spoken(reply)
        history = session.llm.requests[-1].messages
        assert all(message.role != "assistant" for message in history)  # the unheard answer: gone
        assert history[-1].content == "(The caller pressed 0 on their keypad.)"

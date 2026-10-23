"""The real model backends (marker ``backend``; skipped unless selected).

Start the stack first, then run these against it:

    make dev                                  # backends and gateway, from config.yaml
    uv run --locked pytest -m backend         # in another terminal

The addresses come from ``config.yaml`` (and ``MLXRT__`` overrides), as the gateway's own do.
The AICC harness test also needs ``AICC_REPO`` (an AICC checkout) and Go.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path

import httpx2
import numpy as np
import pytest
from openai import AsyncOpenAI
from openai.types.realtime import RealtimeServerEvent

from mlx_realtime.audio import g711
from mlx_realtime.audio.pcm import CANONICAL_RATE, WIRE_RATE
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.audio.trim import SilenceTrimmer
from mlx_realtime.config import (
    Config,
    MlxAudioAsrConfig,
    MlxAudioSpeechConfig,
    MlxLmChatConfig,
    load_config,
)
from mlx_realtime.providers.base import (
    Delta,
    LLMDone,
    Message,
    PCMChunk,
    SpeechTiming,
    TextDelta,
    ToolCall,
    ToolDef,
)
from mlx_realtime.providers.fallback import FallbackTTS
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR
from mlx_realtime.providers.mlx_audio_tts import MlxAudioTTS
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.registry import build_speech
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.text.similarity import overlap
from tests.speech import EN, ZH, clip, silence, to_wire

pytestmark = pytest.mark.backend

REPO = Path(__file__).parents[2]
WEATHER = ToolDef(
    "get_weather",
    "Get the current weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(REPO / "config.yaml", os.environ)


@pytest.fixture
async def llm(config: Config) -> AsyncGenerator[MlxLmChatLLM]:
    assert isinstance(config.providers.llm, MlxLmChatConfig), "config.yaml must use mlx_lm_chat"
    provider = MlxLmChatLLM(config.providers.llm)
    yield provider
    await provider.aclose()  # also when a test failed before closing it


@pytest.fixture
async def asr(config: Config) -> AsyncGenerator[MlxAudioASR]:
    assert isinstance(config.providers.asr, MlxAudioAsrConfig)
    provider = MlxAudioASR(config.providers.asr)
    yield provider
    await provider.aclose()  # an open connection left to the GC fails the run (ResourceWarning)


async def collect(stream: AsyncGenerator[Delta]) -> list[Delta]:
    async with aclosing(stream):
        return [delta async for delta in stream]


async def test_asr_transcribes_the_fixtures(asr: MlxAudioASR) -> None:
    await asr.warmup()
    chinese = await asr.transcribe(clip(ZH), language="zh")
    english = await asr.transcribe(clip(EN))
    await asr.aclose()
    assert "恐龙" in chinese.text
    assert chinese.language == "zh"
    assert "dinosaur" in english.text.lower()
    assert english.language == "en"


async def test_llm_streams_text_with_thinking_off(llm: MlxLmChatLLM, config: Config) -> None:
    messages = [
        Message("system", "Answer in one short sentence."),
        Message("user", "你好，你是谁？"),
    ]
    deltas = await collect(llm.stream(messages, []))
    text = "".join(delta.text for delta in deltas if isinstance(delta, TextDelta))
    done = deltas[-1]
    assert isinstance(done, LLMDone)
    assert done.finish_reason == "stop"
    assert done.usage is not None
    assert text.strip()
    assert "<think>" not in text
    # On the wire: no reasoning delta at all.
    assert isinstance(config.providers.llm, MlxLmChatConfig)
    body = {
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with (
        httpx2.AsyncClient(base_url=config.providers.llm.base_url, timeout=60) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as response,
    ):
        lines = [line async for line in response.aiter_lines() if line.startswith("data: {")]
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines]
    assert not [c for c in chunks for choice in c["choices"] if choice["delta"].get("reasoning")]
    await llm.aclose()


async def test_llm_makes_a_tool_call(llm: MlxLmChatLLM) -> None:
    messages = [
        Message("system", "Use the tools when they help."),
        Message("user", "What is the weather in Beijing right now?"),
    ]
    deltas = await collect(llm.stream(messages, [WEATHER]))
    calls = [delta for delta in deltas if isinstance(delta, ToolCall)]
    done = deltas[-1]
    await llm.aclose()
    assert isinstance(done, LLMDone)
    assert done.finish_reason == "tool_calls"
    assert [call.name for call in calls] == ["get_weather"]
    assert "beijing" in json.loads(calls[0].arguments)["city"].lower()


def aicc_line_session() -> dict[str, object]:
    """AICC's Chinese line-session fixture (instructions and its six flat tools), read from the
    AICC checkout in ``AICC_REPO`` (default: a sibling ``ai-native-callcenter``)."""
    repo = Path(os.environ.get("AICC_REPO", REPO.parent / "ai-native-callcenter"))
    path = repo / "internal" / "provider" / "testdata" / "line_session_zh.json"
    if not path.exists():
        pytest.skip(f"{path} not found: set AICC_REPO to an AICC checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def aicc_tools(session: dict[str, object]) -> list[ToolDef]:
    tools = session["tools"]
    assert isinstance(tools, list)
    return [
        ToolDef(tool["Name"], tool.get("Description"), tool.get("Parameters"))
        for tool in tools  # pyright: ignore[reportUnknownVariableType]
    ]


REPAIR_RESULT = json.dumps(
    {
        "found": 1,
        "status": "维修完成，等待取件",
        "hint": "告诉来电者手机已修好，可以到店取件，然后问还有什么需要。",
    },
    ensure_ascii=False,
)


async def test_llm_tool_round_trip_with_aiccs_tools(llm: MlxLmChatLLM) -> None:
    """M5: AICC's real instructions and tools. A clear repair number makes the model call
    ``repair_status`` with typed arguments, and no markup reaches the text; the result (with
    AICC's hint) replayed after the call gets a spoken answer, not a second call."""
    session = aicc_line_session()
    tools = aicc_tools(session)
    instructions = str(session["instructions"])
    asked = [Message("system", instructions), Message("user", "我想查维修进度，单号是RMA1001。")]
    deltas = await collect(llm.stream(asked, tools))
    calls = [delta for delta in deltas if isinstance(delta, ToolCall)]
    text = "".join(delta.text for delta in deltas if isinstance(delta, TextDelta))
    assert [call.name for call in calls] == ["repair_status"]
    assert "RMA1001" in json.loads(calls[0].arguments)["rma"].upper().replace(" ", "")
    assert "<tool_call>" not in text
    assert "<function=" not in text
    history = [
        *asked,
        Message("assistant", text, tool_calls=(calls[0],)),
        Message("tool", REPAIR_RESULT, tool_call_id=calls[0].call_id),
    ]
    answer = await collect(llm.stream(history, tools))
    await llm.aclose()
    spoken = "".join(delta.text for delta in answer if isinstance(delta, TextDelta))
    assert not [delta for delta in answer if isinstance(delta, ToolCall)]
    assert isinstance(answer[-1], LLMDone)
    assert answer[-1].finish_reason == "stop"
    assert "取" in spoken  # the result reached the model


@pytest.mark.parametrize("interlude", ["merge", "after_output"])
@pytest.mark.parametrize("unanswered", [False, True])
async def test_wait_line_history_on_the_real_template(
    config: Config, interlude: str, unanswered: bool
) -> None:
    """M5: the wait-line shape AICC produces (call → verbatim wait line → result), shaped both
    ways, and the unanswered call while the caller speaks over the line (a pending
    placeholder): the real template accepts each (no HTTP 404), and the model answers in text
    without calling the tool again."""
    session = aicc_line_session()
    tools = aicc_tools(session)
    assert isinstance(config.providers.llm, MlxLmChatConfig)
    settings = config.providers.llm.model_copy(update={"tool_interlude": interlude})
    llm = MlxLmChatLLM(settings)
    call = ToolCall("call-wait-1", "repair_status", '{"rma": "RMA1001"}')
    history = [
        Message("system", str(session["instructions"])),
        Message("user", "我想查维修进度，单号是RMA1001。"),
        Message("assistant", "好的，我查一下。", tool_calls=(call,)),
        Message("assistant", "请稍等，正在为您查询。"),  # the wait line, spoken verbatim
    ]
    if unanswered:
        history.append(Message("user", "喂？还在吗？"))
    else:
        history.append(Message("tool", REPAIR_RESULT, tool_call_id=call.call_id))
    deltas = await collect(llm.stream(history, tools))
    await llm.aclose()
    text = "".join(delta.text for delta in deltas if isinstance(delta, TextDelta))
    assert text.strip()
    assert not [d for d in deltas if isinstance(d, ToolCall) and d.name == "repair_status"]


async def test_sdk_tool_round_trip_with_the_real_stack(config: Config) -> None:
    """M5 end to end: the SDK declares AICC's tools, asks in text, gets a ``function_call`` item
    from the real LLM, returns the result and gets the answer."""
    session = aicc_line_session()
    gateway = config.gateway
    client = AsyncOpenAI(base_url=f"http://{gateway.host}:{gateway.port}/v1", api_key="local")
    tools = [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in aicc_tools(session)
    ]
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": session["instructions"],
                "tools": tools,
            }  # pyright: ignore[reportArgumentType]
        )
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "我想查维修进度，单号是RMA1001。"}],
            }
        )
        await connection.response.create()
        call = None
        started = time.monotonic()
        async with asyncio.timeout(30):
            async for event in connection:
                if event.type == "response.done":
                    calls = [i for i in event.response.output or [] if i.type == "function_call"]
                    assert len(calls) == 1
                    call = calls[0]
                    break
        assert call is not None
        assert call.name == "repair_status"
        assert call.call_id
        assert time.monotonic() - started < 3.0  # AICC's tool-only deadline (from the request)
        await connection.conversation.item.create(
            item={"type": "function_call_output", "call_id": call.call_id, "output": REPAIR_RESULT}
        )
        await connection.response.create()
        text = ""
        async with asyncio.timeout(30):
            async for event in connection:
                if event.type == "response.output_text.delta":
                    text += event.delta
                elif event.type == "response.done":
                    assert event.response.status == "completed"
                    break
    assert "取" in text


async def test_sdk_server_vad_turn_with_the_real_stack(config: Config) -> None:
    """The official SDK streams a spoken question in real time with server_vad; the gateway
    detects the turn, transcribes it and answers with text from the real LLM."""
    gateway = config.gateway
    client = AsyncOpenAI(base_url=f"http://{gateway.host}:{gateway.port}/v1", api_key="local")
    audio = to_wire(silence(500) + clip(ZH) + silence(2500))
    chunk = 960  # 20 ms at 24 kHz, as AICC sends
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(session={"type": "realtime", "output_modalities": ["text"]})
        for start in range(0, len(audio), chunk):
            encoded = base64.b64encode(audio[start : start + chunk]).decode()
            await connection.input_audio_buffer.append(audio=encoded)
            await asyncio.sleep(0.02)
        transcript, text = "", ""
        async with asyncio.timeout(30):
            async for event in connection:
                if event.type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.transcript
                elif event.type == "response.output_text.delta":
                    text += event.delta
                elif event.type == "response.done":
                    assert event.response.status == "completed"
                    break
    assert "恐龙" in transcript
    assert text.strip()


def speech(config: Config, **primary: object) -> FallbackTTS:
    """The configured TTS provider; ``primary`` replaces fields of the primary model's config."""
    tts = config.providers.tts
    assert isinstance(tts, MlxAudioSpeechConfig), "config.yaml must use mlx_audio_speech"
    if primary:
        tts = tts.model_copy(update={"primary": tts.primary.model_copy(update=primary)})
    built = build_speech(tts, VoiceBook(config.voices))
    assert isinstance(built, FallbackTTS)
    return built


async def timed(
    tts: MlxAudioTTS | FallbackTTS, text: str, voice: str, lang: str, timing: SpeechTiming
) -> tuple[float, float, bytes]:
    """(seconds to the first chunk, seconds in all, the audio) of one streamed request."""
    assert lang in ("zh", "en")
    started = time.monotonic()
    first: float | None = None
    audio = b""
    stream: AsyncGenerator[PCMChunk] = tts.stream(
        text, voice, "zh" if lang == "zh" else "en", timing=timing
    )
    async with aclosing(stream):
        async for chunk in stream:
            assert chunk.sample_rate == WIRE_RATE
            assert len(chunk.pcm) % 2 == 0
            first = time.monotonic() - started if first is None else first
            audio += chunk.pcm
    assert first is not None
    return first, time.monotonic() - started, audio


async def test_tts_warmup_loads_both_models(config: Config) -> None:
    """The gateway's TTS warmup (every voice on the primary, the default voice on the 4-bit
    fallback, D13) leaves both models loaded, as readiness requires."""
    tts = speech(config)
    started = time.monotonic()
    await tts.warmup()
    print(f"tts warmup {time.monotonic() - started:.1f} s")
    state = await tts.ready()
    await tts.aclose()
    assert state.ready, state.detail


async def test_tts_streams_with_a_fast_first_chunk_and_faster_than_real_time(
    config: Config,
) -> None:
    """Qwen3-TTS through the provider: the first chunk well inside AICC's 3 s budget, and
    synthesis faster than real time (M0: ~110 ms, RTF ~0.28 at streaming_interval 0.32)."""
    tts = speech(config)
    primary = tts.primary
    assert isinstance(primary, MlxAudioTTS)
    for text, lang in (
        ("您好，欢迎致电客服中心，请问有什么可以帮您？", "zh"),
        ("Hello, thank you for calling. How can I help you today?", "en"),
    ):
        await timed(primary, text, "vivian", lang, SpeechTiming())  # warm
        timing = SpeechTiming()
        first, total, audio = await timed(primary, text, "vivian", lang, timing)
        seconds = len(audio) / 2 / WIRE_RATE
        print(
            f"{lang}: first chunk {first * 1000:.0f} ms, RTF {total / seconds:.2f}, {seconds:.2f} s"
        )
        assert first < 0.5
        assert total / seconds < 1.0
        assert 1.0 < seconds < 15.0
        assert not timing.fallback
    await tts.aclose()


def edges_ms(pcm: bytes, threshold_dbfs: float = -50.0) -> tuple[int, int]:
    """Leading and trailing silence of ``pcm`` (24 kHz s16le), on 10 ms frames."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    frame = WIRE_RATE // 100
    frames = samples[: samples.size // frame * frame].reshape(-1, frame)
    level = 10 * np.log10(np.mean(frames**2, axis=1) / 32768.0**2 + 1e-12)
    voiced = np.flatnonzero(level >= threshold_dbfs)
    assert voiced.size, "no voiced audio"
    return int(voiced[0]) * 10, (frames.shape[0] - 1 - int(voiced[-1])) * 10


async def test_tts_edges_are_trimmed_and_the_token_cap_leaves_speech_whole(
    config: Config,
) -> None:
    """M0 S3 on the real model: each output has ~0.45 s of silence before and ~0.5 s after.
    Through the gateway's trimmer at most ~60 ms (+ a frame) is left at either end, and the
    proportional ``max_tokens`` never cuts a normal sentence short."""
    tts = speech(config)
    primary = tts.primary
    assert isinstance(primary, MlxAudioTTS)
    trim = config.output.trim_silence
    for text, lang in (
        ("您好，欢迎致电客服中心，请问有什么可以帮您？", "zh"),
        ("Hello, thank you for calling. How can I help you today?", "en"),
    ):
        _, _, raw = await timed(primary, text, "vivian", lang, SpeechTiming())
        trimmer = SilenceTrimmer(
            WIRE_RATE,
            threshold_dbfs=trim.threshold_dbfs,
            frame_ms=trim.frame_ms,
            max_lead_ms=trim.max_lead_ms,
            max_tail_ms=trim.max_tail_ms,
        )
        trimmed = trimmer.push(raw) + trimmer.flush()
        raw_lead, raw_tail = edges_ms(raw)
        lead, tail = edges_ms(trimmed)
        assert raw_lead >= 200 or raw_tail >= 200, (raw_lead, raw_tail)  # what S3 measured
        assert lead <= trim.max_lead_ms + trim.frame_ms
        assert tail <= trim.max_tail_ms + trim.frame_ms
        body = primary.request_body(text, "vivian", "zh" if lang == "zh" else "en")
        cap_s = int(body["max_tokens"]) / 12.5  # pyright: ignore[reportArgumentType]
        assert len(raw) / 2 / WIRE_RATE < cap_s * 0.8  # speech ended well before the cap
    await tts.aclose()


async def test_tts_output_is_heard_right_by_the_asr(config: Config, asr: MlxAudioASR) -> None:
    tts = speech(config)
    _, _, audio = await timed(tts, "您好，欢迎致电客服中心。", "vivian", "zh", SpeechTiming())
    resampler = Resampler(WIRE_RATE, CANONICAL_RATE)
    heard = await asr.transcribe(resampler.process(audio) + resampler.flush(), language="zh")
    await tts.aclose()
    await asr.aclose()
    assert "客服" in heard.text


MIXED = "您好，请登录 App 查看您的订单。"  # English inside Chinese: Kokoro garbled it (M0 S3)


@pytest.mark.parametrize("failure", ["unknown-model", "no-first-chunk"])
async def test_the_fallback_speaks_when_the_primary_fails(config: Config, failure: str) -> None:
    """D13: a segment the primary fails before its first chunk is spoken by the 4-bit fallback,
    in the same speaker, Latin letters and all. The primary fails as a model the server cannot
    load does (an error status before any audio), or as a stalled one does (no first chunk
    within its deadline, here 1 ms: the request is abandoned, and the server cancels it)."""
    tts = config.providers.tts
    assert isinstance(tts, MlxAudioSpeechConfig)
    if failure == "unknown-model":
        broken = speech(config, model="mlxrt-test/no-such-tts-model")
    else:
        stalled = tts.primary.timeouts.model_copy(update={"first_item_ms": 1})
        broken = speech(config, timeouts=stalled)
    await broken.fallback.warmup()  # loaded, as the gateway's warmup leaves it
    timing = SpeechTiming()
    first, total, audio = await timed(broken, MIXED, "vivian", "zh", timing)
    await broken.aclose()
    seconds = len(audio) / 2 / WIRE_RATE
    print(
        f"{failure}: primary failed ({timing.primary_error}), {timing.model} spoke; first chunk "
        f"{first * 1000:.0f} ms after the request, RTF {total / seconds:.2f}, {seconds:.2f} s"
    )
    assert timing.fallback
    assert timing.model == tts.fallback.model
    expected = {"internal", "bad_response"} if failure == "unknown-model" else {"timeout"}
    assert timing.primary_error in expected
    assert seconds > 1.0


async def test_the_4bit_fallback_speaks_chinese_english_and_mixed_text(config: Config) -> None:
    """D13 final: Qwen3-TTS 0.6B 4-bit, asked exactly like the primary (the voice's speaker,
    chinese/english, 0.16 s chunks, the token cap), streams zh, en and mixed zh/en segments
    with a fast first chunk, faster than real time."""
    tts = speech(config)
    primary, fallback = tts.primary, tts.fallback
    assert isinstance(primary, MlxAudioTTS)
    assert isinstance(fallback, MlxAudioTTS)
    assert fallback.model == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    await fallback.warmup()
    for text, lang in (
        ("您好，欢迎致电客服中心，请问有什么可以帮您？", "zh"),
        ("Hello, thank you for calling. How can I help you today?", "en"),
        (MIXED, "zh"),
    ):
        tag = "zh" if lang == "zh" else "en"
        asked = primary.request_body(text, "vivian", tag)
        assert fallback.request_body(text, "vivian", tag) == {**asked, "model": fallback.model}
        timing = SpeechTiming()
        first, total, audio = await timed(fallback, text, "vivian", lang, timing)
        seconds = len(audio) / 2 / WIRE_RATE
        print(
            f"4-bit {lang} {text[:16]!r}: first chunk {first * 1000:.0f} ms, "
            f"RTF {total / seconds:.2f}, {seconds:.2f} s"
        )
        assert (timing.model, timing.fallback) == (fallback.model, True)
        assert first < 0.5
        assert total / seconds < 1.0
        assert 1.0 < seconds < 15.0
    await tts.aclose()


async def test_the_fallback_reads_english_inside_chinese(config: Config, asr: MlxAudioASR) -> None:
    """Why D13 dropped Kokoro (M0 S3: 15.9% errors on mixed text, English words garbled): the
    4-bit fallback's mixed zh/en segment is heard right by the ASR, its English word too."""
    tts = speech(config)
    fallback = tts.fallback
    assert isinstance(fallback, MlxAudioTTS)
    await fallback.warmup()
    await asr.warmup()
    _, _, audio = await timed(fallback, MIXED, "vivian", "zh", SpeechTiming())
    resampler = Resampler(WIRE_RATE, CANONICAL_RATE)
    heard = await asr.transcribe(resampler.process(audio) + resampler.flush(), language="zh")
    await tts.aclose()
    await asr.aclose()
    print(f"spoken {MIXED!r}, heard {heard.text!r}")
    assert "app" in heard.text.lower()
    assert "订单" in heard.text


async def test_sdk_full_chinese_spoken_turn_with_audio(config: Config) -> None:
    """The official SDK, only base_url changed: speech in (the fixture, in real time) →
    transcript → LLM → TTS audio out, within AICC's deadlines."""
    gateway = config.gateway
    client = AsyncOpenAI(base_url=f"http://{gateway.host}:{gateway.port}/v1", api_key="local")
    audio = to_wire(silence(500) + clip(ZH) + silence(2500))
    chunk = 960  # 20 ms at 24 kHz, as AICC sends
    session = {
        "type": "realtime",
        "instructions": "你是客服助手，用一句话简短回答。",
        "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}},
    }
    received: list[tuple[float, RealtimeServerEvent]] = []
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(session=session)  # pyright: ignore[reportArgumentType]

        async def stream_audio() -> None:
            for start in range(0, len(audio), chunk):
                encoded = base64.b64encode(audio[start : start + chunk]).decode()
                await connection.input_audio_buffer.append(audio=encoded)
                await asyncio.sleep(0.02)

        sender = asyncio.create_task(stream_audio())
        async with asyncio.timeout(40):
            async for event in connection:
                received.append((time.monotonic(), event))
                if event.type == "response.done":
                    break
        await sender
    transcript, spoken, status = "", "", ""
    created_at: float | None = None
    deltas: list[float] = []
    for at, event in received:
        if event.type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
        elif event.type == "response.created":
            created_at = at
        elif event.type == "response.output_audio_transcript.delta":
            spoken += event.delta
        elif event.type == "response.output_audio.delta":
            deltas.append(at)
        elif event.type == "response.done":
            status = event.response.status or ""
    assert "恐龙" in transcript
    assert spoken.strip()
    assert status == "completed"
    assert created_at is not None
    assert deltas, "no audio came back"
    print(f"first audio {(deltas[0] - created_at) * 1000:.0f} ms after response.created")
    assert deltas[0] - created_at <= 3.0
    assert max((b - a for a, b in itertools.pairwise(deltas)), default=0.0) < 2.0


LONG_LINE = (  # about 20 s of Qwen3-TTS speech: still being sent (10 s lead) when the caller talks
    "您好，欢迎致电诺瓦网络客户服务中心。为了更好地为您服务，请您仔细听取以下菜单选项。"
    "查询话费和账单请按一，办理宽带业务请按二，咨询套餐变更请按三，故障报修请按四，"
    "人工服务请按零。如果您需要重新收听菜单，请按井号键。感谢您的耐心等待。"
)


async def test_sdk_barge_in_with_the_real_stack(config: Config) -> None:
    """M4 on the real stack, with the official SDK (only base_url changed): the caller (the
    English fixture, streamed in real time like AICC's 20 ms appends) talks over a long verbatim
    line past its echo guard. ``speech_started``, then ``response.done{cancelled,
    turn_detected}`` with only the terminal events between; a truncate to the played time is
    accepted; the caller's turn is transcribed and answered. No ``error`` at all."""
    gateway = config.gateway
    client = AsyncOpenAI(base_url=f"http://{gateway.host}:{gateway.port}/v1", api_key="local")
    chunk = 960  # 20 ms at 24 kHz
    room = bytes(chunk)
    caller = to_wire(clip(EN) + silence(1500))
    session = {
        "type": "realtime",
        "instructions": "You answer the phone for NovaNet. Reply in one short sentence.",
        "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}},
    }
    say = "Say exactly this, word for word, and add nothing else:\n"
    received: list[tuple[float, RealtimeServerEvent]] = []
    talking = asyncio.Event()
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(session=session)  # pyright: ignore[reportArgumentType]

        async def stream_audio() -> None:  # room tone, then the caller, in real time
            spoken = 0
            while True:
                if talking.is_set() and spoken < len(caller):
                    frame, spoken = caller[spoken : spoken + chunk], spoken + chunk
                else:
                    frame = room
                await connection.input_audio_buffer.append(audio=base64.b64encode(frame).decode())
                await asyncio.sleep(0.02)

        sender = asyncio.create_task(stream_audio())
        await connection.response.create(response={"instructions": say + LONG_LINE})
        first_audio: float | None = None
        replies = 0
        try:
            async with asyncio.timeout(60):
                async for event in connection:
                    now = time.monotonic()
                    received.append((now, event))
                    if event.type == "response.output_audio.delta" and first_audio is None:
                        first_audio = now
                        asyncio.get_running_loop().call_later(2.6, talking.set)  # past the guard
                    if (
                        event.type == "input_audio_buffer.speech_started"
                        and first_audio
                        and (
                            "input_audio_buffer.speech_started"
                            not in [earlier.type for _, earlier in received[:-1]]
                        )
                    ):
                        played = int((now - first_audio) * 1000) // 20 * 20
                        line = next(
                            e for _, e in received if e.type == "response.output_item.added"
                        )
                        await connection.conversation.item.truncate(
                            item_id=line.item.id or "", content_index=0, audio_end_ms=played
                        )
                    if event.type == "response.done":
                        replies += 1
                        if replies == 2:
                            break
        finally:
            sender.cancel()
    kinds = [event.type for _, event in received]
    assert "error" not in kinds
    started = kinds.index("input_audio_buffer.speech_started")
    done = kinds.index("response.done")
    assert started < done
    assert set(kinds[started + 1 : done]) <= {
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.done",
    }
    cancelled = received[done][1]
    assert cancelled.type == "response.done"
    assert cancelled.response.status == "cancelled"
    assert first_audio is not None
    onset = first_audio + 2.6 + 0.24  # the caller's speech starts 0.24 s into the fixture
    print(f"barge-in confirmed {(received[started][0] - onset) * 1000:.0f} ms after the onset")
    assert received[started][0] - onset < 2.0
    assert "conversation.item.truncated" in kinds
    transcripts = [
        event.transcript
        for _, event in received
        if event.type == "conversation.item.input_audio_transcription.completed"
    ]
    assert any("dinosaur" in text.lower() for text in transcripts), transcripts
    final = received[-1][1]
    assert final.type == "response.done"
    assert final.response.status == "completed"


@pytest.mark.skipif(
    not os.environ.get("AICC_REPO") or shutil.which("go") is None,
    reason="set AICC_REPO to an AICC checkout (and install Go)",
)
def test_aicc_harness_against_the_real_stack(config: Config) -> None:
    """AICC's own client harness (tests/aicc/probe.py, every ``^TestMLX`` test) against the
    running stack: the handshake and 3 s deadlines, the verbatim lines, and the barge-in
    scenarios (M4) with the real VAD, echo canceller and TTS: the caller talking over a long line
    (``TestMLXBargeIn``: the line cancelled, the truncate accepted, a reply after
    ``speech_stopped``) and keypresses over a line, its unheard item deleted
    (``TestMLXKeypressCancel``)."""
    gateway = config.gateway
    url = f"ws://{gateway.host}:{gateway.port}/v1/realtime"
    probe = REPO / "tests" / "aicc" / "probe.py"
    command = [sys.executable, str(probe), "--aicc-repo", os.environ["AICC_REPO"], "--url", url]
    finished = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=900, check=False
    )
    print(finished.stdout[-6000:])
    assert finished.returncode == 0, finished.stdout[-4000:] + finished.stderr[-2000:]
    for name in ("TestMLXBargeIn", "TestMLXKeypressCancel"):
        assert f"--- PASS: {name} " in finished.stdout, name


def to_g711(pcm16k: bytes, law: g711.Law = "ulaw") -> bytes:
    """Canonical PCM as a telephone line carries it: 8 kHz G.711."""
    resampler = Resampler(CANONICAL_RATE, 8000)
    return g711.encode(resampler.process(pcm16k) + resampler.flush(), law)


async def test_sdk_g711_spoken_turn_with_the_real_stack(config: Config, asr: MlxAudioASR) -> None:
    """M7: the official SDK in G.711 μ-law both ways (``{"type": "audio/pcmu"}``, 160-byte
    frames every 20 ms): speech in → transcript → LLM → TTS → μ-law out, within AICC's
    deadlines, and the answer's audio, decoded, is heard right by the ASR."""
    gateway = config.gateway
    client = AsyncOpenAI(base_url=f"http://{gateway.host}:{gateway.port}/v1", api_key="local")
    audio = to_g711(silence(500) + clip(ZH) + silence(2500))
    session = {
        "type": "realtime",
        "instructions": "你是客服助手，用一句话简短回答。",
        "audio": {
            "input": {
                "format": {"type": "audio/pcmu"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
            },
            "output": {"format": {"type": "audio/pcmu"}},
        },
    }
    received: list[tuple[float, RealtimeServerEvent]] = []
    async with client.realtime.connect(model="gpt-realtime") as connection:
        await connection.session.update(session=session)  # pyright: ignore[reportArgumentType]

        async def stream_audio() -> None:
            for start in range(0, len(audio), 160):
                encoded = base64.b64encode(audio[start : start + 160]).decode()
                await connection.input_audio_buffer.append(audio=encoded)
                await asyncio.sleep(0.02)

        sender = asyncio.create_task(stream_audio())
        async with asyncio.timeout(40):
            async for event in connection:
                received.append((time.monotonic(), event))
                if event.type == "response.done":
                    break
        await sender
    transcript, spoken = "", ""
    created_at: float | None = None
    deltas: list[tuple[float, bytes]] = []
    for at, event in received:
        if event.type == "session.updated":
            assert event.session.model_dump(exclude_none=True)["audio"]["output"]["format"] == {
                "type": "audio/pcmu"
            }
        elif event.type == "conversation.item.input_audio_transcription.completed":
            transcript = event.transcript
        elif event.type == "response.created":
            created_at = at
        elif event.type == "response.output_audio_transcript.delta":
            spoken += event.delta
        elif event.type == "response.output_audio.delta":
            deltas.append((at, base64.b64decode(event.delta)))
        elif event.type == "response.done":
            assert event.response.status == "completed"
    assert "恐龙" in transcript
    assert created_at is not None
    assert deltas, "no audio came back"
    assert all(len(data) <= 800 for _, data in deltas)  # 100 ms of 8 kHz μ-law
    first = deltas[0][0] - created_at
    print(f"first audio {first * 1000:.0f} ms after response.created (G.711)")
    assert first <= 3.0
    times = [at for at, _ in deltas]
    assert max((b - a for a, b in itertools.pairwise(times)), default=0.0) < 2.0
    narrow = g711.decode(b"".join(data for _, data in deltas), "ulaw")
    resampler = Resampler(8000, CANONICAL_RATE)
    heard = await asr.transcribe(resampler.process(narrow) + resampler.flush(), language="zh")
    print(f"spoken {spoken!r}, heard back {heard.text!r}")
    assert overlap(spoken, heard.text) >= 0.6  # of the spoken characters, heard in order


@pytest.mark.skipif(
    not os.environ.get("AICC_REPO") or shutil.which("go") is None,
    reason="set AICC_REPO to an AICC checkout (and install Go)",
)
def test_aicc_harness_with_g711_against_the_real_stack(config: Config) -> None:
    """M7: AICC's harness on the profile's G.711 variant (``AcceptsG711: true``, probe.py
    ``--g711``): μ-law passed through both ways on the real VAD, canceller, ASR, LLM and TTS."""
    gateway = config.gateway
    url = f"ws://{gateway.host}:{gateway.port}/v1/realtime"
    probe = REPO / "tests" / "aicc" / "probe.py"
    command = [
        sys.executable,
        str(probe),
        "--aicc-repo",
        os.environ["AICC_REPO"],
        "--url",
        url,
        "--g711",
    ]
    finished = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=900, check=False
    )
    print(finished.stdout[-6000:])
    assert finished.returncode == 0, finished.stdout[-4000:] + finished.stderr[-2000:]
    for name in ("TestMLXGreeting", "TestMLXBargeIn", "TestMLXKeypressCancel"):
        assert f"--- PASS: {name} " in finished.stdout, name

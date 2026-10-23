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
import pytest
from openai import AsyncOpenAI
from openai.types.realtime import RealtimeServerEvent

from mlx_realtime.audio.pcm import CANONICAL_RATE, WIRE_RATE
from mlx_realtime.audio.resample import Resampler
from mlx_realtime.config import (
    Config,
    MlxAudioAsrConfig,
    MlxAudioSpeechConfig,
    MlxLmChatConfig,
    VoicesConfig,
    load_config,
)
from mlx_realtime.hub import hub_cache
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
from mlx_realtime.server.gateway import pinned_snapshots
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
def llm(config: Config) -> MlxLmChatLLM:
    assert isinstance(config.providers.llm, MlxLmChatConfig), "config.yaml must use mlx_lm_chat"
    return MlxLmChatLLM(config.providers.llm)


@pytest.fixture
def asr(config: Config) -> MlxAudioASR:
    assert isinstance(config.providers.asr, MlxAudioAsrConfig)
    return MlxAudioASR(config.providers.asr)


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


def speech(config: Config, voices: VoicesConfig | None = None) -> FallbackTTS:
    tts = config.providers.tts
    assert isinstance(tts, MlxAudioSpeechConfig), "config.yaml must use mlx_audio_speech"
    snapshots = pinned_snapshots(config.backends, hub_cache(os.environ))
    built = build_speech(tts, VoiceBook(voices or config.voices), snapshots=snapshots)
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


async def test_tts_output_is_heard_right_by_the_asr(config: Config, asr: MlxAudioASR) -> None:
    tts = speech(config)
    _, _, audio = await timed(tts, "您好，欢迎致电客服中心。", "vivian", "zh", SpeechTiming())
    resampler = Resampler(WIRE_RATE, CANONICAL_RATE)
    heard = await asr.transcribe(resampler.process(audio) + resampler.flush(), language="zh")
    await tts.aclose()
    await asr.aclose()
    assert "客服" in heard.text


async def test_the_fallback_speaks_when_the_primary_fails(config: Config) -> None:
    """A speaker Qwen3-TTS does not have fails inside generation (the stream is cut); the
    segment is spoken by Kokoro instead."""
    voices = VoicesConfig.model_validate(
        {
            "default": "broken",
            "map": {
                "broken": {"speaker": "nobody", "fallback": {"zh": "zf_xiaobei", "en": "af_heart"}}
            },
        }
    )
    tts = speech(config, voices)
    timing = SpeechTiming()
    _, _, audio = await timed(tts, "您好，请问有什么可以帮您？", "broken", "zh", timing)
    await tts.aclose()
    assert timing.fallback
    assert len(audio) / 2 / WIRE_RATE > 1.0


async def test_kokoro_speaks_chinese_and_english_from_voice_pack_paths(config: Config) -> None:
    tts = speech(config)
    fallback = tts.fallback
    assert isinstance(fallback, MlxAudioTTS)
    body = fallback.request_body("你好", "vivian", "zh")
    assert str(body["voice"]).endswith("voices/zf_xiaobei.safetensors")
    assert body["lang_code"] == "z"
    for text, lang in (("您好，欢迎致电客服中心。", "zh"), ("Hello, how can I help you?", "en")):
        first, total, audio = await timed(fallback, text, "vivian", lang, SpeechTiming())
        seconds = len(audio) / 2 / WIRE_RATE
        print(f"kokoro {lang}: first chunk {first * 1000:.0f} ms, total {total * 1000:.0f} ms")
        assert seconds > 0.8
    await tts.aclose()


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

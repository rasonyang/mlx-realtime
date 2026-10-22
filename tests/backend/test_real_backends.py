"""The real model backends (marker ``backend``; skipped unless selected).

Start the stack first, then run these against it:

    make dev                                  # backends and gateway, from config.yaml
    uv run --locked pytest -m backend         # in another terminal

The addresses come from ``config.yaml`` (and ``MLXRT__`` overrides), as the gateway's own do.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path

import httpx2
import pytest
from openai import AsyncOpenAI

from mlx_realtime.config import Config, MlxAudioAsrConfig, MlxLmChatConfig, load_config
from mlx_realtime.providers.base import Delta, LLMDone, Message, TextDelta, ToolCall, ToolDef
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
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

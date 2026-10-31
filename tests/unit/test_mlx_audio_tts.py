"""``MlxAudioTTS`` against a fake mlx_audio.server: the request (the voice's speaker and the
segment's lang code, the same on the primary and the fallback), the streamed PCM, deadlines,
errors, cancellation at each await point, the in-flight limit and its queue wait, readiness and
warmup."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import cast

import httpx2
import pytest

from mlx_realtime.config import MlxAudioSpeechConfig, TokenCapConfig, VoicesConfig
from mlx_realtime.providers.base import (
    Lang,
    PCMChunk,
    ProviderError,
    ProviderTimeout,
    SpeechTiming,
)
from mlx_realtime.providers.fallback import FallbackTTS
from mlx_realtime.providers.mlx_audio_tts import MlxAudioTTS, Role, SpeechServer, token_cap
from mlx_realtime.providers.mock import EchoLLM
from mlx_realtime.providers.registry import build_speech
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.session.pipeline import run_pipeline
from mlx_realtime.session.signals import PipelineMark, PipelineSignal, SegmentAudio
from tests.fakes.mlx_audio import (
    Chunks,
    Cut,
    Fail,
    FakeMlxAudio,
    Gate,
    Hold,
    Pause,
    Raw,
    Rest,
    Start,
    serve_asgi,
)
from tests.unit.test_pipeline import StepClock, context, request

QWEN = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
QWEN_4BIT = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
TEXT = "您好，欢迎致电客服中心，请问有什么可以帮您？"
AUDIO = 22 * 2880  # 22 characters at the fake's 60 ms each, 24 kHz s16le
CHUNK = 15360  # 0.32 s at 24 kHz: four 80 ms codec frames
FAST = {"connect_ms": 500, "first_item_ms": 2000, "idle_ms": 1000, "total_ms": 5000}


@pytest.fixture
async def fake() -> AsyncIterator[tuple[FakeMlxAudio, str]]:
    server = FakeMlxAudio()
    async with serve_asgi(server) as url:
        yield server, url


def speech_config(base_url: str, **overrides: object) -> MlxAudioSpeechConfig:
    raw: dict[str, object] = {"type": "mlx_audio_speech", "base_url": base_url}
    # 0.32 s chunks (the stream tests count CHUNK); the defaults are tested on their own
    raw["primary"] = {"model": QWEN, "timeouts": FAST, "streaming_interval": 0.32}
    raw["fallback"] = {"model": QWEN_4BIT, "timeouts": FAST}
    raw.update(overrides)
    return MlxAudioSpeechConfig.model_validate(raw)


def primary(
    base_url: str,
    *,
    voices: VoicesConfig | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    role: Role = "primary",
    **overrides: object,
) -> MlxAudioTTS:
    """The primary model's provider (or, with ``role``, the fallback's)."""
    config = speech_config(base_url, **overrides)
    server = SpeechServer(config, transport=transport)
    return MlxAudioTTS(
        server,
        config.primary if role == "primary" else config.fallback,
        VoiceBook(voices or VoicesConfig()),
        role=role,
        warmup_texts=config.warmup_texts,
        line_cache=config.line_cache,
    )


async def collect(stream: AsyncGenerator[PCMChunk]) -> list[PCMChunk]:
    async with aclosing(stream):
        return [chunk async for chunk in stream]


async def test_qwen3_requests_name_the_speaker_and_the_segment_language(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url, primary={"model": QWEN, "timeouts": FAST})  # the defaults
    await collect(tts.stream(TEXT, "serena", "zh"))
    await collect(tts.stream("Hello there.", "aiden", "en"))
    await collect(tts.stream("Hello there.", "flow-voice", "en"))  # not configured: the default
    await collect(tts.stream(TEXT, "flow-voice", "zh"))
    await tts.aclose()
    chinese, english, unknown_en, unknown_zh = (speech.body for speech in server.speeches)
    assert chinese == {
        "model": QWEN,
        "input": TEXT,
        "voice": "serena",
        "lang_code": "chinese",
        "response_format": "pcm",
        "stream": True,
        "streaming_interval": 0.16,  # S3
        "max_tokens": token_cap(TEXT, TokenCapConfig(), None),  # S3: in proportion to the text
    }  # no sampling fields: the model's generation_config.json decides
    assert (english["voice"], english["lang_code"]) == ("aiden", "english")
    # the default voice, vivian, reads English herself (one voice per session)
    assert (unknown_en["voice"], unknown_en["lang_code"]) == ("vivian", "english")
    assert (unknown_zh["voice"], unknown_zh["lang_code"]) == ("vivian", "chinese")


def test_the_token_cap_follows_the_text() -> None:
    cap = TokenCapConfig(base=25, per_cjk=6, per_other=2)
    assert token_cap("你好。", cap, None) == 25 + 12
    assert token_cap("Hi there!", cap, None) == 25 + 14
    assert token_cap("订单 A12", cap, None) == 25 + 12 + 6
    assert token_cap("你" * 500, cap, None) == 1200  # never past the server's own limit
    assert token_cap("你" * 500, cap, 900) == 900  # nor the configured max_tokens


async def test_sampling_fields_and_a_fixed_lang_code_when_configured(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    sampling = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1}
    voices = VoicesConfig.model_validate(
        {"default": "auto", "map": {"auto": {"speaker": "aiden", "lang": "auto"}}}
    )
    tts = primary(
        url,
        voices=voices,
        primary={
            "model": QWEN,
            "max_tokens": 600,
            "token_cap": None,
            "streaming_interval": 0.16,
            **sampling,
        },
    )
    await collect(tts.stream(TEXT, "auto", "zh"))
    await tts.aclose()
    (body,) = (speech.body for speech in server.speeches)
    assert (body["voice"], body["lang_code"], body["max_tokens"]) == ("aiden", "auto", 600)
    assert body["streaming_interval"] == 0.16
    assert {key: body[key] for key in sampling} == sampling


async def test_the_fallback_is_asked_exactly_like_the_primary(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    """D13 final: the 4-bit fallback has the primary's speakers, so each segment, mixed zh/en
    included, goes to it as it goes to the primary (speaker, lang_code, chunk size, token cap);
    only the model differs."""
    server, url = fake
    voices = VoicesConfig.model_validate(
        {
            "default": "vivian",
            "map": {
                "vivian": {"speaker": "vivian", "speakers": {"en": "aiden"}},
                "fixed": {"speaker": "serena", "lang": "auto"},
            },
        }
    )
    segments: list[tuple[str, str, Lang]] = [
        (TEXT, "vivian", "zh"),
        ("Hello there.", "vivian", "en"),
        ("请登录 App 查看订单。", "vivian", "zh"),
        ("Hello there.", "fixed", "en"),
        (TEXT, "flow-voice", "zh"),  # not configured: the default
    ]
    defaults = {"model": QWEN, "timeouts": FAST}  # 0.16 s chunks on both models
    roles: tuple[Role, ...] = ("primary", "fallback")
    timings: list[SpeechTiming] = []
    for role in roles:
        tts = primary(url, voices=voices, role=role, primary=defaults)
        for text, voice, lang in segments:
            timings.append(SpeechTiming())
            chunks = await collect(tts.stream(text, voice, lang, timing=timings[-1]))
            assert len(chunks) > 1  # streamed, both models alike
        await tts.aclose()
    asked = [speech.body for speech in server.speeches]
    first, second = asked[: len(segments)], asked[len(segments) :]
    assert [body["model"] for body in first] == [QWEN] * len(segments)
    assert [{**body, "model": QWEN} for body in second] == first
    assert [(body["voice"], body["lang_code"]) for body in second] == [
        ("vivian", "chinese"),
        ("aiden", "english"),  # vivian's English is aiden's (S3), on the fallback too
        ("vivian", "chinese"),
        ("serena", "auto"),  # a fixed lang_code is the fallback's too
        ("vivian", "chinese"),
    ]
    assert second[2]["input"] == "请登录 App 查看订单。"  # Latin letters are no reason to fail
    assert second[2]["max_tokens"] == token_cap("请登录 App 查看订单。", TokenCapConfig(), None)
    assert {body["streaming_interval"] for body in second} == {0.16}
    assert [(timing.model, timing.fallback) for timing in timings] == [(QWEN, False)] * len(
        segments
    ) + [(QWEN_4BIT, True)] * len(segments)


async def test_the_stream_is_the_servers_pcm_in_whole_samples(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url)
    timing = SpeechTiming()
    chunks = await collect(tts.stream(TEXT, "vivian", "zh", timing=timing))
    server.script_speech(Start(), Raw(b"\x01\x02\x03"), Pause(0.05), Raw(b"\x04\x05\x06\x07\x08"))
    odd = await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert sum(len(chunk.pcm) for chunk in chunks) == server.speeches[0].audio_bytes == AUDIO
    assert [size for _, size in server.speeches[0].chunks] == [CHUNK] * 4 + [AUDIO - 4 * CHUNK]
    assert len(chunks) > 1  # passed on as they arrive, not collected
    assert {chunk.sample_rate for chunk in chunks} == {24000}
    assert timing.model == QWEN
    assert not timing.fallback
    assert timing.queued_s >= 0
    assert b"".join(chunk.pcm for chunk in odd) == bytes(range(1, 9))
    assert all(len(chunk.pcm) % 2 == 0 for chunk in odd)


@pytest.mark.parametrize(
    ("script", "kind", "detail"),
    [
        pytest.param([Fail(500, "Failed to load model")], "internal", "500", id="load-failed"),
        pytest.param([Fail(503, "busy")], "unavailable", "503", id="unavailable"),
        pytest.param([Fail(422, "bad field")], "bad_response", "422", id="rejected"),
        pytest.param([Cut()], "bad_response", "cut", id="cut-before-audio"),
        pytest.param([Start()], "bad_response", "no audio", id="no-audio"),
        pytest.param([Start(), Raw(b"\x00")], "bad_response", "inside a sample", id="half-sample"),
    ],
)
async def test_errors(
    fake: tuple[FakeMlxAudio, str], script: list[object], kind: str, detail: str
) -> None:
    server, url = fake
    server.script_speech(*script)  # pyright: ignore[reportArgumentType]
    tts = primary(url)
    with pytest.raises(ProviderError) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert failed.value.kind == kind
    assert detail in str(failed.value)


async def test_an_unknown_model_is_a_bad_response(fake: tuple[FakeMlxAudio, str]) -> None:
    _, url = fake
    tts = primary(url, primary={"model": "nobody/tts"})
    with pytest.raises(ProviderError) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert (failed.value.kind, "404" in str(failed.value)) == ("bad_response", True)


async def test_a_stream_cut_after_audio_reports_the_audio_first(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    server.script_speech(Chunks(1), Cut())
    tts = primary(url)
    received: list[PCMChunk] = []

    async def consume() -> None:
        async with aclosing(tts.stream(TEXT, "vivian", "zh")) as stream:
            async for chunk in stream:
                received.append(chunk)  # noqa: PERF401 (the chunks before the failure count)

    with pytest.raises(ProviderError) as failed:
        await consume()
    await tts.aclose()
    assert [len(chunk.pcm) for chunk in received] == [CHUNK]
    assert failed.value.kind == "bad_response"


async def test_a_backend_that_is_down_is_unavailable() -> None:
    tts = primary(f"http://127.0.0.1:{free_port()}")
    with pytest.raises(ProviderError) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert (failed.value.kind, failed.value.retryable) == ("unavailable", True)


async def test_the_connect_deadline_and_a_wrong_content_type() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("timed out", request=request)

    def json_answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"oops": True})

    tts = primary("http://tts.invalid", transport=httpx2.MockTransport(slow))
    with pytest.raises(ProviderTimeout) as timed_out:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert timed_out.value.stage == "connect"
    tts = primary("http://tts.invalid", transport=httpx2.MockTransport(json_answer))
    with pytest.raises(ProviderError) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    await tts.aclose()
    assert failed.value.kind == "bad_response"
    assert "content type" in str(failed.value)


@pytest.mark.parametrize(
    ("script", "timeouts", "stage"),
    [
        pytest.param([Hold(Gate())], {"first_item_ms": 150}, "first_item", id="before-headers"),
        pytest.param(
            [Start(), Hold(Gate())], {"first_item_ms": 150}, "first_item", id="first-chunk"
        ),
        pytest.param([Chunks(1), Hold(Gate())], {"idle_ms": 150}, "idle", id="between-chunks"),
        pytest.param(
            [Chunks(1), Pause(0.1), Chunks(1), Pause(0.1), Chunks(1), Pause(0.1), Rest()],
            {"idle_ms": 1000, "total_ms": 220},
            "total",
            id="total",
        ),
    ],
)
async def test_deadlines(
    fake: tuple[FakeMlxAudio, str],
    script: list[object],
    timeouts: dict[str, int],
    stage: str,
) -> None:
    server, url = fake
    server.script_speech(*script)  # pyright: ignore[reportArgumentType]
    tts = primary(url, primary={"model": QWEN, "timeouts": FAST | timeouts})
    with pytest.raises(ProviderTimeout) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    assert (failed.value.stage, failed.value.kind) == (stage, "timeout")
    await server.disconnected()  # the request was abandoned, not left running
    await tts.aclose()


@pytest.mark.parametrize("point", ["awaiting-headers", "before-first-chunk", "between-chunks"])
async def test_cancellation_while_awaiting_the_backend(
    fake: tuple[FakeMlxAudio, str], point: str
) -> None:
    server, url = fake
    gate = Gate()
    match point:
        case "awaiting-headers":
            server.script_speech(Hold(gate), Rest())
        case "before-first-chunk":
            server.script_speech(Start(), Hold(gate), Rest())
        case _:
            server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url)
    received: list[PCMChunk] = []

    async def consume() -> None:
        async with aclosing(tts.stream(TEXT, "vivian", "zh")) as stream:
            async for chunk in stream:
                received.append(chunk)  # noqa: PERF401 (the chunks before the cancel count)

    task = asyncio.create_task(consume())
    await gate.reached()
    if point == "between-chunks":
        async with asyncio.timeout(2):
            while not received:  # noqa: ASYNC110 (the consumer appends; no event to wait on)
                await asyncio.sleep(0.002)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await server.disconnected()
    gate.release()
    assert sum(len(chunk.pcm) for chunk in received) == (CHUNK if point == "between-chunks" else 0)
    await tts.aclose()


async def test_cancellation_while_the_consumer_holds_a_chunk(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url)
    holding = asyncio.Event()

    async def consume() -> None:
        async with aclosing(tts.stream(TEXT, "vivian", "zh")) as stream:
            await anext(stream)
            holding.set()
            await asyncio.Event().wait()  # busy elsewhere, the stream suspended at its yield

    task = asyncio.create_task(consume())
    await holding.wait()
    await gate.reached()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await server.disconnected()
    gate.release()
    await tts.aclose()


async def test_requests_queue_for_the_backend_and_the_wait_is_reported(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url, max_in_flight=1, queue_timeout_ms=5000)
    first = tts.stream(TEXT, "vivian", "zh")
    await anext(first)
    timing = SpeechTiming()
    second = asyncio.create_task(collect(tts.stream(TEXT, "vivian", "zh", timing=timing)))
    await asyncio.sleep(0.2)
    assert len(server.speeches) == 1  # the second request waits in the gateway
    gate.release()
    await collect(first)
    assert sum(len(chunk.pcm) for chunk in await second) == AUDIO
    assert timing.queued_s >= 0.2
    await tts.aclose()


async def test_a_full_queue_is_overloaded(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url, max_in_flight=1, queue_timeout_ms=50)
    first = tts.stream(TEXT, "vivian", "zh")
    await anext(first)
    with pytest.raises(ProviderError) as failed:
        await collect(tts.stream(TEXT, "vivian", "zh"))
    assert (failed.value.kind, failed.value.retryable) == ("overloaded", True)
    gate.release()
    await collect(first)
    await tts.aclose()


async def test_cancellation_while_waiting_for_a_slot(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url, max_in_flight=1, queue_timeout_ms=5000)
    first = tts.stream(TEXT, "vivian", "zh")
    await anext(first)
    waiting = asyncio.create_task(collect(tts.stream(TEXT, "vivian", "zh")))
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert len(server.speeches) == 1  # the queued request never reached the backend
    gate.release()
    await collect(first)
    await collect(tts.stream(TEXT, "vivian", "zh"))  # the slot was given back
    assert len(server.speeches) == 2
    await tts.aclose()


async def test_warmup_loads_both_models_and_tries_every_voice_on_the_primary(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    """The primary speaks every configured voice in both languages, so a speaker it lacks fails
    the warmup. The fallback has the same speakers (D13), so the default voice in both languages
    is enough to load it. Readiness needs both models loaded."""
    server, url = fake
    tts = build_speech(speech_config(url), VoiceBook(VoicesConfig()))
    assert isinstance(tts, FallbackTTS)
    assert not (await tts.ready()).ready  # nothing loaded yet
    await tts.primary.warmup()
    state = await tts.ready()
    assert not state.ready
    assert state.detail == "primary: ok; fallback: fallback model not loaded"
    await tts.fallback.warmup()
    requests = [
        (speech.body["model"], speech.body["voice"], speech.body["input"])
        for speech in server.speeches
    ]
    zh, en = "您好，请问有什么可以帮您？", "Hello, how can I help you?"
    # each speaker and language is warmed once
    assert {(voice, text) for model, voice, text in requests if model == QWEN} == {
        (speaker, text)
        for speaker in ("vivian", "serena", "uncle_fu", "aiden")
        for text in (zh, en)
    }
    # the default voice, vivian, in both languages: 2 requests, not another 7
    assert [(voice, text) for model, voice, text in requests if model == QWEN_4BIT] == [
        ("vivian", zh),
        ("vivian", en),
    ]
    assert len(requests) == len(set(requests)) == 8 + 2
    state = await tts.ready()
    assert state.ready, state.detail
    await tts.aclose()


async def test_a_failed_warmup_request_fails_the_warmup(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    voices = VoicesConfig.model_validate({"default": "bad", "map": {"bad": {"speaker": "nobody"}}})
    tts = primary(url, voices=voices)
    with pytest.raises(ProviderError):
        await tts.warmup()
    assert server.speeches[0].ended == "cut"  # Qwen3-TTS has no such speaker
    await tts.aclose()


async def test_readiness_of_a_backend_that_is_down() -> None:
    tts = primary(f"http://127.0.0.1:{free_port()}")
    state = await tts.ready()
    assert not state.ready
    await tts.aclose()


def test_the_fallback_can_be_switched_off() -> None:
    config = speech_config("http://tts.invalid", fallback={"model": QWEN_4BIT, "enabled": False})
    tts = build_speech(config, VoiceBook(VoicesConfig()))
    assert isinstance(tts, MlxAudioTTS)
    assert tts.model == QWEN
    body = json.dumps(tts.request_body("你好", "vivian", "zh"), ensure_ascii=False)
    assert '"lang_code": "chinese"' in body


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def say(
    tts: MlxAudioTTS, text: str = TEXT, voice: str = "vivian", lang: Lang = "zh"
) -> tuple[list[PCMChunk], SpeechTiming]:
    """One verbatim line, as the pipeline asks for it (``cacheable``)."""
    timing = SpeechTiming(cacheable=True)
    return await collect(tts.stream(text, voice, lang, timing=timing)), timing


async def test_a_repeated_line_is_replayed_and_skips_the_backend_slot(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url, max_in_flight=1, queue_timeout_ms=5000)
    live, first = await say(tts)
    assert not first.cached
    assert len(server.speeches) == 1
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    busy = tts.stream("请稍等，我查一下。", "vivian", "zh")
    await anext(busy)  # another take holds the only slot
    replay, second = await say(tts)
    assert replay == live  # the raw chunks, in order
    assert second.cached
    assert (second.queued_s, second.model, second.fallback) == (0.0, QWEN, False)
    assert len(server.speeches) == 2  # the hit did not reach the backend
    gate.release()
    await collect(busy)
    await tts.aclose()


async def test_a_take_that_is_not_a_verbatim_line_is_neither_stored_nor_replayed(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url)
    await collect(tts.stream(TEXT, "vivian", "zh"))
    await collect(tts.stream(TEXT, "vivian", "zh"))
    await say(tts)  # the takes above stored nothing
    assert len(server.speeches) == 3
    await tts.aclose()


async def test_a_replay_yields_to_the_loop_between_chunks(fake: tuple[FakeMlxAudio, str]) -> None:
    _, url = fake
    tts = primary(url)
    live, _ = await say(tts)
    assert len(live) > 1
    ticks = 0

    async def spin() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    spinner = asyncio.create_task(spin())
    await say(tts)
    spinner.cancel()
    assert ticks >= len(live) - 1
    await tts.aclose()


async def test_a_change_of_voice_or_language_is_a_different_line(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url)
    await say(tts)
    await say(tts, voice="serena")
    await say(tts, lang="en")
    assert len(server.speeches) == 3
    for voice, lang in (("vivian", "zh"), ("serena", "zh"), ("vivian", "en")):
        _, timing = await say(tts, voice=voice, lang=cast(Lang, lang))
        assert timing.cached
    assert len(server.speeches) == 3
    await tts.aclose()


async def test_the_cache_keeps_the_most_recently_used_lines(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    tts = primary(url, line_cache=2)
    for text in ("第一句话。", "第二句话。", "第一句话。", "第三句话。"):
        await say(tts, text)  # the third evicts the second, not the first
    assert len(server.speeches) == 3
    assert (await say(tts, "第一句话。"))[1].cached
    assert not (await say(tts, "第二句话。"))[1].cached
    await tts.aclose()


async def test_a_cancelled_take_is_not_stored(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script_speech(Chunks(1), Hold(gate), Rest())
    tts = primary(url)

    async def consume() -> None:
        stream = tts.stream(TEXT, "vivian", "zh", timing=SpeechTiming(cacheable=True))
        async with aclosing(stream):
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    await gate.reached()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await server.disconnected()
    gate.release()
    assert not (await say(tts))[1].cached
    assert len(server.speeches) == 2
    await tts.aclose()


async def test_a_take_closed_early_is_not_stored(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    tts = primary(url)
    stream = tts.stream(TEXT, "vivian", "zh", timing=SpeechTiming(cacheable=True))
    async with aclosing(stream):
        async for _ in stream:
            break  # the pipeline closes a response at max_audio_ms like this
    assert not (await say(tts))[1].cached
    assert len(server.speeches) == 2
    await tts.aclose()


@pytest.mark.parametrize(("extra", "kept"), [(0, False), (5, True)], ids=["capped", "under-cap"])
async def test_a_take_that_reached_max_tokens_is_not_stored(
    fake: tuple[FakeMlxAudio, str], extra: int, kept: bool
) -> None:
    server, url = fake
    frames = int(AUDIO / (24_000 * 2 * 0.08))  # the codec frames of the fake's answer to TEXT
    model = {"model": QWEN, "timeouts": FAST, "streaming_interval": 0.32}
    tts = primary(url, primary={**model, "max_tokens": frames + extra})
    await say(tts)
    assert (await say(tts))[1].cached is kept
    assert len(server.speeches) == (1 if kept else 2)
    await tts.aclose()


async def test_the_fallback_never_stores_or_reads(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    tts = primary(url, role="fallback")
    await say(tts)
    assert not (await say(tts))[1].cached
    assert len(server.speeches) == 2
    await tts.aclose()


async def test_a_line_cache_of_zero_is_off(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    tts = primary(url, line_cache=0)
    await say(tts)
    assert not (await say(tts))[1].cached
    assert len(server.speeches) == 2
    await tts.aclose()


async def test_a_replayed_segment_goes_through_the_pipeline_like_a_live_one(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    """Trim, pacing, the audio limit and the segment map count the replay as a live take."""
    server, url = fake
    tts = primary(url)
    runs: list[list[PipelineSignal]] = []
    # cut at the limit and live (a take closed there is not stored), a whole one that fills the
    # cache, then the same cut again, now replayed
    for max_audio_ms in (500, 120_000, 500):
        posted: list[PipelineSignal] = []
        pipeline = context(EchoLLM(), tts, posted, StepClock(), max_audio_ms=max_audio_ms)
        await run_pipeline(request(verbatim=TEXT), pipeline)
        runs.append(posted)
    assert len(server.speeches) == 3  # the segments of runs 1 and 2; the replay asked for none

    def audio(posted: list[PipelineSignal]) -> list[tuple[int, bytes]]:
        return [(s.index, s.pcm) for s in posted if isinstance(s, SegmentAudio)]

    marks = [
        dict(s.values)
        for posted in runs
        for s in posted
        if isinstance(s, PipelineMark) and s.name == "tts_first_chunk"
    ]
    assert "tts_cache" not in marks[0]
    assert marks[2] == {"tts_model": QWEN, "tts_cache": True}  # no tts_queue_ms
    assert audio(runs[0]) == audio(runs[2])
    assert 500 * 48 < sum(len(pcm) for _, pcm in audio(runs[2])) < AUDIO  # cut at the limit
    await tts.aclose()

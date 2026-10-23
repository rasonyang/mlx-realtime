"""``MlxAudioTTS`` against a fake mlx_audio.server: the request per model (Qwen3-TTS speakers,
Kokoro voice packs, lang codes), the streamed PCM, deadlines, errors, cancellation at each await
point, the in-flight limit and its queue wait, readiness and warmup."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from pathlib import Path

import httpx2
import pytest

from mlx_realtime.config import MlxAudioSpeechConfig, TokenCapConfig, VoicesConfig
from mlx_realtime.providers.base import PCMChunk, ProviderError, ProviderTimeout, SpeechTiming
from mlx_realtime.providers.fallback import FallbackTTS
from mlx_realtime.providers.mlx_audio_tts import MlxAudioTTS, SpeechServer, token_cap
from mlx_realtime.providers.registry import build_speech
from mlx_realtime.providers.voices import VoiceBook
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

QWEN = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
KOKORO = "mlx-community/Kokoro-82M-bf16"
TEXT = "您好，欢迎致电客服中心，请问有什么可以帮您？"
AUDIO = 22 * 2880  # 22 characters at the fake's 60 ms each, 24 kHz s16le
CHUNK = 15360  # 0.32 s at 24 kHz: four 80 ms codec frames
FAST = {"connect_ms": 500, "first_item_ms": 2000, "idle_ms": 1000, "total_ms": 5000}


@pytest.fixture
async def fake() -> AsyncIterator[tuple[FakeMlxAudio, str]]:
    server = FakeMlxAudio()
    async with serve_asgi(server) as url:
        yield server, url


@pytest.fixture
def packs(tmp_path: Path) -> Path:
    """A Kokoro snapshot with the voice packs of the built-in voice map."""
    voices = tmp_path / "voices"
    voices.mkdir()
    for voice in VoicesConfig().map.values():
        for name in voice.fallback.values():
            (voices / f"{name}.safetensors").write_bytes(b"pack")
    return tmp_path


def speech_config(base_url: str, **overrides: object) -> MlxAudioSpeechConfig:
    raw: dict[str, object] = {"type": "mlx_audio_speech", "base_url": base_url}
    # 0.32 s chunks (the stream tests count CHUNK); the defaults are tested on their own
    raw["primary"] = {"model": QWEN, "timeouts": FAST, "streaming_interval": 0.32}
    raw["fallback"] = {"model": KOKORO, "family": "kokoro", "timeouts": FAST}
    raw.update(overrides)
    return MlxAudioSpeechConfig.model_validate(raw)


def primary(
    base_url: str,
    *,
    voices: VoicesConfig | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    **overrides: object,
) -> MlxAudioTTS:
    config = speech_config(base_url, **overrides)
    server = SpeechServer(config, transport=transport)
    return MlxAudioTTS(
        server,
        config.primary,
        VoiceBook(voices or VoicesConfig()),
        role="primary",
        warmup_texts=config.warmup_texts,
    )


def kokoro(base_url: str, packs: Path | None) -> MlxAudioTTS:
    config = speech_config(base_url)
    return MlxAudioTTS(
        SpeechServer(config),
        config.fallback,
        VoiceBook(VoicesConfig()),
        role="fallback",
        warmup_texts=config.warmup_texts,
        voices_dir=None if packs is None else packs / "voices",
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
    # the default voice, vivian, hands English segments to aiden (S3)
    assert (unknown_en["voice"], unknown_en["lang_code"]) == ("aiden", "english")
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


async def test_kokoro_requests_send_the_voice_pack_path_and_its_lang_code(
    fake: tuple[FakeMlxAudio, str], packs: Path
) -> None:
    server, url = fake
    tts = kokoro(url, packs)
    timing = SpeechTiming()
    chunks = await collect(tts.stream(TEXT, "vivian", "zh", timing=timing))
    await collect(tts.stream("Hello.", "aiden", "en"))
    await tts.aclose()
    chinese, english = (speech.body for speech in server.speeches)
    assert chinese["voice"] == str(packs / "voices" / "zf_xiaobei.safetensors")
    assert chinese["lang_code"] == "z"
    assert "max_tokens" not in chinese  # the token cap is Qwen3-TTS's
    assert english["voice"] == str(packs / "voices" / "am_puck.safetensors")
    assert english["lang_code"] == "a"
    assert sum(len(chunk.pcm) for chunk in chunks) == AUDIO  # Kokoro: one chunk
    assert (timing.model, timing.fallback) == (KOKORO, True)


async def test_a_missing_voice_pack_fails_before_any_request(
    fake: tuple[FakeMlxAudio, str], packs: Path
) -> None:
    server, url = fake
    (packs / "voices" / "zf_xiaobei.safetensors").unlink()
    for tts in (kokoro(url, packs), kokoro(url, None)):
        with pytest.raises(ProviderError) as failed:
            await collect(tts.stream(TEXT, "vivian", "zh"))
        assert failed.value.kind == "unavailable"
        await tts.aclose()
    assert server.speeches == []


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


async def test_warmup_speaks_every_voice_in_both_languages_on_every_model(
    fake: tuple[FakeMlxAudio, str], packs: Path
) -> None:
    server, url = fake
    tts = build_speech(speech_config(url), VoiceBook(VoicesConfig()), snapshots=lambda repo: packs)
    assert isinstance(tts, FallbackTTS)
    assert not (await tts.ready()).ready  # nothing loaded yet
    await tts.warmup()
    requests = [
        (speech.body["model"], speech.body["voice"], speech.body["input"])
        for speech in server.speeches
    ]
    qwen = {(voice, text) for model, voice, text in requests if model == QWEN}
    zh, en = "您好，请问有什么可以帮您？", "Hello, how can I help you?"
    # vivian's English is aiden's (S3), so each speaker and language is warmed once
    assert qwen == {
        ("vivian", zh),
        *((speaker, text) for speaker in ("serena", "uncle_fu", "aiden") for text in (zh, en)),
    }
    kokoro_voices = {Path(voice).stem for model, voice, _ in requests if model == KOKORO}
    assert kokoro_voices == {
        name for voice in VoicesConfig().map.values() for name in voice.fallback.values()
    }
    assert len(requests) == len(set(requests)) == 7 + 8
    state = await tts.ready()
    assert state.ready, state.detail
    await tts.aclose()


async def test_a_failed_warmup_request_fails_the_warmup(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    voices = VoicesConfig.model_validate(
        {"default": "bad", "map": {"bad": {"speaker": "nobody", "fallback": {}}}}
    )
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


def test_the_fallback_can_be_switched_off(packs: Path) -> None:
    config = speech_config("http://tts.invalid", fallback={"model": KOKORO, "enabled": False})
    tts = build_speech(config, VoiceBook(VoicesConfig()))
    assert isinstance(tts, MlxAudioTTS)
    assert tts.model == QWEN
    body = json.dumps(tts.request_body("你好", "vivian", "zh"), ensure_ascii=False)
    assert '"lang_code": "chinese"' in body


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]

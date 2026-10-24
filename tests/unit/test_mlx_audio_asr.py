"""``MlxAudioASR`` against a fake mlx_audio.server: the multipart request, the verbose_json
answer, deadlines, errors, cancellation, readiness and warmup."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Callable

import httpx2
import numpy as np
import pytest

from mlx_realtime.config import MlxAudioAsrConfig
from mlx_realtime.providers.base import ProviderError, ProviderTimeout
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR, parse_transcript, wav_file
from tests.fakes.mlx_audio import Fail, FakeMlxAudio, Garbage, Gate, Hold, Transcribe, serve_asgi

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
SPEECH = (np.sin(np.arange(16000) / 7) * 3000).astype("<i2").tobytes()  # 1 s at 16 kHz


@pytest.fixture
async def fake() -> AsyncIterator[tuple[FakeMlxAudio, str]]:
    server = FakeMlxAudio()
    async with serve_asgi(server) as url:
        yield server, url


def provider(base_url: str, *, context: str | None = None, **timeouts: int) -> MlxAudioASR:
    values = {"connect_ms": 500, "first_item_ms": 2000, "idle_ms": 1000, "total_ms": 5000}
    config = MlxAudioAsrConfig.model_validate(
        {
            "type": "mlx_audio_transcription",
            "base_url": base_url,
            "timeouts": values | timeouts,
            "context": context,
        }
    )
    return MlxAudioASR(config)


async def test_the_request_is_a_16k_mono_wav_with_the_form_fields(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    asr = provider(url)
    transcript = await asr.transcribe(SPEECH, language="zh")
    await asr.transcribe(SPEECH)
    await asr.aclose()
    with_language, auto = server.uploads
    assert with_language.fields == {
        "model": MODEL,
        "response_format": "verbose_json",
        "language": "Chinese",  # Qwen3-ASR names its languages
    }
    assert "language" not in auto.fields  # omitted: the model detects the language
    rate, channels, width, frames = with_language.audio()
    assert (rate, channels, width, frames) == (16000, 1, 2, SPEECH)
    assert (transcript.text, transcript.language, transcript.audio_ms) == ("你好。", "zh", 1000)


async def test_a_context_is_sent_as_the_context_field_only_when_there_is_one(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    asr = provider(url)
    await asr.transcribe(SPEECH, context="NovaNet, eSIM")
    await asr.transcribe(SPEECH, context="")
    await asr.transcribe(SPEECH, context=None)
    await asr.aclose()
    with_context, empty, none = server.uploads
    assert with_context.fields["context"] == "NovaNet, eSIM"
    # No context: the form is exactly what it was before the field existed.
    assert empty.fields == none.fields == {"model": MODEL, "response_format": "verbose_json"}


async def test_the_configured_context_is_the_default_and_a_turns_context_replaces_it(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    asr = provider(url, context="Wi-Fi 7")
    await asr.transcribe(SPEECH)
    await asr.transcribe(SPEECH, context="")  # empty is no context: the default applies
    await asr.transcribe(SPEECH, context="NovaNet")  # replaces it, no merge
    await asr.aclose()
    assert [upload.fields.get("context") for upload in server.uploads] == [
        "Wi-Fi 7",
        "Wi-Fi 7",
        "NovaNet",
    ]


async def test_the_warmup_sends_no_context(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    asr = provider(url, context="Wi-Fi 7")
    await asr.warmup()
    await asr.aclose()
    assert "context" not in server.uploads[0].fields


@pytest.mark.parametrize(
    ("languages", "expected"),
    [(("English",), "en"), (("Cantonese",), "yue"), (("Klingon",), "Klingon"), ((), None)],
)
async def test_the_reported_language_becomes_a_code_when_known(
    fake: tuple[FakeMlxAudio, str], languages: tuple[str, ...], expected: str | None
) -> None:
    server, url = fake
    server.script(Transcribe("  Hello there.  ", languages))
    asr = provider(url)
    transcript = await asr.transcribe(SPEECH)
    await asr.aclose()
    assert (transcript.text, transcript.language) == ("Hello there.", expected)


async def test_empty_audio_is_not_sent(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    asr = provider(url)
    transcript = await asr.transcribe(b"")
    await asr.aclose()
    assert (transcript.text, transcript.audio_ms, server.uploads) == ("", 0, [])


@pytest.mark.parametrize(
    ("step", "kind"),
    [
        (Fail(404, "Model not found"), "bad_response"),
        (Fail(500, "Failed to load model"), "internal"),
        (Garbage(b"<html>"), "bad_response"),
        (Garbage(b'{"segments": []}'), "bad_response"),
    ],
)
async def test_failures(fake: tuple[FakeMlxAudio, str], step: Fail | Garbage, kind: str) -> None:
    server, url = fake
    server.script(step)
    asr = provider(url)
    with pytest.raises(ProviderError) as failed:
        await asr.transcribe(SPEECH)
    await asr.aclose()
    assert failed.value.kind == kind


async def test_a_backend_that_is_down_is_unavailable() -> None:
    asr = provider(f"http://127.0.0.1:{free_port()}")
    with pytest.raises(ProviderError) as failed:
        await asr.transcribe(SPEECH)
    await asr.aclose()
    assert (failed.value.kind, failed.value.retryable) == ("unavailable", True)


async def test_the_transcript_deadline_abandons_the_request(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script(Hold(gate), Transcribe("late"))
    asr = provider(url, first_item_ms=150)
    with pytest.raises(ProviderTimeout) as failed:
        await asr.transcribe(SPEECH)
    assert failed.value.stage == "first_item"
    await until(lambda: server.disconnects == 1)
    gate.release()
    await asr.aclose()


async def test_the_total_deadline(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script(Hold(gate), Transcribe("late"))
    asr = provider(url, first_item_ms=2000, total_ms=150)
    with pytest.raises(ProviderTimeout) as failed:
        await asr.transcribe(SPEECH)
    assert failed.value.stage == "total"
    gate.release()
    await asr.aclose()


async def test_the_connect_deadline() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("timed out", request=request)

    config = MlxAudioAsrConfig.model_validate({"type": "mlx_audio_transcription"})
    asr = MlxAudioASR(config, transport=httpx2.MockTransport(slow))
    with pytest.raises(ProviderTimeout) as failed:
        await asr.transcribe(SPEECH)
    await asr.aclose()
    assert failed.value.stage == "connect"


async def test_cancellation_closes_the_request(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script(Hold(gate), Transcribe("never"))
    asr = provider(url)
    task = asyncio.create_task(asr.transcribe(SPEECH))
    await gate.reached()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await until(lambda: server.disconnects == 1)
    gate.release()
    assert await asr.transcribe(SPEECH)  # the provider still works
    await asr.aclose()


async def test_cancellation_while_waiting_for_a_slot(fake: tuple[FakeMlxAudio, str]) -> None:
    server, url = fake
    gate = Gate()
    server.script(Hold(gate), Transcribe("first"))
    config = MlxAudioAsrConfig.model_validate(
        {"type": "mlx_audio_transcription", "base_url": url, "max_in_flight": 1}
    )
    asr = MlxAudioASR(config)
    first = asyncio.create_task(asr.transcribe(SPEECH))
    await gate.reached()
    queued = asyncio.create_task(asr.transcribe(SPEECH))
    await asyncio.sleep(0.05)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert len(server.uploads) == 1
    gate.release()
    assert (await first).text == "first"
    await asr.aclose()


async def test_readiness_needs_the_model_loaded_and_warmup_loads_it(
    fake: tuple[FakeMlxAudio, str],
) -> None:
    server, url = fake
    asr = provider(url)
    state = await asr.ready()
    assert (state.ready, state.detail) == (False, "model not loaded")
    await asr.warmup()
    assert (await asr.ready()).ready
    (upload,) = server.uploads
    rate, _, _, frames = upload.audio()
    assert (rate, len(frames)) == (16000, 32000)  # a one-second clip
    await asr.aclose()
    down = provider(f"http://127.0.0.1:{free_port()}")
    assert not (await down.ready()).ready
    await down.aclose()


def test_parse_and_wav_helpers() -> None:
    body = b'{"text": " hi ", "language": "English", "segments": []}'
    assert parse_transcript(body, 120).language == "en"
    with pytest.raises(ProviderError):
        parse_transcript(b"[1, 2]", 0)
    header = wav_file(b"\x00\x00" * 10)[:44]
    assert header[:4] == b"RIFF"
    assert header[8:16] == b"WAVEfmt "


async def until(condition: Callable[[], bool], timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not condition():  # noqa: ASYNC110 (state set by the server's tasks)
            await asyncio.sleep(0.002)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]

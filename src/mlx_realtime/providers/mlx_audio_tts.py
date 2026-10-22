"""``MlxAudioTTS``: the TTS provider for ``mlx_audio.server`` (ARCHITECTURE §5.2, §5.3).

One streamed ``POST /v1/audio/speech`` per segment, JSON: ``model``, ``input``, ``voice``,
``lang_code``, ``response_format: "pcm"``, ``stream: true``, ``streaming_interval``, and the
sampling fields only when configured. Without ``stream`` the server takes a slower batch path
and sends nothing until synthesis ends, so every request streams. The body is chunked raw
s16le mono PCM (24 kHz), read under the request's deadlines: the first chunk, the idle gap
between chunks, and the total. Chunks are passed on as they arrive, whole samples only.

The backend's behaviour (mlx-audio 0.6.1, checked in M0) shapes the provider:

- The server runs every stream on one worker thread and never batches them, so the gateway
  keeps at most ``max_in_flight`` requests open (default 1) and queues the rest itself, where
  the wait is measured (``SpeechTiming.queued_s``, the turn timeline) and bounded
  (``queue_timeout_ms``, then ``overloaded``).
- Each request first has the worker confirm its model is loaded, and models load lazily, so a
  cold model costs seconds: the warmup speaks one Chinese and one English sentence per
  configured voice.
- Errors before the audio are FastAPI's ``{"detail": ...}`` with a status (404: unknown model;
  500: the model failed to load). A failure inside generation cuts the chunked body after the
  200 status (M0 A1 saw it with a Kokoro voice name); it is a ``bad_response``, as is a stream
  that ends without audio or inside a sample.
- The worker checks for cancellation once per chunk and sees a client disconnect at once, so
  closing the response is the cancel signal.

There is one provider per model and role, sharing one ``SpeechServer`` (client and slots):

- ``qwen3_tts`` (primary): ``voice`` is the session voice's speaker; ``lang_code`` is the
  segment's language tag through ``lang_codes`` (``chinese``, ``english``) unless the voice
  fixes one. An unknown lang_code would silently become ``auto``.
- ``kokoro`` (fallback): ``voice`` is the path of the voice pack inside the pinned snapshot
  (``voices/<name>.safetensors``): given a name, mlx-audio looks the pack up in a hardcoded
  Hub repository, which fails offline. ``lang_code`` is ``z`` (Mandarin, needs misaki[zh]) or
  ``a`` (US English); the voice is the session voice's fallback voice for that language.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from pathlib import Path
from typing import Literal, cast

import httpx2

from mlx_realtime.audio.pcm import SAMPLE_WIDTH
from mlx_realtime.config import MlxAudioSpeechConfig, TtsFamily, TtsModelConfig
from mlx_realtime.providers.base import (
    Lang,
    PCMChunk,
    ProviderError,
    Readiness,
    SpeechTiming,
)
from mlx_realtime.providers.http import (
    Deadlines,
    RequestClock,
    Slots,
    new_client,
    status_error,
    transport_error,
)
from mlx_realtime.providers.voices import SEGMENT_LANG, VoiceBook

logger = logging.getLogger(__name__)

SPEECH = "/v1/audio/speech"
MODELS = "/v1/models"

FAMILY_LANG_CODES: dict[TtsFamily, dict[Lang, str]] = {
    "qwen3_tts": {"zh": "chinese", "en": "english"},
    "kokoro": {"zh": "z", "en": "a"},
}
_SAMPLING = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")

type Role = Literal["primary", "fallback"]
type Json = dict[str, object]


class SpeechServer:
    """The TTS backend its models share: one HTTP client and one in-flight limit."""

    def __init__(
        self, config: MlxAudioSpeechConfig, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self.sample_rate = config.sample_rate
        self.slots = Slots(config.max_in_flight, config.queue_timeout_ms / 1000)
        self._base_url = config.base_url
        self._connect_s = config.primary.timeouts.connect_ms / 1000
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None

    def client(self) -> httpx2.AsyncClient:
        if self._http is None:
            self._http = new_client(self._base_url, self._connect_s, self._transport)
        return self._http

    async def loaded_models(self) -> set[str] | Readiness:
        """The models the server has loaded, or why that is unknown."""
        try:
            response = await self.client().get(MODELS)
        except httpx2.HTTPError as error:
            return Readiness(ready=False, detail=str(transport_error(error)))
        if response.status_code != 200:
            return Readiness(ready=False, detail=f"{MODELS} answered {response.status_code}")
        try:
            listed = cast(dict[str, list[dict[str, object]]], response.json())["data"]
            return {str(entry.get("id")) for entry in listed}
        except (ValueError, KeyError, TypeError, AttributeError):
            return Readiness(ready=False, detail=f"{MODELS}: unexpected answer")

    async def aclose(self) -> None:
        if self._http is not None:
            http, self._http = self._http, None
            await http.aclose()


class MlxAudioTTS:
    """One model of the TTS backend, in one role, as a ``TTSProvider`` and ``Backend``."""

    def __init__(
        self,
        server: SpeechServer,
        config: TtsModelConfig,
        voices: VoiceBook,
        *,
        role: Role,
        warmup_texts: Mapping[Lang, str],
        voices_dir: Path | None = None,
    ) -> None:
        self._server = server
        self._config = config
        self._voices = voices
        self._role: Role = role
        self._warmup_texts = dict(warmup_texts)
        self._voices_dir = Path(config.voices_dir) if config.voices_dir else voices_dir
        self._lang_codes = {**FAMILY_LANG_CODES[config.family], **config.lang_codes}
        self._deadlines = Deadlines.of(config.timeouts)
        self._warmup_deadlines = Deadlines.warmup(config.timeouts)

    @property
    def model(self) -> str:
        return self._config.model

    def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        """``speed`` is always 1.0 here (D14: the protocol accepts nothing else)."""
        return self._speak(text, voice, lang, timing or SpeechTiming(), self._deadlines)

    def request_body(self, text: str, voice: str, lang: Lang) -> Json:
        body: Json = {
            "model": self._config.model,
            "input": text,
            "voice": self._voice_field(voice, lang),
            "lang_code": self._lang_code(voice, lang),
            "response_format": "pcm",
            "stream": True,
            "streaming_interval": self._config.streaming_interval,
        }
        for name in _SAMPLING:
            value = getattr(self._config, name)
            if value is not None:
                body[name] = value
        return body

    async def warmup(self) -> None:
        """Speak one sentence per configured voice and language: the server loads the model on
        its first request, and a request for every voice catches a misconfigured one."""
        spoken: set[tuple[object, object]] = set()
        for voice in self._voices.voices():
            for lang, text in self._warmup_texts.items():
                body = self.request_body(text, voice.name, lang)
                key = (body["voice"], body["lang_code"])
                if key in spoken:
                    continue
                spoken.add(key)
                stream = self._speak(text, voice.name, lang, SpeechTiming(), self._warmup_deadlines)
                async with aclosing(stream):
                    async for _ in stream:
                        pass
        logger.info("tts warm", extra={"model": self.model, "requests": len(spoken)})

    async def ready(self) -> Readiness:
        loaded = await self._server.loaded_models()
        if isinstance(loaded, Readiness):
            return loaded
        if self._config.model not in loaded:
            return Readiness(ready=False, detail=f"{self._role} model not loaded")
        return Readiness(ready=True, detail="ok")

    async def aclose(self) -> None:
        await self._server.aclose()

    def _voice_field(self, name: str, lang: Lang) -> str:
        voice = self._voices.voice(name)
        chosen = voice.speaker if self._role == "primary" else voice.fallback.get(lang)
        if self._config.family != "kokoro":
            return chosen or voice.speaker
        if chosen is None:
            raise ProviderError("internal", f"tts: voice '{voice.name}' has no {lang} voice pack")
        if self._voices_dir is None:
            raise ProviderError(
                "unavailable",
                f"tts: no voice packs for {self.model}: download it (mlxrt models --mlx) "
                "or set voices_dir",
            )
        pack = self._voices_dir / f"{chosen}.safetensors"
        if not pack.is_file():
            raise ProviderError("unavailable", f"tts: voice pack not found: {pack}")
        return str(pack)

    def _lang_code(self, name: str, lang: Lang) -> str:
        policy = self._voices.voice(name).lang
        if self._role == "primary" and policy != SEGMENT_LANG:
            return policy
        return self._lang_codes[lang]

    async def _speak(
        self, text: str, voice: str, lang: Lang, timing: SpeechTiming, deadlines: Deadlines
    ) -> AsyncGenerator[PCMChunk]:
        body = self.request_body(text, voice, lang)
        timing.model = self.model
        timing.fallback = self._role == "fallback"
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        async with self._server.slots.slot():
            timing.queued_s += loop.time() - queued_at
            clock = RequestClock(deadlines)
            client = self._server.client()
            request = client.build_request(
                "POST",
                SPEECH,
                content=json.dumps(body, ensure_ascii=False).encode(),
                headers={"content-type": "application/json"},
                timeout=httpx2.Timeout(None, connect=deadlines.connect_s),
            )
            try:
                async with clock.waiting():
                    response = await client.send(request, stream=True)
            except httpx2.HTTPError as error:
                raise transport_error(error) from error
            try:
                if response.status_code != 200:
                    raise status_error(response.status_code, await _error_body(response, clock))
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("audio/pcm"):
                    raise _bad(f"unexpected content type {content_type!r}")
                carry = b""
                received = 0
                async with aclosing(response.aiter_raw()) as pieces:
                    while (piece := await _next_piece(pieces, clock)) is not None:
                        data = carry + piece
                        whole = len(data) - len(data) % SAMPLE_WIDTH
                        carry = data[whole:]
                        if whole:
                            received += whole
                            clock.item()
                            yield PCMChunk(data[:whole], self._server.sample_rate)
                if carry:
                    raise _bad("the stream ended inside a sample")
                if not received:
                    raise _bad("the stream carried no audio")
            finally:
                await response.aclose()


async def _error_body(response: httpx2.Response, clock: RequestClock) -> bytes:
    try:
        async with clock.waiting():
            return await response.aread()
    except httpx2.HTTPError as error:
        raise transport_error(error) from error


async def _next_piece(pieces: AsyncGenerator[bytes], clock: RequestClock) -> bytes | None:
    """The next piece of the body; a cut stream is the server failing inside generation."""
    try:
        async with clock.waiting():
            return await anext(pieces, None)
    except httpx2.HTTPError as error:
        raise _bad(f"the stream was cut: {error}") from error


def _bad(message: str) -> ProviderError:
    return ProviderError("bad_response", f"tts: {message}")

"""``MlxAudioTTS``: the TTS provider for ``mlx_audio.server`` (ARCHITECTURE §5.2, §5.3).

One streamed ``POST /v1/audio/speech`` per segment, JSON: ``model``, ``input``, ``voice``,
``lang_code``, ``response_format: "pcm"``, ``stream: true``, ``streaming_interval``, the
``max_tokens`` of ``token_cap``, and the sampling fields only when configured. Without
``stream`` the server takes a slower batch path and sends nothing until synthesis ends, so every
request streams. The body is chunked raw s16le mono PCM (24 kHz), read under the request's
deadlines: the first chunk, the idle gap between chunks, and the total. Chunks are passed on as
they arrive, whole samples only.

The backend's behaviour (mlx-audio 0.6.1, checked in M0) shapes the provider:

- The server runs every stream on one worker thread and never batches them, so the gateway
  keeps at most ``max_in_flight`` requests open (default 1) and queues the rest itself, where
  the wait is measured (``SpeechTiming.queued_s``, the turn timeline), ordered (``scheduling``:
  earliest ``SpeechTiming.deadline`` first, M6) and bounded (``queue_timeout_ms`` past the
  deadline, then ``overloaded``).
- Each request first has the worker confirm its model is loaded, and models load lazily, so a
  cold model costs seconds: the warmup loads each model before the first call needs it.
- Errors before the audio are FastAPI's ``{"detail": ...}`` with a status (404: unknown model;
  500: the model failed to load). A failure inside generation cuts the chunked body after the
  200 status (a speaker the model does not have, for one); it is a ``bad_response``, as is a
  stream that ends without audio or inside a sample.
- The worker checks for cancellation once per chunk and sees a client disconnect at once, so
  closing the response is the cancel signal.

A verbatim line (D16; ``SpeechTiming.cacheable``) repeats, so the primary keeps the raw chunks of
a take that ended normally and was read to its end, keyed by the request body, and replays them
before taking a slot (``line_cache`` entries, LRU; ``SpeechTiming.cached``). A take that ran into
``max_tokens`` is not kept.

There is one provider per model and role (primary, fallback), sharing one ``SpeechServer``
(client and slots). Both models are builds of Qwen3-TTS-12Hz-0.6B-CustomVoice (D13: 8-bit and
4-bit) with the same speakers, and both are asked the same way: ``voice`` is the session
voice's speaker for the segment's language; ``lang_code`` is the segment's language tag through
``lang_codes`` (``chinese``, ``english``) unless the voice fixes one. An unknown lang_code would
silently become ``auto``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import OrderedDict
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Literal, cast

import httpx2

from mlx_realtime.audio.pcm import SAMPLE_WIDTH
from mlx_realtime.config import MlxAudioSpeechConfig, TokenCapConfig, TtsModelConfig
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
from mlx_realtime.text.lang import is_cjk

logger = logging.getLogger(__name__)

SPEECH = "/v1/audio/speech"
MODELS = "/v1/models"
SERVER_MAX_TOKENS = 1200  # mlx_audio.server's default max_tokens (codec frames)
CODEC_FRAME_S = 0.08  # Qwen3-TTS-12Hz: one codec frame, 12.5 per second

LANG_CODES: dict[Lang, str] = {"zh": "chinese", "en": "english"}
"""Qwen3-TTS's lang_code for each segment language tag (``lang_codes`` overrides them)."""
_SAMPLING = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")

type Role = Literal["primary", "fallback"]
type Json = dict[str, object]


class SpeechServer:
    """The TTS backend its models share: one HTTP client and one in-flight limit."""

    def __init__(
        self, config: MlxAudioSpeechConfig, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self.sample_rate = config.sample_rate
        self.slots = Slots(
            config.max_in_flight, config.queue_timeout_ms / 1000, policy=config.scheduling
        )
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
        line_cache: int = 0,
    ) -> None:
        self._server = server
        self._config = config
        self._voices = voices
        self._role: Role = role
        self._warmup_texts = dict(warmup_texts)
        self._lang_codes = {**LANG_CODES, **config.lang_codes}
        self._deadlines = Deadlines.of(config.timeouts)
        self._warmup_deadlines = Deadlines.warmup(config.timeouts)
        # Verbatim lines: request body -> the raw chunks of a finished take (LRU). Only the
        # primary keeps one (D13: the fallback is the degraded path and never stores or reads).
        self._line_cache = line_cache if role == "primary" else 0
        self._lines = OrderedDict[bytes, tuple[PCMChunk, ...]]()

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def queue_depth(self) -> int:
        """Requests waiting for the backend's stream slot now (every model of the backend)."""
        return self._server.slots.waiting

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
        """The request for one segment; the same on either model but for ``model`` (D13)."""
        chosen = self._voices.voice(voice)
        lang_code = self._lang_codes[lang] if chosen.lang == SEGMENT_LANG else chosen.lang
        body: Json = {
            "model": self._config.model,
            "input": text,
            "voice": chosen.speaker_for(lang),
            "lang_code": lang_code,
            "response_format": "pcm",
            "stream": True,
            "streaming_interval": self._config.streaming_interval,
        }
        for name in _SAMPLING:
            value = getattr(self._config, name)
            if value is not None:
                body[name] = value
        if self._config.token_cap is not None:
            body["max_tokens"] = token_cap(text, self._config.token_cap, self._config.max_tokens)
        return body

    async def warmup(self) -> None:
        """Load the model before the first call needs it: the server loads a model on its first
        request. The primary speaks one Chinese and one English sentence in every configured
        voice, so a speaker the model lacks keeps the gateway not ready. The fallback has the
        same speakers (D13), so it speaks the two sentences in the default voice only."""
        voices = self._voices.voices()
        if self._role == "fallback":
            voices = (self._voices.voice(self._voices.default),)
        spoken: set[tuple[object, object]] = set()
        for voice in voices:
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

    async def _speak(
        self, text: str, voice: str, lang: Lang, timing: SpeechTiming, deadlines: Deadlines
    ) -> AsyncGenerator[PCMChunk]:
        body = self.request_body(text, voice, lang)
        timing.model = self.model
        timing.fallback = self._role == "fallback"
        content = json.dumps(body, ensure_ascii=False).encode()
        keep: list[PCMChunk] | None = None
        if timing.cacheable and self._line_cache:
            if (line := self._lines.get(content)) is not None:
                self._lines.move_to_end(content)
                timing.cached = True
                for chunk in line:
                    yield chunk
                    await asyncio.sleep(0)  # a live take yields to the loop between chunks
                return
            keep = []
        loop = asyncio.get_running_loop()
        queued_at = loop.time()
        async with self._server.slots.slot(deadline=timing.deadline, slack_s=timing.slack_s):
            timing.queued_s += loop.time() - queued_at
            clock = RequestClock(deadlines)
            client = self._server.client()
            request = client.build_request(
                "POST",
                SPEECH,
                content=content,
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
                            chunk = PCMChunk(data[:whole], self._server.sample_rate)
                            if keep is not None:
                                keep.append(chunk)
                            yield chunk
                if carry:
                    raise _bad("the stream ended inside a sample")
                if not received:
                    raise _bad("the stream carried no audio")
                if keep is not None and not self._capped(received, body):
                    self._store(content, keep)
            finally:
                await response.aclose()

    def _capped(self, received: int, body: Json) -> bool:
        """The take ran into ``max_tokens``: the server ends such a stream normally and says
        nothing, so it is inferred from the codec frames received (80 ms each)."""
        frames = received / SAMPLE_WIDTH / self._server.sample_rate / CODEC_FRAME_S
        return round(frames) >= cast(int, body.get("max_tokens", SERVER_MAX_TOKENS))

    def _store(self, key: bytes, chunks: list[PCMChunk]) -> None:
        self._lines[key] = tuple(chunks)
        self._lines.move_to_end(key)
        while len(self._lines) > self._line_cache:
            self._lines.popitem(last=False)


def token_cap(text: str, cap: TokenCapConfig, limit: int | None) -> int:
    """The codec frames a segment may take (S3: a runaway stops here instead of at 1200)."""
    cjk = sum(1 for char in text if is_cjk(char))
    other = sum(1 for char in text if char.isalnum() and not is_cjk(char))
    frames = math.ceil(cap.base + cap.per_cjk * cjk + cap.per_other * other)
    return min(frames, limit or SERVER_MAX_TOKENS)


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

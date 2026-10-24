"""``MlxAudioASR``: the ASR provider for ``mlx_audio.server`` (ARCHITECTURE §5.2, §5.3).

``POST /v1/audio/transcriptions``, multipart: the turn as a WAV file (16 kHz mono s16),
``model``, ``language`` when the session sets one (omitted: auto-detect), ``context`` when
there is text to bias the model with, and ``response_format: verbose_json``. The context is the
turn's ``transcription.prompt``, or else ``providers.asr.context``; with neither, the form has
no ``context`` field. Stock ``mlx_audio.server`` drops it for Qwen3-ASR; the backend wrapper
``mlxrt_audio.serve`` hands it to the model as its system prompt. The answer is one JSON
object, ``text`` plus ``language`` (a list of names, as Qwen3-ASR reports them) and
``segments``; it is the provider's single item, bounded by the first-item deadline. The server
loads a model on its first request, so the warmup transcribes a short clip.
"""

from __future__ import annotations

import io
import json
import math
import wave
from typing import cast

import httpx2
import numpy as np

from mlx_realtime.audio.pcm import CANONICAL_RATE, SAMPLE_WIDTH, bytes_to_ms
from mlx_realtime.config import AsrTokenCapConfig, MlxAudioAsrConfig
from mlx_realtime.providers.base import ProviderError, Readiness, Transcript
from mlx_realtime.providers.http import (
    Deadlines,
    RequestClock,
    Slots,
    new_client,
    status_error,
    transport_error,
)

TRANSCRIPTIONS = "/v1/audio/transcriptions"
MODELS = "/v1/models"

# Qwen3-ASR names its languages (its config's support_languages); sessions use ISO 639 codes.
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
_LANGUAGE_CODES = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


class MlxAudioASR:
    def __init__(
        self, config: MlxAudioAsrConfig, *, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self._config = config
        self._deadlines = Deadlines.of(config.timeouts)
        self._warmup_deadlines = Deadlines.warmup(config.timeouts)
        self._slots = Slots(config.max_in_flight, config.queue_timeout_ms / 1000)
        self._transport = transport
        self._http: httpx2.AsyncClient | None = None

    async def transcribe(
        self, pcm16_16k: bytes, *, language: str | None = None, context: str | None = None
    ) -> Transcript:
        return await self._transcribe(
            pcm16_16k, language, context or self._config.context, self._deadlines
        )

    async def warmup(self) -> None:
        """A one-second clip: the server loads the model on its first request."""
        await self._transcribe(warmup_clip(), None, None, self._warmup_deadlines)

    async def ready(self) -> Readiness:
        """The server answers, and its list of loaded models holds ours."""
        try:
            response = await self._client().get(MODELS)
        except httpx2.HTTPError as error:
            return Readiness(ready=False, detail=str(transport_error(error)))
        if response.status_code != 200:
            return Readiness(ready=False, detail=f"{MODELS} answered {response.status_code}")
        try:
            listed = cast(dict[str, list[dict[str, object]]], response.json())["data"]
            loaded = {entry.get("id") for entry in listed}
        except (ValueError, KeyError, TypeError, AttributeError):
            return Readiness(ready=False, detail=f"{MODELS}: unexpected answer")
        if self._config.model not in loaded:
            return Readiness(ready=False, detail="model not loaded")
        return Readiness(ready=True, detail="ok")

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def _transcribe(
        self, pcm: bytes, language: str | None, context: str | None, deadlines: Deadlines
    ) -> Transcript:
        audio_ms = bytes_to_ms(len(pcm), CANONICAL_RATE)
        if not pcm:
            return Transcript(text="", language=None, audio_ms=0)
        form = {"model": self._config.model, "response_format": "verbose_json"}
        if language:
            form["language"] = LANGUAGE_NAMES.get(language.lower(), language)
        if context:
            form["context"] = context
        if (cap := self._config.token_cap) is not None:
            form["max_tokens"] = str(token_cap(audio_ms, cap))
        files = {"file": ("audio.wav", wav_file(pcm), "audio/wav")}
        async with self._slots.slot():
            clock = RequestClock(deadlines)
            client = self._client()
            request = client.build_request("POST", TRANSCRIPTIONS, data=form, files=files)
            try:
                async with clock.waiting():
                    response = await client.send(request, stream=True)
                try:
                    async with clock.waiting():
                        body = await response.aread()
                finally:
                    await response.aclose()
            except httpx2.HTTPError as error:
                raise transport_error(error) from error
        if response.status_code != 200:
            raise status_error(response.status_code, body)
        return parse_transcript(body, audio_ms)

    def _client(self) -> httpx2.AsyncClient:
        if self._http is None:
            self._http = new_client(
                self._config.base_url, self._deadlines.connect_s, self._transport
            )
        return self._http


def token_cap(audio_ms: int, cap: AsrTokenCapConfig) -> int:
    """The tokens a transcription of ``audio_ms`` of audio may take (a runaway stops here)."""
    return math.ceil(cap.base + cap.per_s * audio_ms / 1000)


def parse_transcript(body: bytes, audio_ms: int) -> Transcript:
    try:
        document: object = json.loads(body)
    except ValueError:
        raise _bad("the answer is not JSON") from None
    if not isinstance(document, dict):
        raise _bad("the answer is not an object")
    fields = cast(dict[str, object], document)
    text = fields.get("text")
    if not isinstance(text, str):
        raise _bad("the answer has no text")
    return Transcript(
        text=text.strip(), language=_language(fields.get("language")), audio_ms=audio_ms
    )


def _language(reported: object) -> str | None:
    """Qwen3-ASR reports a list of language names; the first one, as an ISO code if known."""
    if isinstance(reported, list) and reported:
        reported = cast(list[object], reported)[0]
    if not isinstance(reported, str) or not reported:
        return None
    return _LANGUAGE_CODES.get(reported.lower(), reported)


def wav_file(pcm: bytes) -> bytes:
    """Canonical PCM as a WAV file: 16 kHz, mono, 16-bit."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(CANONICAL_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def warmup_clip() -> bytes:
    """One second of a quiet 220 Hz tone (about -30 dBFS)."""
    times = np.arange(CANONICAL_RATE, dtype=np.float64) / CANONICAL_RATE
    tone = 1000 * np.sin(2 * math.pi * 220 * times)
    return tone.astype("<i2").tobytes()


def _bad(message: str) -> ProviderError:
    return ProviderError("bad_response", f"asr: {message}")

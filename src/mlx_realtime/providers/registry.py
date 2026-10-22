"""Builds the configured providers: the HTTP providers for the model backends, or the mocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mlx_realtime.config import (
    ASRConfig,
    LLMConfig,
    MlxAudioAsrConfig,
    MlxLmChatConfig,
    MockTTSConfig,
    ProvidersConfig,
)
from mlx_realtime.providers.base import ASRProvider, Backend, LLMProvider, TTSProvider
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR, SineTTS


class LLMBackend(LLMProvider, Backend, Protocol): ...


class ASRBackend(ASRProvider, Backend, Protocol): ...


class TTSBackend(TTSProvider, Backend, Protocol): ...


@dataclass(frozen=True, slots=True)
class Providers:
    llm: LLMBackend
    asr: ASRBackend
    tts: TTSBackend

    def backends(self) -> dict[str, Backend]:
        return {"llm": self.llm, "asr": self.asr, "tts": self.tts}


def build_providers(config: ProvidersConfig, *, warmup_instructions: str = "") -> Providers:
    """``warmup_instructions`` are the session defaults' instructions (the LLM warmup prompt)."""
    return Providers(
        llm=build_llm(config.llm, warmup_instructions=warmup_instructions),
        asr=build_asr(config.asr),
        tts=build_tts(config.tts),
    )


def build_llm(config: LLMConfig, *, warmup_instructions: str = "") -> LLMBackend:
    if isinstance(config, MlxLmChatConfig):
        return MlxLmChatLLM(config, warmup_instructions=warmup_instructions)
    return EchoLLM(
        first_token_delay_s=config.first_token_delay_ms / 1000,
        token_delay_s=config.token_delay_ms / 1000,
        greeting=config.greeting,
    )


def build_asr(config: ASRConfig) -> ASRBackend:
    if isinstance(config, MlxAudioAsrConfig):
        return MlxAudioASR(config)
    return ScriptedASR(config.transcripts, delay_s=config.delay_ms / 1000)


def build_tts(config: MockTTSConfig) -> TTSBackend:
    return SineTTS(
        ms_per_char=config.ms_per_char,
        chunk_ms=config.chunk_ms,
        first_chunk_delay_s=config.first_chunk_delay_ms / 1000,
        chunk_delay_s=config.chunk_delay_ms / 1000,
        frequency_hz=config.frequency_hz,
    )

"""Builds the configured providers. M1 has one implementation per role: the mocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mlx_realtime.config import MockASRConfig, MockLLMConfig, MockTTSConfig, ProvidersConfig
from mlx_realtime.providers.base import ASRProvider, Backend, LLMProvider, TTSProvider
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


def build_providers(config: ProvidersConfig) -> Providers:
    return Providers(
        llm=build_llm(config.llm), asr=build_asr(config.asr), tts=build_tts(config.tts)
    )


def build_llm(config: MockLLMConfig) -> LLMBackend:
    return EchoLLM(
        first_token_delay_s=config.first_token_delay_ms / 1000,
        token_delay_s=config.token_delay_ms / 1000,
        greeting=config.greeting,
    )


def build_asr(config: MockASRConfig) -> ASRBackend:
    return ScriptedASR(config.transcripts, delay_s=config.delay_ms / 1000)


def build_tts(config: MockTTSConfig) -> TTSBackend:
    return SineTTS(
        ms_per_char=config.ms_per_char,
        chunk_ms=config.chunk_ms,
        first_chunk_delay_s=config.first_chunk_delay_ms / 1000,
        chunk_delay_s=config.chunk_delay_ms / 1000,
        frequency_hz=config.frequency_hz,
    )

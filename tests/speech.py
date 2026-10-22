"""Speech fixtures (tests/fixtures/README.md) and the downloaded ONNX models, for the tests that
run the real VAD models."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from mlx_realtime.audio.pcm import CANONICAL_RATE, WIRE_RATE
from mlx_realtime.audio.resample import Resampler

REPO = Path(__file__).parents[1]
SPEECH = Path(__file__).parent / "fixtures" / "speech"
MODELS = REPO / "models"
SILERO = MODELS / "silero_vad.onnx"
SMART_TURN = MODELS / "smart-turn-v3.2-gpu.onnx"
SMART_TURN_INT8 = MODELS / "smart-turn-v3.2-cpu.onnx"

ZH = "zh_dinosaurs"  # 但鸟类还是有很多地方跟恐龙相似。 speech 0.32-3.66 s
EN = "en_dinosaurs"  # Many people don't think about them as dinosaurs... speech 0.24-3.92 s

needs_models = pytest.mark.skipif(
    not (SILERO.is_file() and SMART_TURN.is_file()),
    reason="the ONNX models are missing: run `uv run --locked mlxrt models`",
)


def clip(name: str) -> bytes:
    """A fixture as canonical PCM: 16 kHz mono s16le."""
    with wave.open(str(SPEECH / f"{name}.wav")) as file:
        assert (file.getframerate(), file.getnchannels(), file.getsampwidth()) == (16000, 1, 2)
        return file.readframes(file.getnframes())


def silence(ms: int, rate: int = CANONICAL_RATE) -> bytes:
    return bytes(2 * rate * ms // 1000)


def to_wire(pcm16k: bytes) -> bytes:
    """Canonical PCM as client audio: ``audio/pcm`` at 24 kHz."""
    resampler = Resampler(CANONICAL_RATE, WIRE_RATE)
    return resampler.process(pcm16k) + resampler.flush()

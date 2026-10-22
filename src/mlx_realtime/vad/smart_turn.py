"""Smart Turn v3.2: the probability that the speaker has finished their turn.

Input preparation follows the reference ``inference.py`` of pipecat-ai/smart-turn and pipecat's
``LocalSmartTurnAnalyzerV3``: the last 8 s of the turn at 16 kHz, zero-padded at the front so
the speech sits at the end of the window, then Whisper log-mel features [80, 800]. The model's
``logits`` output [1, 1] is already a sigmoid probability. Model files (BSD-2-Clause) come from
HF ``pipecat-ai/smart-turn-v3`` at f766f81d, downloaded by ``mlxrt models``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from mlx_realtime.vad.onnx import OnnxModel
from mlx_realtime.vad.whisper_features import compute_whisper_log_mel_features

RATE = 16_000
WINDOW_SAMPLES = 8 * RATE


def turn_window(pcm: bytes) -> npt.NDArray[np.float32]:
    """PCM16 audio as float32: its last 8 s, front-padded with zeros to exactly 8 s."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    if samples.size >= WINDOW_SAMPLES:
        return samples[-WINDOW_SAMPLES:]
    return np.pad(samples, (WINDOW_SAMPLES - samples.size, 0))


def turn_features(pcm: bytes) -> npt.NDArray[np.float32]:
    """The model input for one turn: [1, 80, 800]."""
    return compute_whisper_log_mel_features(turn_window(pcm))[np.newaxis, ...]


class SmartTurnModel:
    """One ONNX session (G3: fp32 file, 2 intra-op threads, no spinning). One thread at a time."""

    def __init__(self, path: Path, *, intra_op_threads: int) -> None:
        self._model = OnnxModel(path, intra_op_threads=intra_op_threads, spinning=False)

    def probability(self, pcm: bytes) -> float:
        """Features and inference, both in the calling (worker) thread."""
        (logits,) = self._model.run({"input_features": turn_features(pcm)})
        return float(logits[0, 0])

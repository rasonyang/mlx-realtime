# Ported for mlx-realtime from pipecat v1.13.0, src/pipecat/audio/vad/silero.py: the numpy-only
# ONNX wrapper SileroOnnxModel, and the model-state reset policy of
# SileroVADAnalyzer._maybe_reset_model. Changes: one stream per object, at 16 kHz only; frames
# arrive as PCM16 bytes; the ONNX session is shared through ``OnnxModel``; the VAD state the reset
# waits for comes from the caller (the speech gate); type annotations. The model file
# (snakers4/silero-vad v6.2.3, MIT) is downloaded by `mlxrt models`, not vendored. The original
# notices follow and are kept as required.
#
# pipecat LICENSE (BSD 2-Clause), reproduced in full:
#
#   Copyright (c) 2024–2026, Daily
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ---------------------------------------------------------------------------
#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Silero VAD v6.2.3 for one audio stream: 512-sample frames at 16 kHz, plus 64 samples of
context, with the recurrent state [2, 1, 128] carried from frame to frame.

Model-state resets (pipecat's notes): non-speech shifts the recurrent state so the model
under-reports the speech that follows. The state is therefore reset about every two seconds of
audio, but only during clear non-speech: the VAD is quiet and the last five windows (about
160 ms) all scored below 0.1. A reset during speech would make the model lose confidence and end
the speech early.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mlx_realtime.vad.base import FRAME_SAMPLES
from mlx_realtime.vad.onnx import OnnxModel

RATE = 16_000
CONTEXT_SAMPLES = 64

_SAMPLE_RATE_INPUT = np.array(RATE, dtype=np.int64)
_RESET_MAX_CONFIDENCE = 0.1
_RESET_LOW_CONFIDENCE_WINDOWS = 5


class SileroStream:
    """One session's Silero state. Not thread-safe: one call at a time (the Silero executor)."""

    def __init__(self, model: OnnxModel, *, reset_interval_s: float | None = 2.0) -> None:
        self._model = model
        self._reset_samples = None if reset_interval_s is None else reset_interval_s * RATE
        self._windows_since_reset = 0
        self._low_confidence_windows = 0
        self.resets = 0
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: bytes, *, in_speech: bool) -> float:
        """Speech probability of one 512-sample PCM16 frame; ``in_speech`` gates the reset."""
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / np.float32(32768.0)
        if samples.size != FRAME_SAMPLES:
            raise ValueError(f"a Silero frame is {FRAME_SAMPLES} samples, not {samples.size}")
        window: npt.NDArray[np.float32] = np.concatenate(
            (self._context, samples[np.newaxis, :]), axis=1
        )
        output, state = self._model.run(
            {"input": window, "state": self._state, "sr": _SAMPLE_RATE_INPUT}
        )
        self._state = state
        self._context = window[:, -CONTEXT_SAMPLES:]
        probability = float(output[0, 0])
        self._maybe_reset(probability, in_speech)
        return probability

    def _maybe_reset(self, probability: float, in_speech: bool) -> None:
        """Reset once a reset is due and the speaker is quiet; it applies from the next window."""
        if self._reset_samples is None:
            return
        self._windows_since_reset += 1
        if probability < _RESET_MAX_CONFIDENCE:
            self._low_confidence_windows += 1
        else:
            self._low_confidence_windows = 0
        if (
            self._windows_since_reset * FRAME_SAMPLES >= self._reset_samples
            and not in_speech
            and self._low_confidence_windows >= _RESET_LOW_CONFIDENCE_WINDOWS
        ):
            self.reset_states()
            self._windows_since_reset = 0
            self.resets += 1

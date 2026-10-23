# Ported for the mlx-realtime M0 spike S7 (throwaway) from pipecat v1.13.0,
# src/pipecat/audio/vad/silero.py (class SileroOnnxModel). Changes: no loguru,
# the caller passes the onnxruntime session (so the spike can vary its options),
# and a small batched-state helper (BatchedSilero) was added for the concurrency
# test. The original notices follow and are kept as required.
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

"""Numpy-only Silero VAD ONNX wrapper (no torch), after pipecat's SileroOnnxModel."""

from typing import cast

import numpy as np


class SileroOnnxModel:
    """ONNX runtime wrapper for the Silero VAD model (one stream, or B streams in lockstep)."""

    def __init__(self, session):
        self.session = session
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        if np.ndim(x) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")
        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)")
        if sr / np.shape(x)[1] > 31.25:
            raise ValueError("Input audio chunk is too short")
        return x, sr

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int) -> np.ndarray:
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256
        if np.shape(x)[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {np.shape(x)[-1]} "
                "(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )
        batch_size = np.shape(x)[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)
        ort_inputs = {"input": x, "state": self._state, "sr": np.array(sr, dtype="int64")}
        ort_outs = self.session.run(None, ort_inputs)
        out = cast(np.ndarray, ort_outs[0])
        self._state = cast(np.ndarray, ort_outs[1])

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return out


class BatchedSilero:
    """Spike-only: per-stream state/context held in arrays, any subset run as one batch (16 kHz)."""

    CONTEXT = 64
    FRAME = 512

    def __init__(self, session, n_streams: int):
        self.session = session
        self.state = np.zeros((2, n_streams, 128), dtype=np.float32)
        self.context = np.zeros((n_streams, self.CONTEXT), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)

    def run(self, idx: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """idx: stream indices [B]; frames: float32 [B, 512]. Returns probabilities [B]."""
        x = np.concatenate((self.context[idx], frames), axis=1)
        out, state = self.session.run(None, {"input": x, "state": self.state[:, idx], "sr": self.sr})
        self.state[:, idx] = state
        self.context[idx] = x[:, -self.CONTEXT :]
        return out[:, 0]

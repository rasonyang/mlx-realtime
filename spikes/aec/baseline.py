"""Numpy reference line-echo canceller for S11 (G7): PBFDAF + Geigel double-talk detector.

Status: smoke-tested only (A3 install check pushes 1 s through it). Correctness and tuning are
part of S11 part 2; do not read anything into its numbers before then.

Design (see spikes/results/a3_aec_research.md, "numpy baseline"):
- Partitioned-block frequency-domain adaptive filter, overlap-save, 50 % overlap
  (FFT size 2N), fully constrained gradient (time-domain taps beyond N zeroed per partition).
- Block = partition = N samples (default 160 = one 10 ms gateway frame at 16 kHz).
- Tail = P * N samples (default P = ceil(128 ms / 10 ms) = 13 partitions, 130 ms).
- Step normalised per bin by the far-end power summed over all partitions (the frequency-domain
  analogue of NLMS normalisation by the filter-length input energy).
- Geigel double-talk detector: double talk if max|near| over the block is at least
  `geigel_threshold` x max|far| over the last tail + 1 block (0.5 = the 6 dB hybrid-loss
  assumption), with a hangover. Adaptation is frozen during double talk.
- No delay estimation, no nonlinear processing, no comfort noise: the reference must already be
  aligned so that the echo falls inside the tail (a gateway-side delay estimator does that).

S11 part 2 (B4) validation: on white noise through G.168 m1 (ERL 10 dB, 20 ms delay) it reaches 20 dB
ERLE after 1 s and 60 dB after 5 s (mu 0.5). With band-limited input (the 3.8 kHz pre-filter) the
per-bin normalization by the far-end power alone lets empty stop-band bins, which hold only rounding
noise, take huge steps, and convergence collapses. `reg_rel` adds a floor of reg_rel x the mean
per-bin far-end power to the normalization; 0 keeps A3's original behaviour.
"""

from __future__ import annotations

import numpy as np


class PbfdafGeigel:
    def __init__(
        self,
        sample_rate: int = 16000,
        block: int = 160,
        tail_ms: float = 128.0,
        mu: float = 0.5,
        geigel_threshold: float = 0.5,
        hangover_ms: float = 30.0,
        eps: float = 1e-6,
        reg_rel: float = 0.0,
    ) -> None:
        self.reg_rel = reg_rel
        self.n = block
        self.p = max(1, int(np.ceil(tail_ms * sample_rate / 1000.0 / block)))
        self.m = 2 * block
        bins = self.m // 2 + 1
        self.w = np.zeros((self.p, bins), dtype=np.complex128)  # partition weights, FD
        self.x = np.zeros((self.p, bins), dtype=np.complex128)  # far-end spectra, newest first
        self.x_prev = np.zeros(block)
        self.far_hist = np.zeros(self.p * block + block)
        self.mu = mu
        self.eps = eps
        self.thr = geigel_threshold
        self.hang = max(0, int(round(hangover_ms * sample_rate / 1000.0 / block)))
        self.hang_left = 0
        self.double_talk = False  # state of the last block, for inspection

    @property
    def tail_samples(self) -> int:
        return self.p * self.n

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """Cancel one block. `near` (capture) and `far` (aligned reference): N samples each."""
        n, m = self.n, self.m
        if near.shape != (n,) or far.shape != (n,):
            raise ValueError(f"blocks must have shape ({n},)")
        d = near.astype(np.float64)
        f = far.astype(np.float64)

        self.far_hist = np.roll(self.far_hist, -n)
        self.far_hist[-n:] = f
        xk = np.fft.rfft(np.concatenate([self.x_prev, f]))
        self.x_prev = f
        self.x = np.roll(self.x, 1, axis=0)
        self.x[0] = xk

        y = np.fft.irfft(np.sum(self.x * self.w, axis=0), m)[n:]
        e = d - y

        near_peak = float(np.max(np.abs(d)))
        if near_peak > 0.0 and near_peak >= self.thr * float(np.max(np.abs(self.far_hist))):
            self.hang_left = self.hang + 1
        self.double_talk = self.hang_left > 0
        if self.hang_left > 0:
            self.hang_left -= 1

        if not self.double_talk:
            ek = np.fft.rfft(np.concatenate([np.zeros(n), e]))
            norm = np.sum(np.abs(self.x) ** 2, axis=0)
            norm = norm + self.eps + self.reg_rel * float(np.mean(norm))
            grad = np.conj(self.x) * (ek / norm)
            g = np.fft.irfft(grad, m, axis=1)
            g[:, n:] = 0.0  # gradient constraint (linear, not circular, convolution)
            self.w += self.mu * np.fft.rfft(g, m, axis=1)
        return e

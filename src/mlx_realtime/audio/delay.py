"""GCC-PHAT delay estimation between the echo canceller's input and its reference (M0 S11, A3).

The estimator watches how far the echo in the capture trails the reference the canceller was
fed: the residual delay. It runs on 8 kHz copies of both streams:

- a 2 s capture window every 0.5 s, cross-correlated with the reference from 1 s before the
  window to 0.3 s after it: lags of -300 ... +1000 ms (a window is evaluated once the capture is
  0.3 s past it, so that reference is there);
- PHAT weighting over 200-3600 Hz;
- only while the reference is active (at least half of the window's 10 ms frames above
  -56 dBov, 30 dB under a -26 dBov far end) and the correlation peak stands out (PSR >= 8);
- an estimate is confirmed when two consecutive valid windows agree within 2 ms.

Every window evaluated with an active reference is reported, with or without a valid peak: a
reference that plays while no echo of it shows up tells the caller the line has none.

All positions count samples of the estimator's own stream (the frames it was fed, gaps
removed); lags are in 16 kHz samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mlx_realtime.audio.resample import Resampler

RATE = 8_000
SOURCE_RATE = 16_000
WINDOW = 2 * RATE  # 2 s
HOP = RATE // 2  # 0.5 s
LAG_MIN = -3 * RATE // 10  # -300 ms
LAG_MAX = RATE  # +1000 ms
AGREE = 16  # 2 ms
ACTIVE_FRACTION = 0.5
ACTIVE_FRAME = RATE // 100  # 10 ms
ACTIVE_POWER = 10 ** (-56.2 / 10)  # mean square of a -56 dBov frame (full scale = 1)
PSR_MIN = 8.0
PEAK_HALF_WIDTH = 16
NFFT = 1 << 15  # >= the reference span (3.3 s): the lags read never wrap
_BAND = (200.0, 3600.0)
_SPAN = WINDOW + LAG_MAX - LAG_MIN  # reference samples per evaluation
_KEEP = _SPAN + HOP  # history kept


@dataclass(frozen=True, slots=True)
class DelayEstimate:
    """One window evaluated while the reference was active."""

    end: int  # where the capture window ended, in 16 kHz samples of the stream
    valid: bool  # a distinct correlation peak: echo of the reference is there
    lag: int = 0  # the echo trails the reference by this many 16 kHz samples (if valid)
    psr: float = 0.0
    confirmed: bool = False  # valid, and the previous valid window agreed within 2 ms


class DelayEstimator:
    """One stream's GCC-PHAT residual-delay tracker. Not thread-safe; frames in order."""

    def __init__(self) -> None:
        self._capture_resampler = Resampler(SOURCE_RATE, RATE)
        self._reference_resampler = Resampler(SOURCE_RATE, RATE)
        self._capture = _History()
        self._reference = _History()
        self._next_end = WINDOW
        self._previous: int | None = None
        frequencies = np.fft.rfftfreq(NFFT, 1 / RATE)
        self._band = (frequencies >= _BAND[0]) & (frequencies <= _BAND[1])
        self.windows = 0  # windows evaluated
        self.valid = 0  # windows that gave an estimate

    def restart(self) -> None:
        """The reference was moved: forget the last estimate and skip windows that mix the old
        and the new alignment."""
        self._previous = None
        self._next_end = max(self._next_end, self._capture.end + WINDOW)

    def push(
        self, capture: npt.NDArray[np.int16], reference: npt.NDArray[np.int16]
    ) -> list[DelayEstimate]:
        """Feed equal lengths of the canceller's input and reference; returns new estimates."""
        if capture.shape != reference.shape:
            raise ValueError("capture and reference must be the same length")
        self._capture.extend(_to_float(self._capture_resampler.process(capture.tobytes())))
        self._reference.extend(_to_float(self._reference_resampler.process(reference.tobytes())))
        estimates: list[DelayEstimate] = []
        ready = min(self._capture.end, self._reference.end)
        while self._next_end - LAG_MIN <= ready:
            estimate = self._evaluate(self._next_end)
            self._next_end += HOP
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def _evaluate(self, end: int) -> DelayEstimate | None:
        self.windows += 1
        start = end - WINDOW
        reference = self._reference.span(start - LAG_MAX, end - LAG_MIN)
        active = reference[LAG_MAX : LAG_MAX + WINDOW]
        frames = active[: WINDOW // ACTIVE_FRAME * ACTIVE_FRAME].reshape(-1, ACTIVE_FRAME)
        if np.mean(np.mean(frames * frames, axis=1) > ACTIVE_POWER) < ACTIVE_FRACTION:
            return None
        capture = self._capture.span(start, end)
        spectrum = np.fft.rfft(reference, NFFT) * np.conj(np.fft.rfft(capture, NFFT))
        weighted = np.where(self._band, spectrum / (np.abs(spectrum) + 1e-12), 0.0)
        magnitude = np.abs(np.fft.irfft(weighted, NFFT)[: _SPAN - WINDOW + 1])
        peak = int(np.argmax(magnitude))
        sides = np.concatenate(
            (magnitude[: max(peak - PEAK_HALF_WIDTH, 0)], magnitude[peak + PEAK_HALF_WIDTH + 1 :])
        )
        psr = float((magnitude[peak] - sides.mean()) / (sides.std() + 1e-12))
        if psr < PSR_MIN:
            return DelayEstimate(end * 2, valid=False, psr=psr)
        self.valid += 1
        lag = LAG_MAX - peak  # 8 kHz samples by which the echo trails the reference
        confirmed = self._previous is not None and abs(lag - self._previous) <= AGREE
        self._previous = lag
        return DelayEstimate(end * 2, valid=True, lag=lag * 2, psr=psr, confirmed=confirmed)


class _History:
    """The most recent samples of a stream, addressed by absolute position."""

    def __init__(self) -> None:
        self._buffer = np.zeros(2 * _KEEP, dtype=np.float64)
        self._length = 0
        self._base = 0  # absolute position of _buffer[0]

    @property
    def end(self) -> int:
        return self._base + self._length

    def extend(self, samples: npt.NDArray[np.float64]) -> None:
        count, capacity = len(samples), len(self._buffer)
        if count >= capacity:
            self._base = self.end + count - capacity
            self._buffer[:] = samples[count - capacity :]
            self._length = capacity
            return
        if self._length + count > capacity:  # compact: keep the newest history
            keep = min(self._length, _KEEP, capacity - count)
            self._buffer[:keep] = self._buffer[self._length - keep : self._length]
            self._base += self._length - keep
            self._length = keep
        self._buffer[self._length : self._length + count] = samples
        self._length += count

    def span(self, first: int, last: int) -> npt.NDArray[np.float64]:
        """Samples [first, last); positions before the history or the stream start are 0."""
        out = np.zeros(last - first, dtype=np.float64)
        low, high = max(first, self._base), min(last, self.end)
        if high > low:
            out[low - first : high - first] = self._buffer[low - self._base : high - self._base]
        return out


def _to_float(pcm: bytes) -> npt.NDArray[np.float64]:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0

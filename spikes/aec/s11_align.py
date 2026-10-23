"""S11 part 2: reference alignment for the canceller. Throwaway.

Methods (A3 sections 3 and 5 "Timeline"):
  engine      the reference on the modelled timeline (60 ms lead), unshifted: the engine absorbs the
              residual delay R = D + 60 ms + (true - modelled play) - eps by itself.
  locked      the same on the "locked" model (AICC's tick phase and mean transit known; s11_chain).
  gcc         a gateway-side GCC-PHAT delay estimator (A3 design point 7) shifts the modelled reference:
                - both signals low-passed at 3.8 kHz (design point 8) and decimated to 8 kHz;
                - a 2 s capture window every 0.5 s of capture time, lags -300 ... +1000 ms, the
                  reference read up to 300 ms ahead of the capture (the gateway holds sent audio);
                - PHAT weighting over 200-3600 Hz;
                - run only while the reference is active (>= 50 % of the window's 10 ms frames above
                  -56 dBov, 30 dB under the far-end level) and the peak is distinct (PSR >= 8);
                - an estimate is confirmed when two consecutive valid windows agree within 2 ms.
              DEVIATION (forced by the data): a confirmed estimate re-shifts the reference only when the
              residual delay it implies leaves a keep range. Re-shifting on every confirmed estimate
              moved the reference by +-1 ms every second and kept the linear filters from converging.
                - linear engines (Speex, baseline): target 20 ms, keep [0, 44] ms, so a 20 ms jump from
                  the tick-phase ambiguity stays inside a 64 ms filter instead of forcing two re-shifts;
                - AEC3: a coarse guard only: target 60 ms, keep [5, 400] ms (AEC3 tracks the rest itself
                  and its estimator ends near 509 ms).
  gcc_locked  GCC-PHAT on the locked model.
  oracle      the reference placed from the true playout (s11_chain ref_oracle), 20 ms before the echo.
"""

from __future__ import annotations

import s11_common as c

import numpy as np  # noqa: E402

from s11_chain import MARGIN_S  # noqa: E402

GCC_WIN_S = 2.0
GCC_HOP_S = 0.5
GCC_LAG_MIN_S, GCC_LAG_MAX_S = -0.300, 1.000
GCC_AGREE_S = 0.002
GCC_ACTIVE_FRAC = 0.5
GCC_PSR_MIN = 8.0
GCC_NFFT = 32768  # >= reference window (3.3 s at 8 kHz = 26400): the lags we read never wrap


KEEP_LINEAR = (MARGIN_S, 0.000, 0.044)  # (target, keep_lo, keep_hi), seconds of residual delay
KEEP_AEC3 = (0.060, 0.005, 0.400)


class GccPhat:
    """Causal (streaming-equivalent) GCC-PHAT shift tracker over a whole call."""

    def __init__(self, keep: tuple[float, float, float] = KEEP_LINEAR):
        self.target, self.keep_lo, self.keep_hi = keep
        self.sr = c.SR_TEL
        self.win = int(GCC_WIN_S * self.sr)
        self.hop16 = int(GCC_HOP_S * c.SR)
        self.back = int(GCC_LAG_MAX_S * self.sr)  # reference history needed before the window
        self.ahead = int(-GCC_LAG_MIN_S * self.sr)  # reference look-ahead
        f = np.fft.rfftfreq(GCC_NFFT, 1 / self.sr)
        self.band = (f >= 200) & (f <= 3600)
        self.shift16 = 0  # applied shift, 16 kHz samples (positive = reference delayed)
        self.prev = None
        self.log: list[dict] = []
        self.cpu_ns = 0

    @staticmethod
    def to8k(x16: np.ndarray) -> np.ndarray:
        y = np.convolve(np.asarray(x16, dtype=np.float64), c.PREFILTER_TAPS)[: len(x16)]
        return y[0::2]

    def track(self, cap16: np.ndarray, ref16: np.ndarray) -> np.ndarray:
        """Shift (16 kHz samples) to apply to the reference for every 10 ms frame of the call.
        Window k ends at capture time (k+1)*hop and is decided before the next frame is processed."""
        import time

        t0 = time.thread_time_ns()
        cap8, ref8 = self.to8k(cap16), self.to8k(ref16)
        n_frames = len(cap16) // c.FRAME
        shifts = np.zeros(n_frames, dtype=np.int64)
        frame_hop = self.hop16 // c.FRAME
        rpow = np.convolve((ref8 / 32768.0) ** 2, np.ones(80) / 80, mode="valid")[::80]  # 10 ms frames
        thr = 10 ** ((c.dbm0_to_dbov(-20.0) - 30) / 10)
        cur = 0
        for end_frame in range(frame_hop, n_frames + 1, frame_hop):
            shifts[cur:end_frame] = self.shift16
            cur = end_frame
            e8 = end_frame * 80  # capture end, 8 kHz samples
            if e8 < self.win:
                continue
            a8 = e8 - self.win
            fa, fb = a8 // 80, e8 // 80
            act = float(np.mean(rpow[fa:fb] > thr)) if fb <= len(rpow) else 0.0
            rec = {"t": round(e8 / self.sr, 3), "ref_active": round(act, 3)}
            if act < GCC_ACTIVE_FRAC:
                rec["valid"] = False
                self.log.append(rec)
                continue
            y = cap8[a8:e8]
            xa, xb = a8 - self.back, e8 + self.ahead
            x = np.zeros(xb - xa)
            sa, sb = max(xa, 0), min(xb, len(ref8))
            x[sa - xa: sb - xa] = ref8[sa:sb]
            X = np.fft.rfft(x, GCC_NFFT)
            Y = np.fft.rfft(y, GCC_NFFT)
            C = X * np.conj(Y)
            W = np.where(self.band, 1.0 / (np.abs(C) + 1e-12), 0.0)
            r = np.fft.irfft(C * W, GCC_NFFT)[: len(x) - len(y) + 1]
            k = int(np.argmax(np.abs(r)))
            mag = np.abs(r)
            side = np.concatenate([mag[: max(k - 16, 0)], mag[k + 17:]])
            psr = float((mag[k] - side.mean()) / (side.std() + 1e-12))
            lag_s = (self.back - k) / self.sr  # echo = reference delayed by lag_s
            rec.update({"lag_ms": round(lag_s * 1000, 3), "psr": round(psr, 2)})
            if psr < GCC_PSR_MIN:
                rec["valid"] = False
                self.log.append(rec)
                continue
            rec["valid"] = True
            if self.prev is not None and abs(lag_s - self.prev) <= GCC_AGREE_S:
                residual = lag_s - self.shift16 / c.SR
                rec["residual_ms"] = round(residual * 1000, 3)
                if not (self.keep_lo <= residual <= self.keep_hi):
                    self.shift16 = int(round((lag_s - self.target) * c.SR))
                    rec["applied_shift_ms"] = round(self.shift16 / c.SR * 1000, 3)
            self.prev = lag_s
            self.log.append(rec)
        shifts[cur:] = self.shift16
        self.cpu_ns = time.thread_time_ns() - t0
        return shifts


def reference_frames(ref16: np.ndarray, shifts: np.ndarray | None, n_frames: int) -> np.ndarray:
    """[n_frames, 160] int16 reference frames, ref[n - shift] (zeros outside the signal)."""
    out = np.zeros((n_frames, c.FRAME), dtype=np.int16)
    if shifts is None:
        m = min(n_frames, len(ref16) // c.FRAME)
        out[:m] = ref16[: m * c.FRAME].reshape(m, c.FRAME)
        return out
    pad = 1 << 16
    padded = np.concatenate([np.zeros(pad, np.int16), ref16, np.zeros(pad, np.int16)])
    for j in range(n_frames):
        a = j * c.FRAME - int(shifts[j]) + pad
        a = min(max(a, 0), len(padded) - c.FRAME)
        out[j] = padded[a:a + c.FRAME]
    return out

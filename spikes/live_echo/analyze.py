"""Offline analysis of the gateway's echo captures (``aec.capture_dir``) from real AICC calls.

    uv run --locked python analyze.py <capture dir>... [--aicc-recording <stereo wav> ...]

Per capture directory it writes ``report.md`` and ``report.json`` beside the audio. Everything
here is independent of the gateway's own estimators: the bulk delay is a plain cross-correlation,
the echo path a least-squares filter, the near-end speech a residual test. All signals are
band-limited to 300-3400 Hz (the telephone band: AICC's uplink is G.711 at 8 kHz) and handled at
8 kHz; times are input-timeline ms.

Method in short (details in the README):

1. Reference-active frames (20 ms, ``ref_aec3`` above -50 dBFS, the gateway's own threshold).
2. Bulk delay: 2 s windows every 1 s, the reference against the raw input, both pre-emphasized,
   plain cross-correlation over -100 ... +700 ms. A window is valid when its peak stands
   ``Z_VALID`` standard deviations above the correlation's other values. Lags count how far the
   echo trails the reference *file*; adding the gateway's shift and subtracting its 60 ms lead
   gives the delay after the playout model ("vs model").
3. Echo path: least squares (ridge) over echo-only frames, 120 ms of taps from 20 ms before the
   bulk delay. ERL = reference energy over predicted-echo energy; dispersion = the shortest span
   holding 90 % of the filter's energy (noise floor removed); ERL over time in 5 s blocks.
4. Near end: a frame is caller speech when what the filter does not explain (input - predicted
   echo) is ``NEAR_DB`` over the line's noise floor; 150 ms runs are speech, the onset is the
   run's first frame, and the ASR track (Speex's output) must carry the speech too. The AEC3
   and ASR tracks are never used to judge echo-only stretches (their residual is what is tested).
5. ERLE per 0.5 s windows of echo-only frames whose predicted echo is ``ECHO_OVER_NOISE_DB``
   above the noise floor: input energy over output energy.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

RATE = 8_000  # analysis rate
FRAME = RATE // 50  # 20 ms
SOURCE_RATE = 16_000
BAND = (300.0, 3400.0)
ACTIVE_DBFS = -50.0
WINDOW_S, HOP_S = 2.0, 1.0
LAG_RANGE_MS = (-100.0, 700.0)
Z_VALID = 8.0
CONSENSUS_MS = 30.0  # valid windows must agree on a delay within this much of the most common one
EXPECT_FRACTION = 0.5  # classes: valid windows / windows with an active reference
WEAK_FRACTION = 0.1
WEAK_ERL_DB = 25.0
NEAR_DB = 12.0
NEAR_RUN_FRAMES = 15
NEAR_SHARE_DB = -9.0  # and what the filter does not explain is at least this share of the input
ECHO_OVER_NOISE_DB = 6.0
ERLE_WINDOW_FRAMES = 25
ERLE_TARGET_DB = 10.0
DRIFT_FLAG_MS = 3.0
TAPS_BEFORE_MS, TAPS_MS = 20.0, 120.0
GCC_KEEP_MS = (5.0, 450.0)  # the configured keep range, overridden by meta.json
SPEEX_KEEP_MS = (0.0, 44.0)
SPEEX_FILTER_MS = 64.0
BARGE_IN_BUDGET_S = 1.5
AICC_CLEAR_MS = 800.0  # AICC's barge guard: our audio keeps playing this long after the flush
TAIL_ECHO_MS = 300.0  # and its echo this long after that
STALL_SHIFT_MS = 15.0  # a bot-channel shift against the model that is not a model error
STALL_ZERO_MS = 100.0
SELF_INTERRUPT_LOOKBACK_S = 3.0
BURST_WINDOW_S, BURST_HOP_S = 0.5, 0.25
BURST_LAGS_MS = (50.0, 700.0)
BURST_SHUFFLES = 12
BURST_BASELINE_Z = 6.0  # a window's peak correlation over the shuffled baseline, in its std
BURST_MIN_NCC = 0.1
BURST_OVER_FLOOR_DB = 6.0
BURST_MERGE_GAP_S = 0.5
BURST_START_S, BURST_TAIL_S = 2.5, 1.5  # position classes, around a response's start and end


# ------------------------------------------------------------------------------------- signals


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """int16 PCM as float64, shape (samples, channels)."""
    with wave.open(str(path), "rb") as file:
        if file.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit PCM WAV is supported")
        raw = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
        return raw.reshape(-1, file.getnchannels()).astype(np.float64), file.getframerate()


def fft_convolve(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Full convolution through the FFT."""
    n = len(x) + len(taps) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(taps, size), size)[:n]


def lowpass_taps(cutoff_hz: float, rate: int, length: int = 255) -> np.ndarray:
    index = np.arange(length) - (length - 1) / 2
    taps = 2 * cutoff_hz / rate * np.sinc(2 * cutoff_hz / rate * index) * np.blackman(length)
    return taps / taps.sum()


def to_analysis(x: np.ndarray, rate: int) -> np.ndarray:
    """Band-limited 300-3400 Hz at 8 kHz, no delay (linear-phase FIR, compensated)."""
    length = 255
    band = lowpass_taps(BAND[1], rate, length) - lowpass_taps(BAND[0], rate, length)
    filtered = fft_convolve(x, band)[(length - 1) // 2 : (length - 1) // 2 + len(x)]
    if rate == RATE:
        return filtered
    if rate % RATE == 0:
        return filtered[:: rate // RATE]
    seconds = len(filtered) / rate
    grid = np.arange(int(seconds * RATE)) / RATE
    return np.interp(grid, np.arange(len(filtered)) / rate, filtered)


def frames_of(x: np.ndarray) -> np.ndarray:
    """Mean square per 20 ms frame."""
    count = len(x) // FRAME
    return np.mean(x[: count * FRAME].reshape(count, FRAME) ** 2, axis=1)


def db(ratio: float) -> float:
    return float(10 * np.log10(max(ratio, 1e-12)))


def active_frames(ref: np.ndarray) -> np.ndarray:
    return frames_of(ref) > 10 ** (ACTIVE_DBFS / 10) * 32768.0**2


def delayed(x: np.ndarray, lag: int) -> np.ndarray:
    """x delayed by ``lag`` samples (negative: advanced), same length."""
    out = np.zeros_like(x)
    if lag >= 0:
        out[lag:] = x[: len(x) - lag] if lag < len(x) else 0
    else:
        out[: len(x) + lag] = x[-lag:]
    return out


def consensus(windows: list[dict]) -> list[dict]:
    """A chance peak sits anywhere in the 800 ms searched; echo repeats at about one delay. Windows
    whose lag is not within ``CONSENSUS_MS`` of the most common lag (in 20 ms bins) lose their
    ``valid``; fewer than two agreeing windows leave none."""
    lags = np.array([w["lag_ms"] for w in windows if w["valid"]])
    if len(lags) == 0:
        return windows
    bins = np.round(lags / 20.0)
    centre = float(np.median(lags[bins == np.bincount((bins - bins.min()).astype(int)).argmax() + bins.min()]))
    agreeing = int(np.sum(np.abs(lags - centre) <= CONSENSUS_MS))
    for w in windows:
        w["valid"] = bool(w["valid"] and abs(w["lag_ms"] - centre) <= CONSENSUS_MS and agreeing >= 2)
    return windows


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    kernel = np.ones(2 * frames + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


# ------------------------------------------------------------------------------- bulk delay


def preemphasis(x: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], x[1:] - 0.9 * x[:-1]))


def window_delay(ref: np.ndarray, inp: np.ndarray, start: int, count: int) -> dict | None:
    """Cross-correlation of the input window [start, start+count) with the reference around it."""
    low, high = (round(ms * RATE / 1000) for ms in LAG_RANGE_MS)
    first = start - high
    segment = np.zeros(count + high - low)
    lo, hi = max(first, 0), min(first + len(segment), len(ref))
    if hi <= lo:
        return None
    segment[lo - first : hi - first] = ref[lo:hi]
    window = inp[start : start + count]
    if len(window) < count or not window.any() or not segment.any():
        return None
    size = 1 << (len(segment) + count - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(segment, size) * np.conj(np.fft.rfft(window, size)), size)
    cc = cc[: high - low + 1] / (np.linalg.norm(window) * np.linalg.norm(segment) + 1e-9)
    magnitude = np.abs(cc)
    peak = int(np.argmax(magnitude))
    rest = np.delete(magnitude, np.arange(max(peak - 8, 0), min(peak + 9, len(magnitude))))
    z = float((magnitude[peak] - rest.mean()) / (rest.std() + 1e-12))
    offset = 0.0
    if 0 < peak < len(cc) - 1:  # parabolic interpolation of the peak
        a, b, c = magnitude[peak - 1], magnitude[peak], magnitude[peak + 1]
        denominator = a - 2 * b + c
        offset = 0.0 if denominator == 0 else 0.5 * (a - c) / denominator
    lag = high - (peak + offset)  # samples by which the echo trails the reference
    return {"lag_ms": lag * 1000 / RATE, "z": z, "valid": z >= Z_VALID}


def delay_windows(
    ref: np.ndarray, inp: np.ndarray, usable: np.ndarray | None = None
) -> list[dict]:
    """Bulk-delay windows over the whole signal. ``usable``: frames that may enter a window
    (echo-only); samples of other frames are zeroed in the input, and a window needs 60 % of
    its frames to be usable and reference-active."""
    pre_ref, pre_in = preemphasis(ref), preemphasis(inp)
    active = active_frames(ref)
    good = active if usable is None else active & usable[: len(active)]
    count, step = round(WINDOW_S * RATE), round(HOP_S * RATE)
    per = count // FRAME
    out: list[dict] = []
    for start in range(0, len(inp) - count + 1, step):
        f0 = start // FRAME
        span = slice(f0, f0 + per)
        if np.mean(active[span]) < 0.6 or (usable is not None and np.mean(good[span]) < 0.5):
            continue
        masked = pre_in[start : start + count].copy()
        if usable is not None:
            keep = np.repeat(usable[span], FRAME)
            masked[: len(keep)] *= keep
        shifted = np.zeros(len(pre_in))
        shifted[start : start + count] = masked
        result = window_delay(pre_ref, shifted, start, count)
        if result is not None:
            out.append({"start_ms": start * 1000 / RATE, "end_ms": (start + count) * 1000 / RATE,
                        **result})
    return out


# ----------------------------------------------------------------------------- echo path (LS)


BLOCK = 8192  # 1 s Welch blocks, half overlapped, for the echo-path estimate


def fit_path(ref: np.ndarray, inp: np.ndarray, echo: np.ndarray, bulk_ms: float, clean: np.ndarray | None = None) -> dict:
    """The echo path from the echo-only frames: H = S_yx / (S_xx + eps) over Hann-windowed
    blocks of the input and the reference delayed by ``k0`` (the bulk delay less 20 ms), so the
    path (a few tens of ms) sits well inside a block. Out-of-band bins carry no energy and get
    no gain. ``h[j]`` is the path at lag ``k0 + j``. A block is used when at least half of its
    frames have an echo (``echo``: frames whose reference played ``bulk_ms`` earlier) and all
    of them are ``clean`` (no caller speech)."""
    k0 = max(round((bulk_ms - TAPS_BEFORE_MS) * RATE / 1000), 0)
    taps = round(TAPS_MS * RATE / 1000)
    shifted = delayed(ref, k0)
    window = np.hanning(BLOCK)
    frames_per = BLOCK // FRAME + 1
    found = []
    for first in range(0, len(echo) - frames_per, BLOCK // FRAME // 2):
        if echo[first : first + frames_per].mean() < 0.5:
            continue
        if clean is not None and not clean[first : first + frames_per].all():
            continue
        a = first * FRAME
        if a + BLOCK > len(inp):
            continue
        spectrum = np.fft.rfft(shifted[a : a + BLOCK] * window)
        found.append((first, spectrum, np.fft.rfft(inp[a : a + BLOCK] * window)))
    if not found:
        return {"ok": False}

    def solve(blocks: list) -> np.ndarray:
        sxx = sum(np.abs(x) ** 2 for _, x, _ in blocks)
        sxy = sum(y * np.conj(x) for _, x, y in blocks)
        return sxy / (sxx + 1e-3 * sxx.mean() + 1e-9)

    # A block the filter does not explain (the delay moved: a tick phase learned, a re-anchor;
    # or leftover caller speech) is dropped and the filter fitted again on the others.
    transfer = solve(found)
    gains = np.array([
        10 * np.log10(np.sum(np.abs(y) ** 2) / max(np.sum(np.abs(y - transfer * x) ** 2), 1e-9))
        for _, x, y in found
    ])
    kept = [b for b, g in zip(found, gains, strict=True) if g >= np.median(gains) - 6.0]
    transfer = solve(kept)
    good = np.zeros(len(echo), dtype=bool)
    for first, _, _ in kept:
        good[first : first + frames_per] = True
    h = np.fft.irfft(transfer, BLOCK)[:taps]
    return {"ok": True, "h": h, "k0": k0, "taps": taps, "blocks": len(kept), "dropped": len(found) - len(kept), "frames": good}


def local_predict(ref: np.ndarray, inp: np.ndarray, echo: np.ndarray, bulk_ms: float) -> np.ndarray:
    """A prediction of the echo from a filter fitted to each 1 s block alone (cut to the path's
    120 ms), for finding the caller where the pooled filter is off (the delay moved by a fraction
    of a millisecond inside an answer). It cannot tell echo from caller speech by its fit, but
    120 ms of taps on a 1 s block explain about 12 % of any uncorrelated speech."""
    k0 = max(round((bulk_ms - TAPS_BEFORE_MS) * RATE / 1000), 0)
    taps = round(TAPS_MS * RATE / 1000)
    shifted = delayed(ref, k0)
    window = np.hanning(BLOCK + 1)[:BLOCK]  # periodic: half-overlapped blocks sum to one
    frames_per = BLOCK // FRAME + 1
    out = np.zeros(len(inp))
    for first in range(0, len(echo) - frames_per, BLOCK // FRAME // 2):
        a = first * FRAME
        if echo[first : first + frames_per].mean() < 0.5 or a + BLOCK > len(inp):
            continue
        x = np.fft.rfft(shifted[a : a + BLOCK] * window)
        y = np.fft.rfft(inp[a : a + BLOCK] * window)
        h = np.fft.irfft(y * np.conj(x) / (np.abs(x) ** 2 + 1e-3 * np.mean(np.abs(x) ** 2) + 1e-9), BLOCK)[:taps]
        block = fft_convolve(shifted[a : a + BLOCK] * window, h)[:BLOCK]
        out[a : a + BLOCK] += block
    return out


def predict(ref: np.ndarray, fit: dict) -> np.ndarray:
    if not fit["ok"]:
        return np.zeros_like(ref)
    full = fft_convolve(ref, fit["h"])
    return delayed(full[: len(ref)], fit["k0"])


def dispersion_ms(h: np.ndarray) -> float:
    energy = h**2
    floor = float(np.median(energy[-len(energy) // 4 :]))
    energy = np.clip(energy - floor, 0.0, None)
    total = energy.sum()
    if total <= 0:
        return float("nan")
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    ends = np.searchsorted(cumulative, cumulative[:-1] + 0.9 * total)
    spans = ends[ends <= len(energy)] - np.arange(len(ends))[ends <= len(energy)]
    return float(spans.min() * 1000 / RATE) if len(spans) else float("nan")


# ----------------------------------------------------------------------------------- one line


def noise_floor(inp: np.ndarray, quiet: np.ndarray) -> float:
    energy = frames_of(inp)
    pool = energy[quiet[: len(energy)]] if quiet.any() else energy
    return float(np.median(pool[pool <= np.median(pool)])) if len(pool) else 1.0


def near_end(inp: np.ndarray, pred: np.ndarray, floor: float) -> np.ndarray:
    """Frames of caller speech: the input less the predicted echo, over the noise floor, in
    runs of at least ``NEAR_RUN_FRAMES``."""
    residual = frames_of(inp - pred)
    loud = (residual > floor * 10 ** (NEAR_DB / 10)) & (
        residual > frames_of(inp) * 10 ** (NEAR_SHARE_DB / 10)
    )
    votes = np.convolve(loud.astype(float), np.ones(5), mode="same") >= 3
    out = np.zeros_like(votes)
    padded = np.concatenate(([False], votes, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        if last - first >= NEAR_RUN_FRAMES:
            out[first:last] = True
    return out


def drop_tail_runs(caller: np.ndarray, tail: np.ndarray | None) -> np.ndarray:
    """Caller-speech runs that begin inside an AICC still-playing window are leftover echo."""
    if tail is None:
        return caller
    out = caller.copy()
    padded = np.concatenate(([False], caller, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        if first < len(tail) and tail[first]:
            out[first:last] = False
    return out


def spans_frames(spans: list[tuple[float, float]] | None, n_frames: int) -> list[tuple[int, int]]:
    """Frame ranges of the reference's answers (their echo follows within 700 ms)."""
    if not spans:
        return [(0, n_frames)]
    return [(max(int(a // 20), 0), min(int(b // 20) + 35, n_frames)) for a, b in spans]


def model_line(
    ref: np.ndarray,
    inp: np.ndarray,
    tail: np.ndarray | None = None,
    spans: list[tuple[float, float]] | None = None,
) -> dict:
    """The line between a reference and the input it echoes into (see the module notes).

    ``spans``: each answer's (start, end) on the reference's timeline. The echo path is fitted
    per answer, at that answer's own delay: the playout model places every run of audio at its
    own 20 ms tick phase, so the delay differs by fractions of a millisecond from run to run,
    enough to decorrelate the upper band if the runs were pooled."""
    n = min(len(ref), len(inp))
    ref, inp = ref[:n], inp[:n]
    active = active_frames(ref)
    n_frames = len(frames_of(inp))
    segs = spans_frames(spans, n_frames)
    result: dict = {"ref_active_frames": int(active.sum()), "frames": len(active)}
    # Stage 0: the bulk delay from every reference-active window.
    windows = consensus(delay_windows(ref, inp))
    if not windows:
        return {**result, "klass": "none", "reason": "the reference was never active"}
    valid = [w for w in windows if w["valid"]]
    bulk = float(np.median([w["lag_ms"] for w in valid])) if valid else 0.0

    def fit_segments(windows: list[dict], usable: np.ndarray | None) -> dict:
        """One filter per answer at its own median delay; the prediction over the whole input."""
        pred, ref_aligned, local = np.zeros(n), np.zeros(n), np.zeros(n)
        echo = np.zeros(n_frames, dtype=bool)
        good = np.zeros(n_frames, dtype=bool)
        fits = []
        for index, (a, b) in enumerate(segs):
            own = [w["lag_ms"] for w in windows if w["valid"] and a * 20 <= (w["start_ms"] + w["end_ms"]) / 2 <= b * 20 + 700]
            lag_ms = float(np.median(own)) if own else bulk
            lag = round(lag_ms * RATE / 1000)
            part = np.zeros(n)
            part[a * FRAME : b * FRAME] = ref[a * FRAME : b * FRAME]
            mask = np.zeros(n_frames, dtype=bool)
            lo, hi = a + lag // FRAME, min(b + lag // FRAME, n_frames)
            mask[lo:hi] = active[a : a + hi - lo]
            mask = dilate(mask, 2)
            echo |= mask
            ref_aligned += delayed(part, lag)
            local += local_predict(part, inp, mask, lag_ms)
            if not valid:
                continue
            fit = fit_path(part, inp, mask, lag_ms, usable)
            if fit["ok"]:
                pred += predict(part, fit)
                good |= fit["frames"][:n_frames]
                fits.append({**fit, "lag_ms": lag_ms, "segment": index})
        return {"pred": pred, "ref_aligned": ref_aligned, "echo": echo, "fits": fits, "good": good, "local": local}

    # Stage 1: a first filter per answer, to find the caller.
    first = fit_segments(windows, None)
    echo_frames = first["echo"]
    floor = noise_floor(inp, ~dilate(echo_frames, 25))
    caller = drop_tail_runs(near_end(inp, first["local"], floor), tail)
    # Stage 2: the delays and the filters again over frames without the caller.
    usable = ~dilate(caller, 10)
    windows = consensus(delay_windows(ref, inp, usable))
    valid = [w for w in windows if w["valid"]]
    result["windows"] = windows
    fraction = len(valid) / max(len(windows), 1)
    result["valid_fraction"] = fraction
    if valid:
        bulk = float(np.median([w["lag_ms"] for w in valid]))
    second = fit_segments(windows, usable)
    pred = second["pred"]
    caller = drop_tail_runs(near_end(inp, second["local"], floor), tail)
    echo_only = second["echo"] & ~dilate(caller, 10)
    result.update(bulk_ms=bulk, noise_floor_db=db(floor / 32768.0**2), caller=caller,
                  echo_only=echo_only, floor=floor, pred=pred)
    fits = second["fits"]
    if fits:
        e_in = frames_of(inp)[:n_frames]
        e_pred = frames_of(pred)[:n_frames]
        e_ref = frames_of(second["ref_aligned"])[:n_frames]
        residual = frames_of(inp - pred)[:n_frames]
        sel = echo_only & second["good"] & (e_pred > 0)  # frames the fitted blocks cover
        result.update(
            filter_peak_ms=float(np.median([(f["k0"] + int(np.argmax(np.abs(f["h"])))) * 1000 / RATE for f in fits])),
            erl_db=db(e_ref[sel].sum() / max((e_in[sel] - floor).clip(0).sum(), 1e-9)) if sel.any() else None,
            erl_ls_db=db(e_ref[sel].sum() / max(e_pred[sel].sum(), 1e-9)) if sel.any() else None,
            dispersion_ms=float(np.nanmedian([dispersion_ms(f["h"]) for f in fits])),
            fit_gain_db=db(e_in[sel].sum() / max(residual[sel].sum(), 1e-9)) if sel.any() else None,
            echo_over_noise_db=db(e_pred[sel].mean() / floor) if sel.any() else None,
            h=fits[0]["h"],
            k0=fits[0]["k0"],
            segment_lags_ms=[f["lag_ms"] for f in fits],
            segment_peaks_ms={f["segment"]: (f["k0"] + int(np.argmax(np.abs(f["h"])))) * 1000 / RATE for f in fits},
        )
        blocks = []
        for start in range(0, n_frames, 250):  # 5 s of frames
            block = sel[start : start + 250]
            if block.sum() >= 50:
                part = slice(start, start + 250)
                blocks.append({"t_ms": start * 20.0, "erl_db": db(e_ref[part][block].sum() / max((e_in[part][block] - floor).clip(0).sum(), 1e-9))})
        result["erl_over_time"] = blocks
    erl = result.get("erl_db")
    if fraction < WEAK_FRACTION or not valid:
        result["klass"] = "none"
    elif fraction < EXPECT_FRACTION or (erl is not None and erl >= WEAK_ERL_DB):
        result["klass"] = "weak"
    else:
        result["klass"] = "echo"
    return result


def erle_series(inp: np.ndarray, out: np.ndarray, line: dict) -> list[dict]:
    """ERLE per 0.5 s window of echo-only frames with echo clearly above the noise."""
    echo_only, pred, floor = line["echo_only"], line["pred"], line["floor"]
    n = min(len(echo_only), len(frames_of(out)), len(frames_of(inp)))
    e_in, e_out = frames_of(inp)[:n], frames_of(out)[:n]
    e_pred = frames_of(pred)[:n]
    values = []
    for first in range(0, n - ERLE_WINDOW_FRAMES + 1, ERLE_WINDOW_FRAMES):
        part = slice(first, first + ERLE_WINDOW_FRAMES)
        if echo_only[part].mean() < 0.8 or e_pred[part].mean() < floor * 10 ** (ECHO_OVER_NOISE_DB / 10):
            continue
        values.append({"t_ms": first * 20.0, "erle_db": db(e_in[part].sum() / max(e_out[part].sum(), 1e-9))})
    return values


def convergence(series: list[dict], since_ms: float) -> tuple[float | None, float | None]:
    """Time after ``since_ms`` when ERLE first holds the target for three windows, and the
    median of the windows after it (after 10 s when it never does)."""
    rows = [row for row in series if row["t_ms"] >= since_ms]
    for i in range(len(rows) - 2):
        if all(r["erle_db"] >= ERLE_TARGET_DB for r in rows[i : i + 3]):
            after = [r["erle_db"] for r in rows[i:]]
            return rows[i]["t_ms"] - since_ms, float(np.median(after))
    late = [r["erle_db"] for r in rows if r["t_ms"] >= since_ms + 10_000]
    return None, float(np.median(late)) if late else None


# ------------------------------------------------------------------------------------- events


def load_events(directory: Path) -> list[dict]:
    events = []
    for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("t") is not None:
                events.append(record)
    return sorted(events, key=lambda e: e["t"])


def responses_of(events: list[dict]) -> list[dict]:
    starts = {e["item"]: e for e in events if e["kind"] == "play_start"}
    ends = {e["item"]: e for e in events if e["kind"] == "play_end"}
    out = []
    for item, start in starts.items():
        end = ends.get(item)
        out.append({"item": item, "start_ms": start["t"], "end_ms": end["t"] if end else None})
    return sorted(out, key=lambda r: r["start_ms"])


def shift_at(events: list[dict], t_ms: float) -> float:
    shift = 0.0
    for e in events:
        if e["kind"] == "realign" and e.get("engine") == "aec3" and e["t"] <= t_ms:
            shift = e["shift_ms"]
    return shift


def straddles(events: list[dict], start_ms: float, end_ms: float) -> bool:
    return any(
        e["kind"] == "realign" and e.get("engine") == "aec3" and start_ms < e["t"] < end_ms
        for e in events
    )


# --------------------------------------------------------------------------------- the report


def analyse_capture(directory: Path, recording: Path | None) -> dict:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    events = load_events(directory)
    tracks = {}
    for name in ("input", "ref_aec3", "aec3", "ref_speex", "asr"):
        data, rate = read_wav(directory / f"{name}.wav")
        tracks[name] = to_analysis(data[:, 0], rate)
    aec = meta.get("aec", {})
    barge = meta.get("barge_in", {})
    lead = float(aec.get("reference_lead_ms", 60))
    keep = (float(aec.get("min_residual_delay_ms", GCC_KEEP_MS[0])),
            float(aec.get("max_residual_delay_ms", GCC_KEEP_MS[1])))
    speex = aec.get("speex", {})
    speex_keep = (0.0, float(speex.get("max_delay_ms", SPEEX_KEEP_MS[1])))
    speex_filter = float(speex.get("filter_ms", SPEEX_FILTER_MS))
    speex_target = float(speex.get("target_delay_ms", 20))
    guard_ms = float(barge.get("echo_guard_ms", 2000))
    bypass_after_ms = aec.get("bypass_without_echo_ms", 1500)

    inp, ref = tracks["input"], tracks["ref_aec3"]
    responses = responses_of(events)
    # AICC waits out its 800 ms barge guard before it clears its TX queue: after our flush its
    # client keeps playing (and echoing) our audio for up to that long. Those windows are
    # "AICC still playing": no caller speech starts in them and no speech_started there is a
    # self-interruption; the playout model is not at fault for them.
    flushes = [e["t"] for e in events if e["kind"] == "flush"]
    tail = np.zeros(len(inp) // FRAME, dtype=bool)
    tail_windows = []
    for t in flushes:
        tail[int(t // 20) : int((t + AICC_CLEAR_MS + TAIL_ECHO_MS) // 20) + 1] = True
        tail_windows.append({"flush_ms": t, "until_ms": t + AICC_CLEAR_MS + TAIL_ECHO_MS})
    spans = [(r["start_ms"], r["end_ms"] if r["end_ms"] is not None else r["start_ms"] + 10_000) for r in responses]
    line = model_line(ref, inp, tail, spans or None)
    first_play = responses[0]["start_ms"] if responses else None
    report: dict = {
        "capture": str(directory),
        "session_id": meta.get("session_id"),
        "started_utc": meta.get("started_utc"),
        "duration_s": len(inp) / RATE,
        "class": line["klass"],
        "class_rule": (
            f"none: fewer than {WEAK_FRACTION:.0%} of the reference-active 2 s windows show a "
            f"cross-correlation peak (z >= {Z_VALID}); weak: fewer than {EXPECT_FRACTION:.0%}, "
            f"or ERL >= {WEAK_ERL_DB} dB; echo: otherwise"
        ),
        "aicc_still_playing": tail_windows,
        "echo_summary": meta.get("echo_summary"),
        "capped": meta.get("capped"),
    }
    if "bulk_ms" not in line:
        report["note"] = line.get("reason", "no analysis")
        return report
    windows = line["windows"]
    valid = [w for w in windows if w["valid"]]
    report["valid_window_fraction"] = line["valid_fraction"]
    report["noise_floor_dbfs"] = line["noise_floor_db"]
    # Delay per window, after the playout model: lag + shift - lead.
    rows = []
    for w in windows:
        if not w["valid"] or straddles(events, w["start_ms"] - 700, w["end_ms"] + 100):
            continue
        centre = (w["start_ms"] + w["end_ms"]) / 2
        shift = shift_at(events, centre)
        rows.append({"t_ms": centre, "lag_ms": w["lag_ms"], "shift_ms": shift,
                     "vs_model_ms": w["lag_ms"] + shift - lead, "z": w["z"]})
    report["delay"] = {
        "lag_ms_median": line["bulk_ms"],
        "vs_model_ms_median": float(np.median([r["vs_model_ms"] for r in rows])) if rows else None,
        "windows_valid": len(valid),
        "windows": len(windows),
        "filter_peak_ms": line.get("filter_peak_ms"),
        "per_window": rows,
    }
    # Per response: the first and last third of its valid windows; and across answers.
    per_response = []
    for resp in responses:
        end = resp["end_ms"] if resp["end_ms"] is not None else resp["start_ms"] + 10_000
        lag = line["bulk_ms"]
        own = [r for r in rows if resp["start_ms"] + lag <= r["t_ms"] <= end + lag]
        entry = {"item": resp["item"], "start_ms": resp["start_ms"], "end_ms": resp["end_ms"],
                 "windows": len(own)}
        if len(own) >= 3:
            third = max(len(own) // 3, 1)
            head = float(np.median([r["vs_model_ms"] for r in own[:third]]))
            tail = float(np.median([r["vs_model_ms"] for r in own[-third:]]))
            entry.update(start_vs_model_ms=head, end_vs_model_ms=tail, drift_ms=tail - head,
                         median_vs_model_ms=float(np.median([r["vs_model_ms"] for r in own])))
        per_response.append(entry)
    medians = [r["median_vs_model_ms"] for r in per_response if "median_vs_model_ms" in r]
    drifts = [r["drift_ms"] for r in per_response if "drift_ms" in r]
    report["delay"]["per_response"] = per_response
    report["delay"]["across_answers_spread_ms"] = (max(medians) - min(medians)) if medians else None
    report["delay"]["max_abs_drift_ms"] = max((abs(d) for d in drifts), default=None)
    report["delay"]["drift_flag"] = bool(
        (drifts and max(abs(d) for d in drifts) > DRIFT_FLAG_MS)
        or (len(medians) > 1 and max(medians) - min(medians) > DRIFT_FLAG_MS)
    )
    for key in ("erl_db", "erl_ls_db", "dispersion_ms", "fit_gain_db", "echo_over_noise_db"):
        report[key] = line.get(key)
    report["erl_over_time"] = line.get("erl_over_time", [])
    report["segment_peaks_ms"] = line.get("segment_peaks_ms", {})
    # ERLE of both cancellers.
    erle = {}
    if "h" in line:
        for name in ("aec3", "asr"):
            series = erle_series(inp, tracks[name], line)
            conv, steady = convergence(series, first_play if first_play is not None else 0.0)
            erle[name] = {"windows": len(series), "convergence_s": None if conv is None else conv / 1000,
                          "steady_median_db": steady,
                          "median_db": float(np.median([r["erle_db"] for r in series])) if series else None,
                          "series": series}
    report["erle"] = erle
    # The gateway's own estimator against the independent delay.
    gcc = [e for e in events if e["kind"] == "gcc"]
    agree = []
    for e in gcc:
        if not e["valid"]:
            continue
        near = [w for w in windows if w["valid"] and abs(w["end_ms"] - e["t"]) <= 500]
        if near:
            best = min(near, key=lambda w: abs(w["end_ms"] - e["t"]))
            agree.append(e["lag_ms"] - best["lag_ms"])
    report["gateway_estimator"] = {
        "windows": len(gcc),
        "valid": sum(1 for e in gcc if e["valid"]),
        "confirmed": sum(1 for e in gcc if e["confirmed"]),
        "compared": len(agree),
        "median_abs_diff_ms": float(np.median(np.abs(agree))) if agree else None,
        "within_2ms": float(np.mean(np.abs(agree) <= 2.0)) if agree else None,
        "median_lag_ms": float(np.median([e["lag_ms"] for e in gcc if e["valid"]])) if any(e["valid"] for e in gcc) else None,
        "aec3_realignments": [e for e in events if e["kind"] == "realign" and e.get("engine") == "aec3"],
    }
    # Speex: first alignment, and where its reference stands against the echo.
    speex_first = [e for e in events if e["kind"] == "realign" and e.get("engine") == "speex" and e.get("first")]
    sline = None
    if tracks["ref_speex"].any():
        sline = {"windows": [w for w in delay_windows(tracks["ref_speex"], inp, ~dilate(line["caller"], 10)) if w["valid"]]}
    report["speex"] = {
        "first_alignment_after_first_answer_s": (speex_first[0]["t"] - first_play) / 1000 if speex_first and first_play is not None else None,
        "realignments": [e for e in events if e["kind"] == "realign" and e.get("engine") == "speex" and not e.get("first")],
        "reference_delay_ms_median": float(np.median([w["lag_ms"] for w in sline["windows"]])) if sline and sline["windows"] else None,
        "target_ms": speex_target,
    }
    # Bypass.
    flips = [e for e in events if e["kind"] == "bypass"]
    ons = [e for e in flips if e["on"]]
    offs = [e for e in flips if not e["on"]]
    report["bypass"] = {
        "events": [{"t_ms": e["t"], "on": e["on"]} for e in flips],
        "engaged": len(ons),
        "released": len(offs),
        "first_on_after_first_answer_s": (ons[0]["t"] - first_play) / 1000 if ons and first_play is not None else None,
        "flapping": len(offs) >= 2,
        "expected_after_ms": bypass_after_ms,
    }
    # Caller speech, barge-ins, self-interruptions.
    caller = line["caller"]
    runs = []
    padded = np.concatenate(([False], caller, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    asr_frames = frames_of(tracks["asr"])
    asr_floor = noise_floor(tracks["asr"], ~dilate(line["echo_only"], 25))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        confirmed = bool(np.mean(asr_frames[a : min(b, len(asr_frames))] > asr_floor * 10 ** (9 / 10)) >= 0.3)
        if confirmed:
            runs.append({"onset_ms": a * 20.0, "end_ms": b * 20.0})
    started = [e for e in events if e["kind"] == "speech_started"]
    playing = [(r["start_ms"], (r["end_ms"] if r["end_ms"] is not None else r["start_ms"] + 3e4) + keep[1])
               for r in responses]
    outcomes = {e["item"]: e["outcome"] for e in events if e["kind"] == "barge_in_outcome"}
    barge_rows, selfs = [], []
    for e in started:
        onset = e.get("onset_ms", e["t"])
        match = [r for r in runs if -300 <= e["t"] - r["onset_ms"] <= SELF_INTERRUPT_LOOKBACK_S * 1000]
        was_playing = any(a <= e["t"] <= b for a, b in playing)
        in_tail = any(w["flush_ms"] < e["t"] <= w["until_ms"] for w in tail_windows)
        if in_tail and not match:
            continue  # AICC is still playing what we flushed: not ours to count
        if match:
            caller_onset = max(match, key=lambda r: r["onset_ms"])["onset_ms"]
            start = next((r["start_ms"] for r in responses if r["start_ms"] <= caller_onset), None)
            guard_left = max(0.0, start + guard_ms - caller_onset) if start is not None and was_playing else 0.0
            latency = e["t"] - caller_onset
            barge_rows.append({"item": e["item"], "caller_onset_ms": caller_onset, "speech_started_ms": e["t"],
                               "latency_ms": latency, "guard_left_ms": guard_left,
                               "excess_ms": latency - guard_left, "during_playback": was_playing,
                               "outcome": outcomes.get(e["item"])})
        elif was_playing or e.get("responding"):
            selfs.append({"item": e["item"], "t_ms": e["t"], "onset_ms": onset, "outcome": outcomes.get(e["item"])})
    report["caller"] = {
        "method": (
            "a 20 ms frame is caller speech when the input less the least-squares echo estimate "
            f"is {NEAR_DB:.0f} dB over the noise floor, in runs of >= {NEAR_RUN_FRAMES * 20} ms "
            "(smoothed 3 of 5), is at least {NEAR_SHARE_DB:.0f} dB of the input's energy, and the ASR track (Speex's output) is 9 dB over its own floor "
            "in 30 % of the run; the onset is the run's first frame"
        ),
        "runs": runs,
        "barge_ins": barge_rows,
        "self_interruptions": selfs,
        "echo_turns": sum(1 for e in events if e["kind"] == "echo_turn"),
        "barge_in_outcomes": {o: list(outcomes.values()).count(o) for o in ("real", "empty", "echo")},
        "caller_runs_without_speech_started": [
            r for r in runs if not any(-300 <= e["t"] - r["onset_ms"] <= 3000 for e in started)
        ],
    }
    report["residual_echo_bursts"] = detect_bursts(ref, inp, events, line["floor"], responses)
    if recording is not None:
        report["aicc_recording"] = analyse_recording(recording, report, ref, events)
    report["comparison"] = comparison(report, keep, speex_keep, speex_filter, guard_ms, bypass_after_ms)
    report["verdict"] = verdict(report, keep, speex_keep, speex_filter)
    return report


def analyse_recording(path: Path, capture_report: dict, capture_ref: np.ndarray, events: list[dict]) -> dict:
    """AICC's stereo recording at the switch: left = caller, right = bot (as the switch sends)."""
    data, rate = read_wav(path)
    if data.shape[1] < 2:
        return {"error": "the recording is not stereo"}
    left, right = to_analysis(data[:, 0], rate), to_analysis(data[:, 1], rate)
    line = model_line(right, left)
    out: dict = {"path": str(path), "rate": rate, "duration_s": len(left) / RATE,
                 "class": line["klass"]}
    if "bulk_ms" not in line:
        return out
    valid = [w for w in line["windows"] if w["valid"]]
    third = max(len(valid) // 3, 1)
    out.update(
        switch_delay_ms=line["bulk_ms"],
        windows_valid=len(valid),
        windows=len(line["windows"]),
        start_ms=float(np.median([w["lag_ms"] for w in valid[:third]])) if valid else None,
        end_ms=float(np.median([w["lag_ms"] for w in valid[-third:]])) if valid else None,
        erl_db=line.get("erl_db"),
        dispersion_ms=line.get("dispersion_ms"),
        erl_over_time=line.get("erl_over_time", []),
    )
    ours = capture_report.get("delay", {}).get("vs_model_ms_median")
    if ours is not None:
        out["end_to_end_vs_model_ms"] = ours
        out["remainder_ms"] = ours - line["bulk_ms"]
        out["split_note"] = (
            "end-to-end (echo vs our modelled playout) = switch-side path + remainder; the "
            "remainder is AICC's TX queue, the switch-to-AICC hop and the playout model's error"
        )
    out["bot_channel"] = bot_channel_flags(data[:, 1], rate, right, capture_ref, events)
    return out


def global_lag(ref: np.ndarray, other: np.ndarray, max_s: float = 180.0) -> tuple[float, float]:
    """How far ``other`` trails ``ref`` over the whole signals (ms), and the peak's z-score."""
    x, y = preemphasis(ref), preemphasis(other)
    size = 1 << (len(x) + len(y)).bit_length()
    cc = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size)), size)
    span = round(max_s * RATE)
    lags = np.concatenate((np.arange(0, span), np.arange(-span, 0)))
    values = np.concatenate((cc[:span], cc[size - span :]))
    magnitude = np.abs(values)
    peak = int(np.argmax(magnitude))
    z = float((magnitude[peak] - magnitude.mean()) / (magnitude.std() + 1e-12))
    return float(lags[peak]) * 1000 / RATE, z


def bot_channel_flags(raw: np.ndarray, rate: int, right: np.ndarray, capture_ref: np.ndarray, events: list[dict]) -> dict:
    """After a barge-in AICC's pacer can stall and send no RTP until its next frame arrives
    (a known rare race). In the bot channel (what the switch sends) that shows as (a) a shift of
    the audio against our modelled playout that was not there before the barge-in, or (b) exact
    silence where our reference says audio plays. Times are recording seconds (and capture ms)."""
    offset, z = global_lag(capture_ref, right)
    out: dict = {"alignment_lag_ms": offset, "alignment_z": z, "flags": []}
    if z < Z_VALID:
        out["note"] = "the bot channel could not be aligned with our reference: no stall check"
        return out
    lag = round(offset * RATE / 1000)
    padded = np.concatenate((capture_ref, np.zeros(max(len(right) - len(capture_ref) + abs(lag), 0))))
    shifted = delayed(padded, lag)[: max(len(right), len(capture_ref))]
    windows = [w for w in delay_windows(shifted, right) if w["valid"]]
    flushes = [e for e in events if e["kind"] == "flush"]
    reanchors = [e["t"] for e in events if e["kind"] == "reanchor"]
    first_flush = min((f["t"] + offset for f in flushes), default=None)
    before = [w["lag_ms"] for w in windows if first_flush is None or w["end_ms"] <= first_flush]
    baseline = float(np.median(before)) if before else 0.0
    out["baseline_shift_ms"] = baseline
    for w in windows:
        if first_flush is None or w["start_ms"] < first_flush:
            continue
        if abs(w["lag_ms"] - baseline) > STALL_SHIFT_MS:
            near = [f["t"] + offset for f in flushes if 0 <= w["end_ms"] - (f["t"] + offset) <= 30_000]
            out["flags"].append({
                "kind": "shift_after_barge_in", "recording_s": w["start_ms"] / 1000,
                "capture_ms": w["start_ms"] - offset, "shift_ms": w["lag_ms"] - baseline,
                "reanchored_before": any(0 <= w["end_ms"] - (r + offset) <= 5000 for r in reanchors),
                "barge_in_recording_s": near[-1] / 1000 if near else None,
            })
    if rate and len(raw):
        zero = np.concatenate(([False], raw == 0, [False]))
        edges = np.flatnonzero(np.diff(zero.astype(int)))
        active = active_frames(shifted)
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if (b - a) * 1000 / rate < STALL_ZERO_MS:
                continue
            f0, f1 = int(a / rate * 50) + 3, int(b / rate * 50) - 3  # 60 ms margins
            if f1 > f0 and f1 <= len(active) and active[f0:f1].mean() >= 0.9:
                out["flags"].append({"kind": "silence_while_playing", "recording_s": a / rate,
                                     "capture_ms": a / rate * 1000 - offset,
                                     "duration_ms": (b - a) * 1000 / rate})
    merged: list[dict] = []  # a stall shifts everything after it: one flag per run of windows
    for flag in sorted(out["flags"], key=lambda f: f["recording_s"]):
        last = merged[-1] if merged else None
        if (last and flag["kind"] == last["kind"] == "shift_after_barge_in"
                and flag["recording_s"] - last["last_recording_s"] <= 15 and abs(flag["shift_ms"] - last["shift_ms"]) < 10):
            last["last_recording_s"] = flag["recording_s"]
            last["windows"] += 1
        else:
            merged.append({**flag, "last_recording_s": flag["recording_s"], "windows": 1})
    out["flags"] = merged
    return out


def bypass_intervals(events: list[dict], duration_ms: float) -> list[tuple[float, float]]:
    out, start = [], None
    for e in (e for e in events if e["kind"] == "bypass"):
        if e["on"] and start is None:
            start = e["t"]
        elif not e["on"] and start is not None:
            out.append((start, e["t"]))
            start = None
    if start is not None:
        out.append((start, duration_ms))
    return out


def window_ncc(ref_seg: np.ndarray, window: np.ndarray, lo: int, hi: int) -> tuple[float, int]:
    """Peak normalized cross-correlation of ``window`` with ``ref_seg`` (which starts ``hi``
    samples before the window) over lags lo..hi; returns (peak, lag)."""
    size = 1 << (len(ref_seg) + len(window)).bit_length()
    cc = np.fft.irfft(np.fft.rfft(ref_seg, size) * np.conj(np.fft.rfft(window, size)), size)[: hi - lo + 1]
    # normalise by the window energy and each lag's reference energy
    squares = np.concatenate(([0.0], np.cumsum(ref_seg**2)))
    energy = squares[len(window) : len(window) + hi - lo + 1] - squares[: hi - lo + 1]
    ncc = cc / (np.linalg.norm(window) * np.sqrt(energy) + 1e-9)
    peak = int(np.argmax(np.abs(ncc)))
    return float(abs(ncc[peak])), hi - peak


def detect_bursts(ref: np.ndarray, inp: np.ndarray, events: list[dict], floor: float, responses: list[dict], seed: int = 1) -> dict:
    """Residual-echo bursts while the canceller is bypassed (the phone's own AEC leaks echo past
    a bypassed canceller). A 0.5 s window with an active reference is a burst window when its
    input correlates with the reference at 50-700 ms clearly better than with the reference of
    other times (a shuffled baseline: peak above the baseline's mean by ``BURST_BASELINE_Z``
    robust standard deviations (MAD) and ``BURST_MIN_NCC``), and its energy is ``BURST_OVER_FLOOR_DB`` over the
    noise floor. Consecutive burst windows merge."""
    duration_ms = len(inp) / RATE * 1000
    intervals = bypass_intervals(events, duration_ms)
    result: dict = {"bypassed_s": sum(b - a for a, b in intervals) / 1000, "bursts": []}
    result["method"] = (
        f"0.5 s windows with an active reference inside the bypassed stretches; peak normalized "
        f"cross-correlation at {BURST_LAGS_MS[0]:g}-{BURST_LAGS_MS[1]:g} ms against {BURST_SHUFFLES} shuffled "
        f"references (z >= {BURST_BASELINE_Z:g}, ncc >= {BURST_MIN_NCC:g}), energy >= {BURST_OVER_FLOOR_DB:g} dB over the floor"
    )
    if not intervals:
        result["note"] = "the canceller was never bypassed: no residual-echo analysis"
        return result
    rng = np.random.default_rng(seed)
    pre_ref, pre_in = preemphasis(ref), preemphasis(inp)
    count, hop = round(BURST_WINDOW_S * RATE), round(BURST_HOP_S * RATE)
    lo, hi = (round(ms * RATE / 1000) for ms in BURST_LAGS_MS)
    active = active_frames(ref)
    per = count // FRAME
    candidates = [s for s in range(hi, len(ref) - count, hop) if active[s // FRAME : s // FRAME + per].mean() >= 0.6]
    ref_active_s = len(candidates) * BURST_HOP_S
    total_active_s = float(active.sum()) * 0.02
    flagged = []
    bypassed_active_s = 0.0
    for start in range(hi, len(inp) - count, hop):
        t0 = start * 1000 / RATE
        if not any(a <= t0 and t0 + BURST_WINDOW_S * 1000 <= b for a, b in intervals):
            continue
        if active[start // FRAME : start // FRAME + per].mean() < 0.6:
            continue
        bypassed_active_s += BURST_HOP_S
        window = pre_in[start : start + count]
        level = float(np.mean(inp[start : start + count] ** 2))
        if level < floor * 10 ** (BURST_OVER_FLOOR_DB / 10):
            continue
        peak, lag = window_ncc(pre_ref[start - hi : start + count - lo], window, lo, hi)
        others = [c for c in candidates if abs(c - start) > 2 * RATE]
        if len(others) < 3:
            continue
        picks = rng.choice(others, size=min(BURST_SHUFFLES, len(others)), replace=False)
        base = np.array([window_ncc(pre_ref[c - hi : c + count - lo], window, lo, hi)[0] for c in picks])
        # median and MAD: an answer that repeats earlier audio makes a few shuffles match
        z = (peak - np.median(base)) / (1.4826 * np.median(np.abs(base - np.median(base))) + 1e-3)
        if z >= BURST_BASELINE_Z and peak >= BURST_MIN_NCC:
            ref_level = float(np.mean(ref[start - lag : start - lag + count] ** 2)) if start - lag >= 0 else 1.0
            flagged.append({"start_ms": t0, "end_ms": t0 + BURST_WINDOW_S * 1000, "lag_ms": lag * 1000 / RATE,
                            "ncc": peak, "z": float(z), "level_dbfs": db(level / 32768.0**2),
                            "over_floor_db": db(level / floor), "vs_reference_db": db(level / max(ref_level, 1e-9))})
    merged: list[dict] = []
    for w in flagged:
        if merged and w["start_ms"] - merged[-1]["end_ms"] <= BURST_MERGE_GAP_S * 1000:
            m = merged[-1]
            m["end_ms"] = w["end_ms"]
            m["parts"].append(w)
        else:
            merged.append({"start_ms": w["start_ms"], "end_ms": w["end_ms"], "parts": [w]})
    started = [e for e in events if e["kind"] == "speech_started"]
    outcomes = {e["item"]: e["outcome"] for e in events if e["kind"] == "barge_in_outcome"}
    bursts = []
    for m in merged:
        parts = m["parts"]
        mid = (m["start_ms"] + m["end_ms"]) / 2
        resp = next((r for r in responses if r["start_ms"] <= mid <= (r["end_ms"] or r["start_ms"] + 3e4) + BURST_LAGS_MS[1]), None)
        if resp is None:
            position = "outside any response"
        else:
            rs = resp["start_ms"]
            re = resp["end_ms"] if resp["end_ms"] is not None else rs + 3e4
            if m["start_ms"] < rs + BURST_START_S * 1000 + BURST_LAGS_MS[0]:
                position = "start of a response"
            elif m["start_ms"] > re - BURST_TAIL_S * 1000:
                position = "tail"
            else:
                position = "mid-answer"
        hits = [e for e in started if m["start_ms"] <= e["t"] <= m["end_ms"] + 1000]
        half = BURST_WINDOW_S * 500  # a window flags when its middle is in the burst: trim half
        bursts.append({
            "start_ms": m["start_ms"] + half, "end_ms": m["end_ms"] - half, "duration_s": (m["end_ms"] - m["start_ms"]) / 1000 - BURST_WINDOW_S,
            "lag_ms": float(np.median([p["lag_ms"] for p in parts])),
            "level_dbfs": float(np.max([p["level_dbfs"] for p in parts])),
            "over_floor_db": float(np.max([p["over_floor_db"] for p in parts])),
            "vs_reference_db": float(np.median([p["vs_reference_db"] for p in parts])),
            "position": position,
            "speech_started": bool(hits),
            "barge_in": any(h.get("barge_in") for h in hits),
            "outcomes": [outcomes.get(h["item"]) for h in hits],
        })
    result["bursts"] = bursts
    result["bot_speech_min_bypassed"] = bypassed_active_s / 60
    result["bot_speech_min_total"] = total_active_s / 60
    result["per_min_of_bot_speech"] = len(bursts) / (bypassed_active_s / 60) if bypassed_active_s else None
    result["with_speech_started"] = sum(b["speech_started"] for b in bursts)
    result["with_barge_in"] = sum(b["barge_in"] for b in bursts)
    del ref_active_s
    return result


def comparison(report: dict, keep, speex_keep, speex_filter, guard_ms, bypass_after) -> list[dict]:
    delay = report["delay"]
    speex = report["speex"]["reference_delay_ms_median"]
    disp = report.get("dispersion_ms")
    rows = [
        ("GCC keep range of the reference file's residual delay (ms)", f"[{keep[0]:g}, {keep[1]:g}]",
         f"{delay['lag_ms_median']:.1f}" if delay["lag_ms_median"] is not None else "n/a"),
        ("Speex reference: echo delay behind it (ms)", f"keep [{speex_keep[0]:g}, {speex_keep[1]:g}]",
         "n/a" if speex is None else f"{speex:.1f}"),
        ("Speex filter span vs echo dispersion (ms)", f"{speex_filter:g}",
         "n/a" if disp is None or speex is None else f"{speex + disp:.1f} (delay + 90 % dispersion {disp:.1f})"),
        ("ERL (dB)", "10 (simulation)", "n/a" if report.get("erl_db") is None else f"{report['erl_db']:.1f}"),
        ("Bypass after reference without echo (s)", "1.5",
         "n/a" if report["bypass"]["first_on_after_first_answer_s"] is None
         else f"{report['bypass']['first_on_after_first_answer_s']:.2f} (first engaged, after the first answer starts)"),
        ("Onset guard (ms)", f"{guard_ms:g}",
         f"{len([b for b in report['caller']['barge_ins'] if b['guard_left_ms'] > 0])} caller onset(s) inside it"),
    ]
    return [{"quantity": a, "assumed": b, "measured": c} for a, b, c in rows]


def verdict(report: dict, keep, speex_keep, speex_filter) -> list[dict]:
    klass = report["class"]
    out = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        out.append({"criterion": name, "result": "N/A" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})

    selfs = report["caller"]["self_interruptions"]
    add("zero self-interruptions in echo-only stretches", not selfs, f"{len(selfs)} speech_started with no caller speech before it; {report['caller']['echo_turns']} echo turn(s)")
    barge = [b for b in report["caller"]["barge_ins"] if b["during_playback"]]
    late = [b for b in barge if b["excess_ms"] > BARGE_IN_BUDGET_S * 1000]
    missed = report["caller"]["caller_runs_without_speech_started"]
    add("caller barge-in speech_started within 1.5 s of onset (past the guard)",
        None if not barge and not missed else (not late and not missed),
        f"{len(barge)} barge-in(s), worst excess over the guard {max((b['excess_ms'] for b in barge), default=0):.0f} ms; {len(late)} late; {len(missed)} caller run(s) never produced speech_started")
    steady = report["erle"].get("aec3", {}).get("steady_median_db") if report["erle"] else None
    add("median AEC3 ERLE >= 10 dB on a line with echo",
        None if klass != "echo" else (steady is not None and steady >= ERLE_TARGET_DB),
        f"line class {klass}; steady-state median {steady if steady is None else round(steady, 1)} dB")
    bypass = report["bypass"]
    if klass == "none":
        add("bypass engaged and not flapping on an echo-free line", bypass["engaged"] >= 1 and not bypass["flapping"],
            f"engaged {bypass['engaged']}x, released {bypass['released']}x")
    else:
        add("bypass engaged and not flapping on an echo-free line", None, f"line class {klass}; bypass engaged {bypass['engaged']}x, released {bypass['released']}x")
    lag = report["delay"]["lag_ms_median"]
    sx = report["speex"]["reference_delay_ms_median"]
    disp = report.get("dispersion_ms")
    if klass == "none":
        add("measured delay and dispersion inside the configured ranges", None, "no echo to measure")
    else:
        inside = keep[0] <= lag <= keep[1]
        sx_ok = sx is None or (speex_keep[0] <= sx <= speex_keep[1] and (disp is None or sx + disp <= speex_filter))
        add("measured delay and dispersion inside the configured ranges", inside and sx_ok,
            f"GCC residual {lag:.1f} ms in [{keep[0]:g}, {keep[1]:g}]; Speex echo delay {sx if sx is None else round(sx, 1)} ms in [{speex_keep[0]:g}, {speex_keep[1]:g}], plus dispersion {disp if disp is None else round(disp, 1)} ms within the {speex_filter:g} ms filter")
    add("playout model holds (no drift above a few ms)", None if klass == "none" else not report["delay"]["drift_flag"],
        f"max drift within an answer {report['delay']['max_abs_drift_ms']}, spread across answers {report['delay']['across_answers_spread_ms']} (flag above {DRIFT_FLAG_MS} ms)")
    return out


# --------------------------------------------------------------------------------- the writing


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(report: dict) -> str:
    out = [f"# Echo capture {report['session_id']}", ""]
    out += [f"- capture: `{report['capture']}`", f"- session start (UTC): {report['started_utc']}",
            f"- input length: {report['duration_s']:.1f} s", f"- line class: **{report['class']}**  ",
            f"  ({report['class_rule']})", ""]
    if "delay" not in report:
        out += [report.get("note", ""), ""]
        return "\n".join(out)
    delay = report["delay"]
    out += ["## Verdict", "", "| criterion | result | detail |", "|---|---|---|"]
    out += [f"| {v['criterion']} | {v['result']} | {v['detail']} |" for v in report["verdict"]]
    out += ["", "## Bulk delay (independent cross-correlation)", "",
            f"- reference file to input: median {fmt(delay['lag_ms_median'])} ms over {delay['windows_valid']} of {delay['windows']} windows",
            f"- after the playout model (lag + shift - lead): median {fmt(delay['vs_model_ms_median'])} ms",
            f"- least-squares filter's dominant tap: {fmt(delay['filter_peak_ms'])} ms",
            f"- drift flag (> {DRIFT_FLAG_MS} ms within or across answers): **{delay['drift_flag']}**; "
            f"max within an answer {fmt(delay['max_abs_drift_ms'])} ms, spread of answer medians {fmt(delay['across_answers_spread_ms'])} ms", "",
            "| answer | starts (ms) | windows | start (ms) | end (ms) | drift (ms) | median (ms) |", "|---|---|---|---|---|---|---|"]
    for r in delay["per_response"]:
        out.append(f"| {r['item'][-6:]} | {fmt(r['start_ms'], 0)} | {r['windows']} | {fmt(r.get('start_vs_model_ms'))} | {fmt(r.get('end_vs_model_ms'))} | {fmt(r.get('drift_ms'))} | {fmt(r.get('median_vs_model_ms'))} |")
    out += ["", "(start, end and median are delays after the playout model, from the first and last third of the answer's valid windows.)", "",
            "## Echo path", "",
            f"- ERL: {fmt(report.get('erl_db'))} dB (reference energy over the input's energy less the noise floor, on echo-only frames the fitted filter covers, 300-3400 Hz; the filter's own estimate: {fmt(report.get('erl_ls_db'))} dB)",
            f"- dispersion (90 % of the filter's energy): {fmt(report.get('dispersion_ms'))} ms",
            f"- linear fit: the filter removes {fmt(report.get('fit_gain_db'))} dB of the input's energy on echo-only frames (low values on a line with echo mean a time-varying or nonlinear path)",
            f"- echo over the line's noise floor: {fmt(report.get('echo_over_noise_db'))} dB (noise floor {fmt(report.get('noise_floor_dbfs'))} dBFS per frame)", ""]
    if report["erl_over_time"]:
        out += ["| from (s) | ERL (dB) |", "|---|---|"]
        out += [f"| {b['t_ms'] / 1000:.0f} | {b['erl_db']:.1f} |" for b in report["erl_over_time"]]
        out.append("")
    out += ["## ERLE (0.5 s windows of echo-only frames, 300-3400 Hz)", "",
            "| canceller | windows | convergence to 10 dB after the first answer starts (s) | steady-state median (dB) | median of all (dB) |", "|---|---|---|---|---|"]
    for name, e in report["erle"].items():
        out.append(f"| {'AEC3' if name == 'aec3' else 'Speex (ASR path)'} | {e['windows']} | {fmt(e['convergence_s'], 2)} | {fmt(e['steady_median_db'])} | {fmt(e['median_db'])} |")
    g = report["gateway_estimator"]
    out += ["", "(Speex's output includes SpeexDSP's preprocessor, whose noise suppression adds to its ERLE.)", "",
            "## Gateway estimator and alignment", "",
            f"- GCC-PHAT windows {g['windows']}, valid {g['valid']}, confirmed {g['confirmed']}; median lag {fmt(g['median_lag_ms'])} ms",
            f"- against the independent delay: {g['compared']} compared, median |difference| {fmt(g['median_abs_diff_ms'], 2)} ms, {fmt(None if g['within_2ms'] is None else g['within_2ms'] * 100, 0)} % within 2 ms",
            f"- AEC3 re-alignments: {len(g['aec3_realignments'])}",
            f"- Speex first aligned {fmt(report['speex']['first_alignment_after_first_answer_s'], 2)} s after the first answer starts; later re-alignments {len(report['speex']['realignments'])}; its reference stands {fmt(report['speex']['reference_delay_ms_median'])} ms ahead of the echo (target {report['speex']['target_ms']:g})",
            f"- bypass: engaged {report['bypass']['engaged']}x (first {fmt(report['bypass']['first_on_after_first_answer_s'], 2)} s after the first answer starts), released {report['bypass']['released']}x, flapping: {report['bypass']['flapping']}", ""]
    caller = report["caller"]
    out += ["## Caller, barge-ins and self-interruptions", "", f"Caller onset method: {caller['method']}.", "",
            f"- caller speech runs: {len(caller['runs'])}; self-interruptions: {len(caller['self_interruptions'])}; echo turns: {caller['echo_turns']}; barge-in outcomes: {caller['barge_in_outcomes']}", ""]
    if caller["barge_ins"]:
        out += ["| caller onset (s) | speech_started (s) | latency (ms) | guard left (ms) | excess (ms) | playing | outcome |", "|---|---|---|---|---|---|---|"]
        out += [f"| {b['caller_onset_ms'] / 1000:.2f} | {b['speech_started_ms'] / 1000:.2f} | {b['latency_ms']:.0f} | {b['guard_left_ms']:.0f} | {b['excess_ms']:.0f} | {b['during_playback']} | {b['outcome']} |" for b in caller["barge_ins"]]
        out.append("")
    if caller["self_interruptions"]:
        out += ["Self-interruptions: " + ", ".join(f"{s['t_ms'] / 1000:.2f} s" for s in caller["self_interruptions"]), ""]
    bursts = report["residual_echo_bursts"]
    out += ["## Residual-echo bursts while the canceller is bypassed", "", f"Method: {bursts['method']}.", ""]
    if bursts.get("note"):
        out += [bursts["note"], ""]
    else:
        per_min = bursts["per_min_of_bot_speech"]
        out += [f"- bypassed for {bursts['bypassed_s']:.1f} s ({bursts['bot_speech_min_bypassed'] * 60:.1f} s of bot speech); bursts: {len(bursts['bursts'])} ({fmt(per_min, 2)} per minute of bot speech); with speech_started: {bursts['with_speech_started']}; with a barge-in: {bursts['with_barge_in']}", ""]
        if bursts["bursts"]:
            out += ["| start (s) | duration (s) | lag (ms) | level (dBFS) | over floor (dB) | vs reference (dB) | position | speech_started | barge-in | outcome |", "|---|---|---|---|---|---|---|---|---|---|"]
            out += [f"| {b['start_ms'] / 1000:.2f} | {b['duration_s']:.2f} | {b['lag_ms']:.0f} | {b['level_dbfs']:.1f} | {b['over_floor_db']:.1f} | {b['vs_reference_db']:.1f} | {b['position']} | {b['speech_started']} | {b['barge_in']} | {b['outcomes']} |" for b in bursts["bursts"]]
            out.append("")
    out += ["## Against the simulation's assumptions", "", "| quantity | assumed | measured |", "|---|---|---|"]
    out += [f"| {c['quantity']} | {c['assumed']} | {c['measured']} |" for c in report["comparison"]]
    out.append("")
    rec = report.get("aicc_recording")
    if rec:
        out += ["## AICC recording (switch side)", ""]
        if "switch_delay_ms" not in rec:
            out += [f"No echo found in the recording (class {rec['class']}).", ""]
        else:
            out += [f"- class {rec['class']}; switch-side echo delay {fmt(rec['switch_delay_ms'])} ms ({rec['windows_valid']} of {rec['windows']} windows; start {fmt(rec['start_ms'])}, end {fmt(rec['end_ms'])})",
                    f"- switch-side ERL {fmt(rec['erl_db'])} dB, dispersion {fmt(rec['dispersion_ms'])} ms"]
            if "remainder_ms" in rec:
                out += [f"- our end-to-end delay after the model {fmt(rec['end_to_end_vs_model_ms'])} ms = switch side {fmt(rec['switch_delay_ms'])} ms + remainder {fmt(rec['remainder_ms'])} ms ({rec['split_note']})"]
            out.append("")
        bot = rec.get("bot_channel")
        if bot:
            out += ["### Bot channel after barge-ins (AICC pacer stall check)", ""]
            if bot.get("note"):
                out += [bot["note"], ""]
            elif not bot["flags"]:
                out += [f"No stall or discontinuity found (bot channel aligned with our reference at {fmt(bot['alignment_lag_ms'])} ms, baseline shift {fmt(bot['baseline_shift_ms'])} ms).", ""]
            else:
                out += ["Report these to AICC (recording time; the call's start is 0):", "", "| kind | recording (s) | detail |", "|---|---|---|"]
                for f in bot["flags"]:
                    detail = (f"audio shifted by {f['shift_ms']:.0f} ms against our playout; after a re-anchor: {f['reanchored_before']}"
                              if f["kind"] == "shift_after_barge_in" else f"silent for {f['duration_ms']:.0f} ms while our audio should play")
                    out.append(f"| {f['kind']} | {f['recording_s']:.2f} | {detail} |")
                out.append("")
    return "\n".join(out)


def clean(value):
    """JSON-safe: numpy scalars and arrays out, NaN to null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--aicc-recording", nargs="*", type=Path, default=[],
                        help="AICC's stereo recording per capture, in the same order")
    args = parser.parse_args()
    if args.aicc_recording and len(args.aicc_recording) != len(args.captures):
        raise SystemExit("give one --aicc-recording per capture directory, in the same order")
    for index, directory in enumerate(args.captures):
        recording = args.aicc_recording[index] if args.aicc_recording else None
        report = analyse_capture(directory, recording)
        (directory / "report.json").write_text(json.dumps(clean(report), indent=2) + "\n")
        (directory / "report.md").write_text(markdown(report) + "\n")
        print(f"{directory}: class {report['class']}; report.md, report.json written")
        for v in report.get("verdict", []):
            print(f"  {v['result']:4} {v['criterion']}")


if __name__ == "__main__":
    main()

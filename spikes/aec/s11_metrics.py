"""S11 part 2 metrics (A3 section 5 "Metrics"). Throwaway.

EO (echo-only) = far end active at the capture (played downlink shifted by D, 30 dB under the far-end
level, 50 ms hangover, +20 ms for dispersion) and no near-end speech within 200 ms (ground truth).

ERLE(W) = 10 log10(sum c^2 / sum e^2) over 200 ms windows (100 ms hop) whose twenty 10 ms frames are all
EO; c is the capture as the gateway receives it (before any pre-filter) and e the canceller output, so
for AEC3 it includes suppression and for _pf variants the removed 4-8 kHz images.
ERLE_ss = median from 5 s of far-end-active time after the onset until the change (and from 5 s after
the change to the end).
Convergence = far-end-active time from the onset (or the change) until ERLE stays >= threshold for 1 s
of EO (ten consecutive EO windows); computed for 20 dB (A3) and, for reference, for ERL + ERLE >= 20 dB.
Silero (v6.2.3 via the spikes/cpu wrapper, one continuous stream per signal): rate = EO windows with
p >= 0.5 / EO windows; speech_started = hysteresis run (0.5 / 0.35) reaching min_speech_ms (150, 400),
counted as false when no near-end speech is present from the run start to the confirmation.
"""

from __future__ import annotations

import s11_common as c

import numpy as np  # noqa: E402

WIN, HOP = 20, 10  # 200 ms windows, 100 ms hop, in 10 ms frames
F_S = c.FRAME / c.SR  # 0.01 s


def frame_energy(x: np.ndarray, nf: int) -> np.ndarray:
    return np.sum((np.asarray(x[: nf * c.FRAME], dtype=np.float64).reshape(nf, c.FRAME)) ** 2, axis=1)


def erle_windows(cin: np.ndarray, cout: np.ndarray, eo: np.ndarray) -> dict:
    nf = min(len(cin), len(cout)) // c.FRAME
    nf = min(nf, len(eo))
    ein, eout = frame_energy(cin, nf), frame_energy(cout, nf)
    starts, vals = [], []
    for s in range(0, nf - WIN + 1, HOP):
        if eo[s:s + WIN].all():
            num, den = ein[s:s + WIN].sum(), eout[s:s + WIN].sum()
            starts.append(s)
            vals.append(10 * np.log10((num + 1e-9) / (den + 1e-9)))
    return {"start": np.asarray(starts, dtype=np.int64), "erle": np.asarray(vals)}


def measurable(cin: np.ndarray, clean: np.ndarray, w: dict, enr_db: float = 25.0) -> np.ndarray:
    """Per ERLE window: the echo stands >= enr_db above the no-echo floor (the clean chain = near end +
    noise, which in an EO window is just the noise). Speech has quiet stretches where the echo sits at
    the noise floor; there c/e is ~0 dB whatever the canceller does. DEVIATION from A3 (forced by the
    data): convergence and the post-double-talk check are also computed on measurable windows only."""
    nf = min(len(cin), len(clean)) // c.FRAME
    ein, ecl = frame_energy(cin, nf), frame_energy(clean, nf)
    thr = 10 ** (enr_db / 10)
    return np.array([ein[s:s + WIN].sum() >= thr * ecl[s:s + WIN].sum() for s in w["start"]], dtype=bool)


def far_time(far_active: np.ndarray, frm: int, to: int) -> float:
    """Far-end-active time (s) in frames [frm, to)."""
    return float(np.count_nonzero(far_active[frm:max(frm, to)])) * F_S


def onset_frame(far_active: np.ndarray, frm: int = 0) -> int | None:
    idx = np.flatnonzero(far_active[frm:])
    return int(frm + idx[0]) if idx.size else None


def convergence(w: dict, far_active: np.ndarray, frm: int, to: int, thr: float) -> float | None:
    """Far-end-active time from frame `frm` to the end of the first window of a run of >= 10 consecutive
    EO windows (1 s of EO) that all have ERLE >= thr; None if never before frame `to`."""
    s, e = w["start"], w["erle"]
    sel = (s >= frm) & (s + WIN <= to)
    s, e = s[sel], e[sel]
    run = 0
    for i in range(len(s)):
        if i > 0 and s[i] - s[i - 1] != HOP:  # a non-EO gap does not reset the 1 s of EO, a low ERLE does
            pass
        if e[i] >= thr:
            run += 1
            if run >= 10:
                first = s[i - 9]
                return round(far_time(far_active, frm, first + WIN), 3)
        else:
            run = 0
    return None


def median_in(w: dict, far_active: np.ndarray, frm: int, to: int, min_far_s: float) -> float | None:
    s, e = w["start"], w["erle"]
    ok = [(e[i]) for i in range(len(s)) if s[i] >= frm and s[i] + WIN <= to and far_time(far_active, frm, s[i]) >= min_far_s]
    return round(float(np.median(ok)), 2) if ok else None


BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 15), (15, 60)]


def erle_curve(w: dict, far_active: np.ndarray, frm: int, to: int) -> list:
    """Median ERLE in bins of far-end-active time since frame `frm` (ERLE over time, compact)."""
    s, e = w["start"], w["erle"]
    out = []
    ft = np.array([far_time(far_active, frm, si + WIN) for si in s]) if len(s) else np.zeros(0)
    for a, b in BINS:
        sel = (s >= frm) & (s + WIN <= to) & (ft >= a) & (ft < b)
        out.append(round(float(np.median(e[sel])), 2) if np.any(sel) else None)
    return out


# ------------------------------------------------------------------------------------------ Silero


def silero_masks(n_win: int, eo: np.ndarray) -> np.ndarray:
    """Silero window i (512 samples) is EO when every 10 ms frame it overlaps is EO."""
    out = np.zeros(n_win, dtype=bool)
    for i in range(n_win):
        a = (i * c.SILERO_WIN) // c.FRAME
        b = ((i + 1) * c.SILERO_WIN - 1) // c.FRAME
        if b < len(eo):
            out[i] = bool(eo[a:b + 1].all())
    return out


def win_time(i: int) -> float:
    return i * c.SILERO_WIN / c.SR


def silero_metrics(p: np.ndarray, eo: np.ndarray, far_active: np.ndarray, near_present: np.ndarray,
                   onsets: list[float], t_change: float | None) -> dict:
    n = len(p)
    eow = silero_masks(n, eo)
    t = np.array([win_time(i) for i in range(n)])
    fr = (t / F_S).astype(np.int64)
    cum = np.cumsum(far_active) * F_S  # far-active time up to frame
    far_t = np.array([cum[min(f, len(cum) - 1)] for f in fr])
    if t_change is not None:
        fc = int(round(t_change / F_S))
        after = t >= t_change
        far_since_change = np.where(after, far_t - (cum[fc - 1] if fc > 0 else 0.0), np.inf)
    else:
        after = np.zeros(n, dtype=bool)
        far_since_change = np.full(n, np.inf)
    in_onset1 = np.zeros(n, dtype=bool)
    for o in onsets:
        in_onset1 |= (t >= o) & (t < o + 1.0)
    hit = p >= c.SILERO_ON

    def rate(mask):
        m = eow & mask
        k = int(np.count_nonzero(m))
        return {"eo_windows": k, "rate": round(float(np.count_nonzero(hit & m)) / k, 5) if k else None}

    res = {
        "all_eo": rate(np.ones(n, dtype=bool)),
        "after_1s": rate((far_t >= 1.0) & (far_since_change >= 1.0)),
        "after_2s": rate((far_t >= 2.0) & (far_since_change >= 2.0)),
        "after_1s_ignoring_change": rate(far_t >= 1.0),
        "first_1s_of_playbacks": rate(in_onset1),
        "after_1s_not_onset": rate((far_t >= 1.0) & (far_since_change >= 1.0) & ~in_onset1),
        "change_first_2s": rate(after & (far_since_change < 2.0)),
    }
    ev = {}
    for ms in (150, 400):
        evs = []
        for a, confirm, b in c.vad_runs(p, ms):
            if confirm < 0:
                continue
            fa_, fb_ = int(win_time(a) / F_S), int(win_time(confirm + 1) / F_S)
            if np.any(near_present[fa_:fb_ + 1]):
                continue
            tc = win_time(confirm + 1)
            since = [tc - o for o in onsets if o <= tc]
            evs.append({"t": round(tc, 3), "echo": bool(np.any(far_active[fa_:fb_ + 1])),
                        "since_onset_s": round(min(since), 3) if since else None,
                        "far_t": round(float(far_t[min(confirm, n - 1)]), 3),
                        "since_change_far_s": None if not np.isfinite(far_since_change[min(confirm, n - 1)])
                        else round(float(far_since_change[min(confirm, n - 1)]), 3)})
        ev[str(ms)] = {
            "false_total": len(evs),
            "false_after_1s": sum(1 for e in evs if e["far_t"] >= 1.0 and (e["since_change_far_s"] is None or e["since_change_far_s"] >= 1.0)),
            "false_after_2s": sum(1 for e in evs if e["far_t"] >= 2.0 and (e["since_change_far_s"] is None or e["since_change_far_s"] >= 2.0)),
            "events": evs[:50],
        }
    res["speech_started"] = ev
    return res


# ------------------------------------------------------------------------------------- near end


def band(x: np.ndarray) -> np.ndarray:
    h = c.bandpass_taps(200.0, 3800.0, c.SR, 127)
    return np.convolve(np.asarray(x, dtype=np.float64), h)[: len(x)]


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 200) -> int:
    """Lag (samples) of b relative to a that maximizes their correlation, in [-40, max_lag]."""
    n = min(len(a), len(b))
    best, best_v = 0, -np.inf
    for lag in range(-40, max_lag + 1, 1):
        if lag >= 0:
            v = float(np.dot(a[: n - lag], b[lag:n]))
        else:
            v = float(np.dot(a[-lag:n], b[: n + lag]))
        if v > best_v:
            best, best_v = lag, v
    return best


def sdr(ref: np.ndarray, est: np.ndarray, mask16: np.ndarray, lag: int) -> float | None:
    """Scale-invariant SDR (dB) of est vs ref over masked samples, est advanced by `lag`."""
    if lag > 0:
        est = np.concatenate([est[lag:], np.zeros(lag)])
    elif lag < 0:
        est = np.concatenate([np.zeros(-lag), est[:lag]])
    r, e = ref[mask16], est[mask16]
    if r.size == 0 or np.dot(r, r) == 0:
        return None
    a = np.dot(e, r) / np.dot(r, r)
    s = a * r
    return round(10 * np.log10(np.dot(s, s) / (np.dot(e - s, e - s) + 1e-9)), 2)


def spec_sdr(ref: np.ndarray, est: np.ndarray, frame_mask: np.ndarray, lag: int) -> float | None:
    """Phase-insensitive distortion: SDR of STFT magnitudes (512-sample Hann, hop 160, 200-3800 Hz) of
    est (advanced by `lag`) against ref, over STFT frames whose centre 10 ms frame is in frame_mask, with
    one fitted gain. Unlike the waveform SDR it ignores AEC3's high-pass phase and its fractional delay."""
    n = min(len(ref), len(est) - max(lag, 0))
    if n < 4096:
        return None
    x = np.asarray(ref[:n], dtype=np.float64)
    y = np.asarray(est[max(lag, 0):max(lag, 0) + n], dtype=np.float64)
    win = np.hanning(512)
    starts = np.arange(0, n - 512, 160)
    centre = (starts + 256) // c.FRAME
    keep = centre < len(frame_mask)
    starts, centre = starts[keep], centre[keep]
    starts = starts[frame_mask[centre]]
    if len(starts) < 10:
        return None
    f = np.fft.rfftfreq(512, 1 / c.SR)
    band_sel = (f >= 200) & (f <= 3800)
    X = np.abs(np.fft.rfft(np.stack([x[s:s + 512] * win for s in starts]), axis=1))[:, band_sel]
    Y = np.abs(np.fft.rfft(np.stack([y[s:s + 512] * win for s in starts]), axis=1))[:, band_sel]
    g = float(np.sum(X * Y) / np.sum(X * X))
    return round(10 * np.log10(np.sum((g * X) ** 2) / (np.sum((Y - g * X) ** 2) + 1e-9)), 2)


def near_metrics(arr: dict, out16: np.ndarray, meta: dict, spectral: bool = False) -> dict:
    cap, clean = arr["capture"], arr["clean"]
    nf = min(len(cap), len(out16)) // c.FRAME
    na, fa = arr["near_active"][:nf], arr["far_active"][:nf]
    ein, eout = frame_energy(cap, nf), frame_energy(out16, nf)
    st = na & ~fa  # near-end single talk
    dt = na & fa
    res = {"single_talk_frames": int(st.sum()), "double_talk_frames": int(dt.sum())}
    if st.any():
        res["single_talk_gain_db"] = round(10 * np.log10(eout[st].sum() / ein[st].sum()), 2)
        per = []
        for s_, e_ in [(int(u["start"] / F_S), int(u["end"] / F_S)) for u in meta["near"]]:
            m = st[s_:e_]
            if m.sum() >= 20:
                per.append(10 * np.log10(eout[s_:e_][m].sum() / ein[s_:e_][m].sum()))
        res["single_talk_gain_db_worst_utt"] = round(float(min(per)), 2) if per else None
    if dt.any():
        bc, bo, bi = band(clean[: nf * c.FRAME]), band(out16[: nf * c.FRAME]), band(cap[: nf * c.FRAME])
        m16 = np.repeat(dt, c.FRAME)
        idx = np.flatnonzero(dt)
        a, b = idx[0] * c.FRAME, min((idx[0] + 1000) * c.FRAME, (idx[-1] + 1) * c.FRAME)  # <= 10 s, contiguous
        lag = best_lag(bc[a:b], bo[a:b]) if b - a > 1600 else 0
        res["dt_lag_samples"] = lag
        res["dt_sdr_out_db"] = sdr(bc, bo, m16, lag)
        res["dt_sdr_in_db"] = sdr(bc, bi, m16, 0)
        if spectral:
            res["dt_spec_sdr_out_db"] = spec_sdr(clean, out16, dt, max(lag, 0))
            res["dt_spec_sdr_in_db"] = spec_sdr(clean, cap, dt, 0)
    if spectral and st.any():
        idx = np.flatnonzero(st)
        a, b = idx[0] * c.FRAME, min((idx[0] + 1000) * c.FRAME, (idx[-1] + 1) * c.FRAME)
        lag_st = best_lag(band(clean[a:b]), band(out16[a:b])) if b - a > 1600 else 0
        res["st_spec_sdr_out_db"] = spec_sdr(clean, out16, st, max(lag_st, 0))
        res["st_spec_sdr_in_db"] = spec_sdr(clean, cap, st, 0)
    return res


def post_dt(w: dict, meta: dict, far_active: np.ndarray) -> list[dict]:
    """G.168 3B-style: ERLE in the EO windows right after each double-talk burst vs the 3 s before it."""
    out = []
    s, e = w["start"], w["erle"]
    fc = int(round(meta["t_change"] / F_S)) if meta.get("t_change") is not None else len(far_active)
    for u in meta["near"]:
        a, b = int(u["start"] / F_S), int(u["end"] / F_S)
        if far_time(far_active, a, b) < 1.0:
            continue  # not a double-talk burst
        pre = e[(s + WIN <= a) & (s >= a - 300)]
        sel = (s >= b + 20) & (s <= b + 200) & (s + WIN <= fc)  # EO windows 0.2-2 s after it, before a path change
        post, post_s = e[sel][:5], s[sel][:5]  # the first 0.5 s of them
        if len(pre) >= 3 and len(post) >= 3:
            out.append({"utt": u["utt"], "end": u["end"], "erle_pre": round(float(np.median(pre)), 2),
                        "erle_post": round(float(np.median(post)), 2),
                        "drop_db": round(float(np.median(pre) - np.median(post)), 2)})
    return out

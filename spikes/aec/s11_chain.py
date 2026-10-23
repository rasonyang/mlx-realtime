"""S11 part 2: build one simulated telephone call (signals + ground truth) for a condition. Throwaway.

Chain (spikes/results/a3_aec_research.md section 5; deviations are marked DEVIATION):

 1. Far end. Manifest zh/en speech (16 kHz copies) resampled to 24 kHz with soxr HQ = the gateway's
    TTS output. DEVIATION: no S3 TTS WAVs existed when this ran, so manifest speech stands in for
    Qwen3-TTS (the B4 brief allows exactly this).
 2. Send schedule. TTS cadence from S3 (spikes/results/s3_stream.json, Qwen3-TTS 0.6B 8-bit, warm):
    first 0.32 s chunk after 0.10 s, then one 0.32 s chunk every 0.32 x RTF s (RTF 0.29). Deltas are
    sent as soon as available, capped at max_lead_ms = 10 s ahead of real time (D6). One slice runs at
    RTF 1.1 to force underruns.
 3. True playout. Gateway -> AICC transit 3 ms + Exp(1 ms) per chunk, then AICC's playout state
    machine (voice/rtp.go Run): 20 ms ticks at a random phase; start at >= 3 queued frames, or at
    >= 1 frame once the 60 ms prebuffer deadline set on an empty tick has passed; while playing, an
    empty queue plays silence and more than 2 empty ticks stop playback (re-buffer).
 4. Downlink. AICC's 33-tap Hamming windowed sinc, decimate by 3 (media/resample.go), 20 ms framer
    with silence padding at the end of a response (aicall/framer.go), G.711 round trip.
 5. Echo. G.168 Annex D path g(k) = 10^(-ERL/20) K_css m_i(k) (Eq. D-1), then the bulk delay D (all
    path delays from AICC's tick to the gateway's capture index lumped into D). An echo-path change
    switches the model at a fixed capture time.
 6. Near end (8 kHz, soxr HQ from the 16 kHz copies) and band-limited (300-3400 Hz) white noise.
    Levels per G.168 6.4.1.2: far end at -20 dBm0 (P.56 active level at the 8 kHz Rin point), near end
    at -20 dBm0 + near_rel_db, noise at noise_dbm0 (RTP power). 0 dBm0 = -6.2 dBov.
 7. Uplink. G.711 round trip, AICC's per-20-ms-frame linear 8 -> 24 kHz upsampler (last sample held),
    soxr HQ 24 -> 16 kHz, int16. This is the capture the canceller sees.
 8. Reference on the modelled timeline. The gateway replays the same state machine on its own send
    log with transit 0 and tick phase 0 (it knows neither), adds the timeline-model error eps
    (positive = the model believes playback happened later than it did), and leads by 60 ms:
    ref[n] = TTS audio (soxr 24 -> 16 kHz) placed at model_play - 60 ms + eps.
    The residual delay the canceller must absorb is R = D + 60 ms + (true - modelled play) - eps.
 9. Oracle reference: the same audio placed at the true play time + D - 20 ms margin.
10. Ground truth kept: far-end activity at the capture (from the played downlink, shifted by D),
    near-end activity, playback onsets, near-end utterance spans, the clean chain
    (near end + noise through the same uplink, no echo) for SDR and the ASR baseline, and Silero
    probabilities of the capture (the "before" numbers).
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass, field

import s11_common as c

import numpy as np  # noqa: E402  (after s11_common, which caps BLAS threads)

LEAD_S = 0.060  # causality lead of the reference (A3 design point 4)
MARGIN_S = 0.020  # oracle / GCC-PHAT shift leaves the echo this far after the reference
TTFC_S = 0.100  # S3: warm time to first chunk ~103 ms
CHUNK_S = 0.32  # S3: streaming_interval 0.32 s
RTF = 0.29  # S3: steady RTF 0.284
TRANSIT_S, TRANSIT_JIT_S = 0.003, 0.001
MAX_LEAD_S = 10.0  # output.max_lead_ms (D6)
FAR_DBM0 = -20.0
CHANGE_T = 45.0  # echo-path change (std60 layout)
DT_START = 10.0  # first near-end utterance in a dt-layout call
DT_GAP = 2.5  # far-end-only time after each double-talk burst (post-DT residual, G.168 3B)

FRAMES_PER_CHUNK = round(CHUNK_S / c.TICK_S)  # 16


@dataclass
class Cond:
    cid: str
    part: str
    layout: str = "std60"  # std60 | dt
    path: str = "m1"  # m1..m8, or "open" (no echo)
    path2: str | None = None  # model after the change (std60); default: the next model + 4
    erl: float = 10.0
    d_ms: int = 120
    eps_ms: float = 0.0
    eps_step_t: float | None = None  # model play time (s) where eps changes to eps2_ms
    eps2_ms: float = 0.0
    law: str = "ulaw"
    noise_dbm0: float = -65.0
    rtf: float = RTF
    near_rel_db: float = 0.0
    lang: str = "zh"
    seed: int = 0
    near_ids: list[str] = field(default_factory=list)  # dt layout: the near-end utterances

    @property
    def seed_int(self) -> int:
        return zlib.crc32(f"{self.cid}|{self.seed}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------------------- material

_FAR_CACHE: dict[str, dict] = {}
_NEAR_CACHE: dict[tuple[str, float], np.ndarray] = {}
_ROWS: dict[str, dict] | None = None


def rows() -> dict[str, dict]:
    global _ROWS
    if _ROWS is None:
        _ROWS = {r["id"]: r for r in c.manifest()}
    return _ROWS


def far_material(uid: str) -> dict:
    """The far-end utterance as the gateway's TTS output (24 kHz int16), levelled so that AICC's 8 kHz
    downlink copy has a P.56 active level of -20 dBm0; plus the gateway's 16 kHz reference copy and
    AICC's 8 kHz copy."""
    if uid in _FAR_CACHE:
        return _FAR_CACHE[uid]
    x16, sr = c.read_wav(c.DATA / rows()[uid]["path_16k"])
    assert sr == c.SR
    x24 = c.soxr_resample(x16, c.SR, c.SR_TTS)
    lvl, _ = c.p56_active_level(c.aicc_downsample(c.to_i16(x24)).astype(np.float64) / 32768.0, c.SR_TEL)
    gain = 10 ** ((c.dbm0_to_dbov(FAR_DBM0) - lvl) / 20)
    x24 = c.to_i16(x24 * gain)
    m = {"x24": x24, "x8": c.aicc_downsample(x24), "x16": c.to_i16(c.soxr_resample(x24, c.SR_TTS, c.SR)),
         "gain_db": 20 * math.log10(gain), "dur": len(x24) / c.SR_TTS}
    _FAR_CACHE[uid] = m
    return m


def near_material(uid: str, dbm0: float) -> np.ndarray:
    """The near-end utterance at 8 kHz (soxr HQ, as A2's telephone copies), P.56 level `dbm0`, float."""
    key = (uid, dbm0)
    if key not in _NEAR_CACHE:
        x16, sr = c.read_wav(c.DATA / rows()[uid]["path_16k"])
        x8 = c.soxr_resample(x16, c.SR, c.SR_TEL)
        lvl, _ = c.p56_active_level(x8 / 32768.0, c.SR_TEL)
        _NEAR_CACHE[key] = x8 * 10 ** ((c.dbm0_to_dbov(dbm0) - lvl) / 20)
    return _NEAR_CACHE[key]


def lang_ids(lang: str) -> list[str]:
    return sorted(uid for uid, r in rows().items() if r["set"] == lang)


# ----------------------------------------------------------------------------------------- layouts


def layout(cond: Cond, rng: np.random.Generator) -> dict:
    """Far-end requests and near-end placements (seconds, capture clock)."""
    ids = lang_ids(cond.lang)
    if cond.layout == "dt":
        near_ids = list(cond.near_ids)
        far_pool = [u for u in ids if u not in near_ids]
    else:
        perm = [str(u) for u in rng.permutation(ids)]
        near_ids, far_pool = perm[:12], perm[12:]
    far_pool = [str(u) for u in rng.permutation(far_pool)]
    fi = 0
    used_far: set[str] = set()

    def dur(u: str) -> float:
        return far_material(u)["dur"]

    def ndur(u: str) -> float:
        return rows()[u]["duration_s"]

    def next_far(max_dur: float = 1e9) -> str | None:
        """The next far-end utterance in the shuffled pool that fits, preferring unused ones."""
        nonlocal fi
        order = [far_pool[(fi + i) % len(far_pool)] for i in range(len(far_pool))]
        for cand in [u for u in order if u not in used_far] + order:
            if dur(cand) <= max_dur:
                fi = far_pool.index(cand) + 1
                used_far.add(cand)
                return cand
        return None

    far, near = [], []
    if cond.layout == "dt":
        t = DT_START
        for u in near_ids:
            near.append((u, round(t, 3)))
            t += ndur(u) + DT_GAP
        total = math.ceil(t + 0.5)
        tf = 0.5
        while tf < total:
            u = next_far()
            far.append((u, round(tf, 3)))
            tf += TTFC_S + dur(u) + 0.2
        return {"far": far, "near": near, "duration": float(total), "t_change": None,
                "phases": {"eo_until": DT_START}}

    # std60: EO ~0-15 s, alternating single talk ~15-30 s, double talk ~30-44 s, path change at 45 s.
    used_near: set[str] = set()

    def next_near(max_dur: float) -> str | None:
        for u in sorted(near_ids, key=lambda x: (x in used_near, near_ids.index(x))):
            if ndur(u) <= max_dur:
                used_near.add(u)
                return u
        return None

    t = 0.5
    while True:
        u = next_far(15.0 - t - TTFC_S)
        if u is None:
            break
        far.append((u, round(t, 3)))
        t += TTFC_S + dur(u) + 0.5
    t_alt = t
    turn = "near"
    while True:
        if turn == "near":
            u = next_near(30.0 - t)
            if u is not None:
                near.append((u, round(t, 3)))
                t += ndur(u) + 0.5
        else:
            u = next_far(30.0 - t - TTFC_S)
            if u is not None:
                far.append((u, round(t, 3)))
                t += TTFC_S + dur(u) + 0.5
        if u is None:
            break
        turn = "far" if turn == "near" else "near"
    t_dt = t
    tf = t_dt
    while tf < 60.0:
        u = next_far()
        far.append((u, round(tf, 3)))
        tf += TTFC_S + dur(u) + 0.2
    tn = t_dt + 1.0
    while True:
        u = next_near(CHANGE_T - 1.0 - tn)
        if u is None:
            break
        near.append((u, round(tn, 3)))
        tn += ndur(u) + 1.0
    return {"far": far, "near": near, "duration": 60.0, "t_change": CHANGE_T,
            "phases": {"eo_until": round(t_alt, 3), "alt": [round(t_alt, 3), round(t_dt, 3)], "dt": [round(t_dt, 3), CHANGE_T]}}


# ----------------------------------------------------------------------------- send + AICC playout


def send_times(req_t: float, n_frames: int, rtf: float, first_send: float | None) -> np.ndarray:
    """Per-frame send time: chunk i (16 frames) is available at req + TTFC + i * CHUNK * RTF; every delta
    is sent at once, but never more than MAX_LEAD_S ahead of real time since the first send."""
    f = np.arange(n_frames)
    t = req_t + TTFC_S + (f // FRAMES_PER_CHUNK) * CHUNK_S * rtf
    if first_send is not None:
        t = np.maximum(t, first_send + (f + 1) * c.TICK_S - MAX_LEAD_S)
    return t


def aicc_playout(arrivals: np.ndarray, phase: float, n_ticks: int) -> tuple[np.ndarray, list[int]]:
    """voice/rtp.go RTPSession.Run on frame arrival times (sorted). Returns the play tick of every frame
    (-1 = not played within the call) and the ticks where a playout run starts."""
    play = np.full(len(arrivals), -1, dtype=np.int64)
    starts: list[int] = []
    head = arrived = 0
    playing, underruns, deadline = False, 0, None
    for k in range(n_ticks):
        now = phase + k * c.TICK_S
        while arrived < len(arrivals) and arrivals[arrived] <= now + 1e-12:
            arrived += 1
        queued = arrived - head
        if playing:
            if queued > 0:
                play[head] = k
                head += 1
                underruns = 0
            else:
                underruns += 1
                if underruns > 2:
                    playing, underruns = False, 0
        elif queued >= 3 or (queued > 0 and deadline is not None and now >= deadline - 1e-12):
            playing = True
            starts.append(k)
            play[head] = k
            head += 1
        elif queued == 0:
            deadline = now + 3 * c.TICK_S
    return play, starts


# -------------------------------------------------------------------------------------- the build


def frame_activity(x: np.ndarray, frame: int, thr_dbov: float, hang: int) -> np.ndarray:
    n = len(x) // frame
    p = np.mean((np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame) / 32768.0) ** 2, axis=1)
    act = 10 * np.log10(p + 1e-20) >= thr_dbov
    if hang > 0:
        idx = np.flatnonzero(act)
        out = act.copy()
        for h in range(1, hang + 1):
            j = idx + h
            out[j[j < n]] = True
        act = out
    return act


def dilate(mask: np.ndarray, before: int, after: int) -> np.ndarray:
    out = mask.copy()
    idx = np.flatnonzero(mask)
    n = len(mask)
    for h in range(1, max(before, after) + 1):
        if h <= after:
            j = idx + h
            out[j[j < n]] = True
        if h <= before:
            j = idx - h
            out[j[j >= 0]] = True
    return out


def uplink(sum8: np.ndarray, law: str) -> np.ndarray:
    """Phone side 8 kHz -> G.711 -> AICC per-frame linear x3 -> soxr 24 -> 16 kHz -> int16."""
    up24 = c.aicc_upsample(c.g711_roundtrip(c.to_i16(sum8), law))
    return c.to_i16(c.soxr_resample(up24, c.SR_TTS, c.SR))


def build(cond: Cond, silero: c.Silero | None = None) -> tuple[dict, dict]:
    """Return (arrays, meta) for one call."""
    rng = np.random.default_rng(cond.seed_int)
    lay = layout(cond, rng)
    dur_s = lay["duration"]
    n8 = int(round(dur_s * c.SR_TEL))
    n8 -= n8 % c.TEL_FRAME
    n16 = 2 * n8
    n_ticks = n8 // c.TEL_FRAME
    d8 = int(round(cond.d_ms * c.SR_TEL / 1000))

    # ---- send schedule, global frame list
    frame_resp, frame_idx, send = [], [], []
    resp_meta = []
    last_send = -1.0
    for r, (uid, req_t) in enumerate(lay["far"]):
        m = far_material(uid)
        nf = math.ceil(len(m["x8"]) / c.TEL_FRAME)
        # one response at a time: with RTF > 1 the next one waits until the previous is fully sent.
        # TTFC jitter (U 0-25 ms) gives each response a random send phase against AICC's 20 ms ticks.
        req_eff = max(req_t, last_send + 0.05) + float(rng.uniform(0.0, 0.025))
        st = send_times(req_eff, nf, cond.rtf, None)
        st = np.maximum(st, st[0] + (np.arange(nf) + 1) * c.TICK_S - MAX_LEAD_S)  # max_lead_ms, per response
        last_send = float(st[-1])
        frame_resp += [r] * nf
        frame_idx += list(range(nf))
        send.append(st)
        resp_meta.append({"utt": str(uid), "req_t": round(req_eff, 3), "frames": nf, "gain_db": round(m["gain_db"], 2)})
    send_t = np.concatenate(send)
    frame_resp = np.asarray(frame_resp)
    frame_idx = np.asarray(frame_idx)
    order = np.argsort(send_t, kind="stable")  # FIFO by send time (responses never overlap in practice)
    assert np.all(order == np.arange(len(order))), "responses overlap in the send log"

    # ---- truth: transit per chunk, then AICC's state machine at a random tick phase
    chunk_key = frame_resp * 10000 + frame_idx // FRAMES_PER_CHUNK
    _, inv = np.unique(chunk_key, return_inverse=True)
    jit = TRANSIT_S + rng.exponential(TRANSIT_JIT_S, size=inv.max() + 1)
    arrival = np.maximum.accumulate(send_t + jit[inv])
    phase8 = int(rng.integers(0, c.TEL_FRAME))  # tick phase in 8 kHz samples (0-20 ms)
    phase = phase8 / c.SR_TEL
    play_true, starts_true = aicc_playout(arrival, phase, n_ticks)
    # ---- the gateway's model: transit 0, tick phase 0, plus eps ("blind": A3's recipe as written)
    play_model, starts_model = aicc_playout(send_t, 0.0, n_ticks)
    # ---- DEVIATION (forced by the data): a "locked" model that has learned AICC's tick phase and the mean
    # transit. The blind model mispredicts each playback start by one 20 ms tick about half the time.
    play_locked, _ = aicc_playout(send_t + TRANSIT_S + TRANSIT_JIT_S, phase, n_ticks)

    # ---- downlink: frames on the true ticks, G.711 round trip
    dl8 = np.zeros(n8, dtype=np.float64)
    for g in range(len(send_t)):
        k = play_true[g]
        if k < 0:
            continue
        a = phase8 + k * c.TEL_FRAME
        if a >= n8:
            continue
        x8 = far_material(lay["far"][frame_resp[g]][0])["x8"]
        seg = x8[frame_idx[g] * c.TEL_FRAME:(frame_idx[g] + 1) * c.TEL_FRAME]
        b = min(a + len(seg), n8)
        dl8[a:b] = seg[: b - a]
    dl8 = c.g711_roundtrip(c.to_i16(dl8), cond.law).astype(np.float64)

    # ---- echo, with the path change
    models = c.load_g168()
    names = sorted(models)
    t_change = lay["t_change"]
    if cond.path == "open":
        echo8 = np.zeros(n8)
        erl_eff = erl2_eff = None
        path2 = None
    else:
        g1, erl_eff = c.echo_path(models[cond.path], cond.erl)
        e1 = np.convolve(dl8, g1)[:n8]
        echo8 = np.concatenate([np.zeros(d8), e1[: n8 - d8]])
        path2, erl2_eff = None, None
        if t_change is not None:
            path2 = cond.path2 or names[(names.index(cond.path) + 4) % len(names)]
            g2, erl2_eff = c.echo_path(models[path2], cond.erl)
            e2 = np.convolve(dl8, g2)[:n8]
            e2 = np.concatenate([np.zeros(d8), e2[: n8 - d8]])
            nc = int(round(t_change * c.SR_TEL))
            echo8[nc:] = e2[nc:]

    # ---- near end and noise
    near_dbm0 = FAR_DBM0 + cond.near_rel_db
    near8 = np.zeros(n8)
    near_meta = []
    for uid, t0 in lay["near"]:
        x = near_material(uid, near_dbm0)
        a = int(round(t0 * c.SR_TEL))
        b = min(a + len(x), n8)
        near8[a:b] += x[: b - a]
        near_meta.append({"utt": uid, "start": t0, "end": round(b / c.SR_TEL, 4), "text_norm": rows()[uid]["text_norm"],
                          "set": rows()[uid]["set"], "has_number": rows()[uid]["has_number"]})
    noise8 = c.band_noise(n8, c.SR_TEL, cond.noise_dbm0, rng)

    capture16 = uplink(echo8 + near8 + noise8, cond.law)
    clean16 = uplink(near8 + noise8, cond.law)

    # ---- references at 16 kHz
    lead16 = int(round(LEAD_S * c.SR))
    ref_model = np.zeros(n16, dtype=np.float64)
    ref_locked = np.zeros(n16, dtype=np.float64)
    ref_oracle = np.zeros(n16, dtype=np.float64)
    margin16 = int(round(MARGIN_S * c.SR))
    true_minus_model = []
    locked_err = []
    for g in range(len(send_t)):
        x16 = far_material(lay["far"][frame_resp[g]][0])["x16"]
        seg = x16[frame_idx[g] * 320:(frame_idx[g] + 1) * 320]
        for k_m, ph16, dst in ((play_model[g], 0, ref_model), (play_locked[g], 2 * phase8, ref_locked)):
            if k_m < 0:
                continue
            t_model = k_m * c.TICK_S + ph16 / c.SR
            eps = cond.eps_ms if (cond.eps_step_t is None or t_model < cond.eps_step_t) else cond.eps2_ms
            a = k_m * 320 + ph16 + int(round(eps * c.SR / 1000)) - lead16
            if 0 <= a < n16:
                b = min(a + len(seg), n16)
                dst[a:b] = seg[: b - a]
        km = play_model[g]
        kt = play_true[g]
        if kt >= 0 and play_locked[g] >= 0:
            locked_err.append((kt - play_locked[g]) * c.TICK_S)
        if kt >= 0:
            a = 2 * (phase8 + kt * c.TEL_FRAME + d8) - margin16
            if 0 <= a < n16:
                b = min(a + len(seg), n16)
                ref_oracle[a:b] = seg[: b - a]
        if km >= 0 and kt >= 0:
            true_minus_model.append((phase + kt * c.TICK_S) - km * c.TICK_S)

    # ---- ground truth masks (10 ms frames at 16 kHz = 80-sample frames at 8 kHz)
    fa_play = frame_activity(dl8, 80, c.dbm0_to_dbov(FAR_DBM0) - 30, hang=5)
    far_active = np.zeros(n8 // 80, dtype=bool)
    sh = d8 // 80
    far_active[sh:] = fa_play[: len(far_active) - sh]
    far_active = dilate(far_active, 0, 2)  # path dispersion <= 16 ms
    near_active = frame_activity(near8, 80, c.dbm0_to_dbov(near_dbm0) - 30, hang=5)
    near_present = dilate(near_active, 20, 20)  # 200 ms guard on both sides
    eo = far_active & ~near_present

    onsets = [round(phase + k * c.TICK_S + cond.d_ms / 1000, 4) for k in starts_true]
    resp_true_start, resp_model_start = [], []
    for r in range(len(lay["far"])):
        sel = np.flatnonzero(frame_resp == r)
        kt = play_true[sel]
        km = play_model[sel]
        resp_meta[r]["ticks_true_blind_locked"] = [int(kt[0]), int(km[0]), int(play_locked[sel][0])]
        resp_meta[r]["first_arrival"] = round(float(arrival[sel][0]), 5)
        resp_meta[r]["first_send"] = round(float(send_t[sel][0]), 5)
        resp_true_start.append(round(phase + kt[kt >= 0].min() * c.TICK_S, 4) if np.any(kt >= 0) else None)
        resp_model_start.append(round(km[km >= 0].min() * c.TICK_S, 4) if np.any(km >= 0) else None)
    for r, m in enumerate(resp_meta):
        m["true_start"] = resp_true_start[r]
        m["model_start"] = resp_model_start[r]
    tmm = np.asarray(true_minus_model) if true_minus_model else np.zeros(1)

    arrays = {
        "capture": capture16,
        "ref_model": c.to_i16(ref_model),
        "ref_locked": c.to_i16(ref_locked),
        "ref_oracle": c.to_i16(ref_oracle),
        "clean": clean16,
        "far_active": far_active,
        "near_active": near_active,
        "near_present": near_present,
        "eo": eo,
    }
    if silero is not None:
        arrays["silero_before"] = silero.probs(capture16)
    meta = {
        "cond": asdict(cond),
        "duration_s": n16 / c.SR,
        "layout": {k: v for k, v in lay.items() if k not in ("far", "near")},
        "responses": resp_meta,
        "near": near_meta,
        "path2": path2,
        "erl_eff": erl_eff,
        "erl2_eff": erl2_eff,
        "t_change": t_change,
        "tick_phase_ms": round(phase * 1000, 3),
        "playout_onsets": onsets,
        # playout runs that start in the middle of a response = AICC re-buffered after an underrun
        "underrun_restarts_true": int(sum(1 for k in starts_true if frame_idx[np.flatnonzero(play_true == k)[0]] > 0)),
        "underrun_restarts_model": int(sum(1 for k in starts_model if frame_idx[np.flatnonzero(play_model == k)[0]] > 0)),
        "silence_ticks_inside_responses_true": int(sum(
            int(play_true[sel].max() - play_true[sel].min() + 1 - len(sel))
            for sel in (np.flatnonzero((frame_resp == r) & (play_true >= 0)) for r in range(len(lay["far"])))
            if len(sel) > 1)),
        "true_minus_model_ms": {"p50": round(float(np.median(tmm)) * 1000, 2), "min": round(float(tmm.min()) * 1000, 2),
                                "max": round(float(tmm.max()) * 1000, 2)},
        "residual_delay_ms_nominal": round(cond.d_ms + LEAD_S * 1000 + float(np.median(tmm)) * 1000 - cond.eps_ms, 2),
        "true_minus_model_by_response_ms": [None if (m["true_start"] is None or m["model_start"] is None)
                                            else round((m["true_start"] - m["model_start"]) * 1000, 2) for m in resp_meta],
        "locked_model_tick_errors": int(np.count_nonzero(np.abs(np.asarray(locked_err)) > 1e-9)) if locked_err else 0,
        "locked_model_frames": len(locked_err),
        "frames_unplayed": int(np.sum(play_true < 0)),
        "near_dbm0": near_dbm0,
    }
    return arrays, meta


def cache_path(cond: Cond):
    return c.CACHE / cond.part / f"{cond.cid}.npz"


def generate(cond: Cond, silero: c.Silero | None = None, force: bool = False):
    p = cache_path(cond)
    if p.exists() and not force:
        return p
    arrays, meta = build(cond, silero)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    tmp.rename(p)
    return p


def load(cond_or_path) -> tuple[dict, dict]:
    p = cond_or_path if not isinstance(cond_or_path, Cond) else cache_path(cond_or_path)
    z = np.load(p)
    meta = json.loads(bytes(z["meta"]).decode())
    return {k: z[k] for k in z.files if k != "meta"}, meta

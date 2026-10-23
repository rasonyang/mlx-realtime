"""G7 dual path: simulated telephone calls with double talk, as M0's S11 built them, plus the send
log the gateway's echo path needs. Throwaway spike code.

The chain is B4's (spikes/aec/s11_chain.py, A3's recipe): the far end is manifest speech at
24 kHz sent with S3's TTS cadence, played by AICC's playout state machine (20 ms ticks at a random
phase, 3-frame prebuffer, re-buffer after 2 empty ticks), AICC's 33-tap downsampler and G.711, an
ITU-T G.168 echo path (read from the git-ignored spikes/data/echo/g168_paths.json; nothing
ITU-derived is written), the near end and band noise, G.711 back and AICC's per-frame linear
upsampler. Unlike B4 it stops at AICC's 24 kHz uplink: the gateway's own InputDecoder makes the
16 kHz capture.

Two layouts, each 32 calls (40 zh + 40 en FLEURS near-end utterances, ERL 6/15 dB, near end 0 or
-15 dB re the far end, as B4's `dt` grid):

- ``dt``: B4's continuous double talk: the far end talks throughout (responses back to back) and
  each near-end utterance lies entirely under it. Same seeds and RNG order as B4, so the capture
  is B4's (checked: soxr of this uplink equals B4's cached capture).
- ``bargein``: barge-in realistic: the far end stops ``BARGE_AFTER_S`` (0.5 s) after each near-end
  onset, as the gateway's barge-in does (the response is cancelled, AICC flushes what it has not
  played and truncates the item to what it played); the next answer is requested
  ``REPLY_AFTER_S`` after the utterance ends. Utterances are ``GAP_S`` apart, so each onset falls
  past the 2 s echo guard of the answer playing.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aec"))

import s11_common as c  # noqa: E402

import numpy as np  # noqa: E402

import s11_chain as ch  # noqa: E402
import s11_run as run  # noqa: E402

BARGE_AFTER_S = 0.5
REPLY_AFTER_S = 1.0
GAP_S = 3.5
SAMPLES24_PER_FRAME = 480  # one 20 ms frame of the gateway's 24 kHz output
CACHE = c.DATA / "g7dual"


@dataclass
class Delta:
    t: float  # when the gateway hands it to the socket (s, the call's clock)
    response: int
    a24: int  # [a24, b24) of the response's 24 kHz audio
    b24: int


@dataclass
class Cut:
    t: float  # the barge-in: the gateway cancels and flushes; AICC flushes its queue
    response: int
    played_frames: int  # what AICC really played of the item (its truncate)
    complete: bool  # every frame was sent before the cut: the response was already done


@dataclass
class Call:
    cid: str
    layout: str
    cond: ch.Cond
    up24: np.ndarray  # AICC's uplink at 24 kHz, int16: what the gateway receives
    up24_clean: np.ndarray  # the same chain without echo (the ASR reference)
    deltas: list[Delta]
    cuts: list[Cut]
    responses: list[str]  # far-end utterance per response
    near: list[dict]  # near-end utterances: start/end (s), text_norm, set, has_number
    meta: dict


def conds() -> list[ch.Cond]:
    return run.grid("dt")


def _bargein_layout(cond: ch.Cond, rng: np.random.Generator) -> tuple[dict, list[float]]:
    """Near-end utterances GAP_S apart from 10 s; the far end back to back until each onset,
    cut BARGE_AFTER_S after it, then the next answer REPLY_AFTER_S after the utterance ends."""
    ids = ch.lang_ids(cond.lang)
    near_ids = list(cond.near_ids)
    far_pool = [str(u) for u in rng.permutation([u for u in ids if u not in near_ids])]
    fi = 0

    def next_far() -> str:
        nonlocal fi
        u = far_pool[fi % len(far_pool)]
        fi += 1
        return u

    def dur(u: str) -> float:
        return ch.far_material(u)["dur"]

    far, near, cut_times = [], [], []
    t_near = ch.DT_START
    t = 0.5
    for u in near_ids:
        while t < t_near:
            f = next_far()
            far.append((f, round(t, 3)))
            t += ch.TTFC_S + dur(f) + 0.2
        near.append((u, round(t_near, 3)))
        cut_times.append(round(t_near + BARGE_AFTER_S, 3))
        end = t_near + ch.rows()[u]["duration_s"]
        t = end + REPLY_AFTER_S
        t_near = end + GAP_S
    f = next_far()  # one more answer after the last utterance
    far.append((f, round(t, 3)))
    total = math.ceil(max(t + ch.TTFC_S + dur(f) + 1.0, t_near) + 0.5)
    return {"far": far, "near": near, "duration": float(total), "t_change": None,
            "phases": {"eo_until": ch.DT_START}}, cut_times


def playout(arrivals: np.ndarray, responses: np.ndarray, phase: float, n_ticks: int,
            flushes: list[tuple[float, int]]) -> np.ndarray:
    """AICC's RTPSession.Run (s11_chain.aicc_playout) with barge-in flushes: at a flush, the
    queued frames of that response are dropped, and later ones are dropped on arrival."""
    play = np.full(len(arrivals), -1, dtype=np.int64)
    flushed: set[int] = set()
    queue: list[int] = []
    arrived = 0
    playing, underruns, deadline = False, 0, None
    fi = 0
    for k in range(n_ticks):
        now = phase + k * c.TICK_S
        while fi < len(flushes) and flushes[fi][0] <= now + 1e-12:
            flushed.add(flushes[fi][1])
            queue = [g for g in queue if responses[g] not in flushed]
            fi += 1
        while arrived < len(arrivals) and arrivals[arrived] <= now + 1e-12:
            if responses[arrived] not in flushed:
                queue.append(arrived)
            arrived += 1
        queued = len(queue)
        if playing:
            if queued > 0:
                play[queue.pop(0)] = k
                underruns = 0
            else:
                underruns += 1
                if underruns > 2:
                    playing, underruns = False, 0
        elif queued >= 3 or (queued > 0 and deadline is not None and now >= deadline - 1e-12):
            playing = True
            play[queue.pop(0)] = k
        elif queued == 0:
            deadline = now + 3 * c.TICK_S
    return play


def build(cond: ch.Cond, layout: str) -> Call:
    rng = np.random.default_rng(cond.seed_int)
    if layout == "dt":
        lay = ch.layout(cond, rng)
        cut_times: list[float] = []
    else:
        lay, cut_times = _bargein_layout(cond, rng)
    dur_s = lay["duration"]
    n8 = int(round(dur_s * c.SR_TEL))
    n8 -= n8 % c.TEL_FRAME
    n_ticks = n8 // c.TEL_FRAME
    d8 = int(round(cond.d_ms * c.SR_TEL / 1000))

    # ---- send schedule (s11_chain.build), then the barge-in cuts on the gateway's side
    frame_resp, frame_idx, send = [], [], []
    last_send = -1.0
    for r, (uid, req_t) in enumerate(lay["far"]):
        m = ch.far_material(uid)
        nf = math.ceil(len(m["x8"]) / c.TEL_FRAME)
        req_eff = max(req_t, last_send + 0.05) + float(rng.uniform(0.0, 0.025))
        st = ch.send_times(req_eff, nf, cond.rtf, None)
        st = np.maximum(st, st[0] + (np.arange(nf) + 1) * c.TICK_S - ch.MAX_LEAD_S)
        last_send = float(st[-1])
        frame_resp += [r] * nf
        frame_idx += list(range(nf))
        send.append(st)
    send_t = np.concatenate(send)
    frame_resp = np.asarray(frame_resp)
    frame_idx = np.asarray(frame_idx)
    assert np.all(np.diff(send_t) >= 0), "responses overlap in the send log"

    # Which response each cut interrupts: the one whose frames are being sent or played then
    # (the latest response requested before the cut).
    cut_resp = []
    for tc in cut_times:
        rs = [r for r, (_, req_t) in enumerate(lay["far"]) if req_t <= tc]
        cut_resp.append(rs[-1] if rs else None)
    sent = np.ones(len(send_t), dtype=bool)
    for tc, r in zip(cut_times, cut_resp):
        if r is not None:
            sent &= ~((frame_resp == r) & (send_t > tc))  # the pipeline is cancelled: never sent

    # ---- truth: transit per chunk, AICC's state machine at a random tick phase, with flushes
    chunk_key = frame_resp * 10000 + frame_idx // ch.FRAMES_PER_CHUNK
    _, inv = np.unique(chunk_key, return_inverse=True)
    jit = ch.TRANSIT_S + rng.exponential(ch.TRANSIT_JIT_S, size=inv.max() + 1)
    arrival = np.maximum.accumulate(send_t + jit[inv])
    phase8 = int(rng.integers(0, c.TEL_FRAME))
    phase = phase8 / c.SR_TEL
    keep = np.flatnonzero(sent)
    flushes = sorted((tc + ch.TRANSIT_S, r) for tc, r in zip(cut_times, cut_resp) if r is not None)
    play_kept = playout(arrival[keep], frame_resp[keep], phase, n_ticks, flushes)
    play_true = np.full(len(send_t), -1, dtype=np.int64)
    play_true[keep] = play_kept

    # ---- downlink, echo, near end, noise, uplink (s11_chain.build)
    dl8 = np.zeros(n8, dtype=np.float64)
    for g in np.flatnonzero(play_true >= 0):
        a = phase8 + play_true[g] * c.TEL_FRAME
        if a >= n8:
            continue
        x8 = ch.far_material(lay["far"][frame_resp[g]][0])["x8"]
        seg = x8[frame_idx[g] * c.TEL_FRAME:(frame_idx[g] + 1) * c.TEL_FRAME]
        b = min(a + len(seg), n8)
        dl8[a:b] = seg[: b - a]
    dl8 = c.g711_roundtrip(c.to_i16(dl8), cond.law).astype(np.float64)
    models = c.load_g168()
    g1, erl_eff = c.echo_path(models[cond.path], cond.erl)
    e1 = np.convolve(dl8, g1)[:n8]
    echo8 = np.concatenate([np.zeros(d8), e1[: n8 - d8]])
    near_dbm0 = ch.FAR_DBM0 + cond.near_rel_db
    near8 = np.zeros(n8)
    near_meta = []
    for uid, t0 in lay["near"]:
        x = ch.near_material(uid, near_dbm0)
        a = int(round(t0 * c.SR_TEL))
        b = min(a + len(x), n8)
        near8[a:b] += x[: b - a]
        row = ch.rows()[uid]
        near_meta.append({"utt": uid, "start": t0, "end": round(b / c.SR_TEL, 4), "text_norm": row["text_norm"],
                          "set": row["set"], "has_number": row["has_number"]})
    noise8 = c.band_noise(n8, c.SR_TEL, cond.noise_dbm0, rng)

    def uplink24(sum8: np.ndarray) -> np.ndarray:
        return c.aicc_upsample(c.g711_roundtrip(c.to_i16(sum8), cond.law))

    up24 = uplink24(echo8 + near8 + noise8)
    up24_clean = uplink24(near8 + noise8)

    # ---- the gateway's deltas: the frames it sent, at most 100 ms (5 frames) per delta
    deltas: list[Delta] = []
    for r, (uid, _) in enumerate(lay["far"]):
        x24 = ch.far_material(uid)["x24"]
        idx = np.flatnonzero((frame_resp == r) & sent)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and j + 1 - i < 5 and send_t[idx[j + 1]] == send_t[idx[i]]:
                j += 1
            f0, f1 = int(frame_idx[idx[i]]), int(frame_idx[idx[j]]) + 1
            a24, b24 = f0 * SAMPLES24_PER_FRAME, min(f1 * SAMPLES24_PER_FRAME, len(x24))
            if b24 > a24:
                deltas.append(Delta(float(send_t[idx[i]]), r, a24, b24))
            i = j + 1
    cuts = []
    for tc, r in zip(cut_times, cut_resp):
        if r is None:
            continue
        played = int(np.count_nonzero((frame_resp == r) & (play_true >= 0)))
        complete = not np.any((frame_resp == r) & (send_t > tc))
        cuts.append(Cut(tc, r, played, complete))
    meta = {"layout": layout, "duration_s": n8 / c.SR_TEL, "erl_eff": erl_eff, "tick_phase_ms": round(phase * 1000, 3),
            "near_dbm0": near_dbm0, "responses": len(lay["far"]), "cuts": len(cuts),
            "frames_sent": int(sent.sum()), "frames_played": int(np.count_nonzero(play_true >= 0))}
    return Call(cid=f"{cond.cid}__{layout}", layout=layout, cond=cond, up24=up24, up24_clean=up24_clean,
                deltas=deltas, cuts=cuts, responses=[u for u, _ in lay["far"]], near=near_meta, meta=meta)

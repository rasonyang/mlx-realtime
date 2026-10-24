"""Turn the synthesized and human audio into the 16 kHz clips the ASR hears, one per manifest row.

    uv run --locked --project spikes/asr_context python spikes/asr_context/degrade.py [--only u001,..]

Output: data/clips/<id>.wav (16 kHz mono PCM16, what the gateway would hand the ASR) and
data/clips/meta.json (per-clip parameters). Everything is deterministic from the row id.

Degradations (all but ``clean`` go through the exact AICC -> gateway path):

``clean``  wideband: TTS 24 kHz -> the gateway's InputDecoder (soxr HQ 24 -> 16 kHz, as for a
           PCM session), levelled to -26 dBov (P.56 active level), 300 ms of silence either side.
``g711``   the caller's speech as the phone network delivers it: soxr HQ to 8 kHz, level -20 dBm0,
           G.711 mu-law encode/decode (stdlib audioop), AICC's per-20 ms-frame linear 8 -> 24 kHz
           upsampler (ported from AICC, spikes/aec/s11_common.aicc_upsample), then the gateway's
           InputDecoder 24 -> 16 kHz in 20 ms appends. Same silence padding.
``noise``  ``g711`` with real background noise mixed in at 8 kHz before the codec, SNR = P.56
           active speech level minus noise RMS, 10, 15 or 20 dB (by index). Two sources, alternating:
           ``babble`` six other real speakers (FLEURS zh/en + ASCEND recordings, never the target)
           summed; ``room`` the quietest 50 ms frames of the ASCEND recordings (real room noise
           floors), concatenated with 20 ms crossfades. No dedicated noise corpus is in the
           datasets (spikes/datasets), so these are the real-noise sources available there.
``onset``  ``g711`` with the first 60, 100 or 140 ms (by index) after the detected speech onset
           removed: the clip starts inside the first syllable, as when VAD fires late; no leading
           silence.
``echo``   the caller starts 0.6-1.4 s before the bot's own playback ends. Far end: a 11-14 s bot
           answer played by the model of AICC's playout (20 ms ticks), AICC downsampler, G.711,
           an ITU-T G.168 echo path (ERL 6 or 15 dB, 120 ms delay), the caller (``g711`` signal,
           0 dB re the far end), -65 dBm0 band noise, G.711, AICC upsampler: spikes/g7dual's
           chain. The 24 kHz uplink goes through the gateway's own echo stage
           (g7_gateway.run_call, aec.asr_input from config.yaml = speex, AEC3 + Speex +
           GCC-PHAT estimator) and the clip is the ASR track around the utterance +-300 ms.
``human``  ASCEND rows: ``clean`` = the untouched 16 kHz copy; ``g711`` = the same recording through
           the g711 chain (16 kHz -> 8 kHz ...).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import common as c

sys.path[:0] = [str(c.REPO / "spikes" / "g7dual"), str(c.REPO / "spikes" / "aec")]
import s11_common as sc  # noqa: E402  (caps BLAS threads; before numpy-heavy work)

import numpy as np  # noqa: E402

import g7_chain as gc  # noqa: E402
import g7_gateway as gg  # noqa: E402
import s11_chain as ch  # noqa: E402
from mlx_realtime.audio.codec import InputDecoder  # noqa: E402
from mlx_realtime.audio.formats import PCM  # noqa: E402
from mlx_realtime.config import AecConfig  # noqa: E402

sc.G168_A2 = c.shared_data() / "echo" / "g168_paths.json"  # git-ignored, (c) ITU
CLIPS = c.DATA / "clips"
PAD_S = 0.3
TARGET_DBOV = -26.0
SPEECH_DBM0 = -20.0
APPEND = 480


def rng_for(rid: str, salt: str = "") -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{rid}|{salt}".encode()) & 0x7FFFFFFF)


def index_of(rid: str) -> int:
    return int("".join(ch_ for ch_ in rid if ch_.isdigit()) or 0)


# ------------------------------------------------------------------------------ building blocks


def p56(x: np.ndarray, fs: int) -> float:
    return sc.p56_active_level(np.asarray(x, dtype=np.float64) / 32768.0, fs)[0]


def gain_to(x: np.ndarray, fs: int, dbov: float) -> np.ndarray:
    return x * 10 ** ((dbov - p56(x, fs)) / 20)


def decode16(x24: np.ndarray) -> np.ndarray:
    """The gateway's InputDecoder for a PCM session: 24 -> 16 kHz in 20 ms appends."""
    return gg.decode(sc.to_i16(x24))


def pad(x: np.ndarray, fs: int, head: bool = True) -> np.ndarray:
    n = int(PAD_S * fs)
    return np.concatenate([np.zeros(n if head else 0), x, np.zeros(n)])


def aicc_uplink24(x8: np.ndarray) -> np.ndarray:
    """mu-law round trip at 8 kHz, then AICC's per-frame upsampler to 24 kHz."""
    return sc.aicc_upsample(sc.g711_roundtrip(sc.to_i16(x8), "ulaw")).astype(np.float64)


def to_8k(x: np.ndarray, fs: int) -> np.ndarray:
    return sc.soxr_resample(x, fs, 8000)


def g711_chain(x: np.ndarray, fs: int, *, noise8: np.ndarray | None = None, snr_db: float | None = None) -> np.ndarray:
    """Speech at `fs` (int16 range) -> the 16 kHz clip the gateway's ASR path gets."""
    x8 = to_8k(x, fs)
    n = len(x8) - len(x8) % sc.TEL_FRAME
    x8 = x8[:n]
    x8 = x8 * 10 ** ((sc.dbm0_to_dbov(SPEECH_DBM0) - p56(x8, 8000)) / 20)
    if noise8 is not None:
        noise = np.resize(noise8, len(x8)).astype(np.float64)
        speech_level = p56(x8, 8000)
        noise *= 10 ** ((speech_level - snr_db) / 20) * 32768.0 / (np.sqrt(np.mean(noise ** 2)) + 1e-9)
        x8 = x8 + noise
    return decode16(aicc_uplink24(x8))


# --------------------------------------------------------------------------------------- noise


def _shared(path: str) -> np.ndarray:
    x, sr = c.read_wav(c.shared_data() / path)
    assert sr == 16000
    return x.astype(np.float64)


class NoiseBank:
    """Real noise from the datasets in spikes/data (see the module docstring)."""

    def __init__(self) -> None:
        man = [json.loads(line) for line in (c.REPO / "spikes" / "datasets" / "manifest.jsonl").read_text().splitlines()]
        self.speech = {r["id"]: r for r in man}
        self.babble_pool = [r["id"] for r in man if r["set"] in ("zh", "en", "mix")]
        self._room: np.ndarray | None = None

    def babble(self, rid: str) -> np.ndarray:
        rng = rng_for(rid, "babble")
        parts = []
        for k in rng.choice(len(self.babble_pool), size=6, replace=False):
            x = _shared(self.speech[self.babble_pool[int(k)]]["path_16k"])
            x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
            n = 16000 * 12
            parts.append(np.resize(np.roll(x, int(rng.integers(len(x)))), n))
        return to_8k(np.sum(parts, axis=0) * 3000, 16000)

    def room(self, rid: str) -> np.ndarray:
        if self._room is None:
            frames = []
            raw = c.shared_data() / "raw" / "ascend" / "test"
            for p in sorted(raw.glob("*.wav")):
                x, sr = c.read_wav(p)
                if sr != 16000:
                    continue
                x = x.astype(np.float64)
                w = 800
                f = x[: len(x) // w * w].reshape(-1, w)
                e = np.sqrt((f ** 2).mean(axis=1))
                for k in np.argsort(e)[: max(2, len(e) // 8)]:
                    if 20 * np.log10(e[k] / 32768 + 1e-9) < -45:
                        frames.append(f[k])
            fade = np.linspace(0, 1, 320)
            out = frames[0].copy()
            for fr in frames[1:]:
                fr = fr - fr.mean()
                out = np.concatenate([out[:-320], out[-320:] * fade[::-1] + fr[:320] * fade, fr[320:]])
            self._room = out - out.mean()
        x = np.roll(self._room, int(rng_for(rid, "room").integers(len(self._room))))
        return to_8k(x, 16000)


# ------------------------------------------------------------------------------------------ echo


def load_aec() -> AecConfig:
    cfg = yaml.safe_load((c.REPO / "config.yaml").read_text())["aec"]
    return AecConfig.model_validate(cfg)


def build_echo_call(rid: str, near24: np.ndarray, bot_i: int, path_name: str, erl: float) -> tuple[gc.Call, float, float]:
    """One call: a bot answer, then the caller's clip over its tail. Returns the call, the caller's
    onset (s) and the clip's duration (s). The send/playout/echo model is g7_chain.build's."""
    rng = rng_for(rid, "echo")
    uid = f"bot{bot_i}"
    bot, sr = c.read_wav(c.DATA / "bot" / f"{uid}.wav")
    x24 = bot.astype(np.float64)
    lvl, _ = sc.p56_active_level(sc.aicc_downsample(sc.to_i16(x24)).astype(np.float64) / 32768.0, 8000)
    x24 = sc.to_i16(x24 * 10 ** ((sc.dbm0_to_dbov(ch.FAR_DBM0) - lvl) / 20))
    ch._FAR_CACHE[uid] = {"x24": x24, "x8": sc.aicc_downsample(x24), "dur": len(x24) / 24000}
    x8_far = ch._FAR_CACHE[uid]["x8"]

    nf = math.ceil(len(x8_far) / sc.TEL_FRAME)
    req_t = 0.5 + float(rng.uniform(0, 0.025))
    send_t = ch.send_times(req_t, nf, ch.RTF, None)
    send_t = np.maximum(send_t, send_t[0] + (np.arange(nf) + 1) * sc.TICK_S - ch.MAX_LEAD_S)
    chunk = np.arange(nf) // ch.FRAMES_PER_CHUNK
    jit = ch.TRANSIT_S + rng.exponential(ch.TRANSIT_JIT_S, size=chunk.max() + 1)
    arrival = np.maximum.accumulate(send_t + jit[chunk])
    phase8 = int(rng.integers(0, sc.TEL_FRAME))
    phase = phase8 / sc.SR_TEL
    n_ticks_max = int((send_t[-1] + 40) / sc.TICK_S)
    play, _ = ch.aicc_playout(arrival, phase, n_ticks_max)
    played = np.flatnonzero(play >= 0)
    end_s = phase + (play[played[-1]] + 1) * sc.TICK_S  # the last frame of the bot answer is played
    near8 = to_8k(near24, 24000)
    near8 = near8 * 10 ** ((sc.dbm0_to_dbov(ch.FAR_DBM0) - p56(near8, 8000)) / 20)
    near_dur = len(near8) / 8000
    t_near = end_s - float(rng.uniform(0.6, 1.4))
    dur_s = math.ceil(max(end_s, t_near + near_dur) + 1.0)
    n8 = int(dur_s * sc.SR_TEL)
    n8 -= n8 % sc.TEL_FRAME

    dl8 = np.zeros(n8)
    for g in played:
        a = phase8 + play[g] * sc.TEL_FRAME
        if a >= n8:
            continue
        seg = x8_far[g * sc.TEL_FRAME:(g + 1) * sc.TEL_FRAME]
        b = min(a + len(seg), n8)
        dl8[a:b] = seg[: b - a]
    dl8 = sc.g711_roundtrip(sc.to_i16(dl8), "ulaw").astype(np.float64)
    model = sc.load_g168()[path_name]
    g1, erl_eff = sc.echo_path(model, erl)
    e1 = np.convolve(dl8, g1)[:n8]
    d8 = int(round(0.120 * sc.SR_TEL))
    echo8 = np.concatenate([np.zeros(d8), e1[: n8 - d8]])
    near_full = np.zeros(n8)
    a = int(round(t_near * 8000))
    b = min(a + len(near8), n8)
    near_full[a:b] += near8[: b - a]
    noise8 = sc.band_noise(n8, 8000, -65.0, rng)

    def up(sum8: np.ndarray) -> np.ndarray:
        return sc.aicc_upsample(sc.g711_roundtrip(sc.to_i16(sum8), "ulaw"))

    deltas = []
    for i0 in range(0, nf, ch.FRAMES_PER_CHUNK):  # a chunk's frames are sent together, <= 5 per delta
        for j0 in range(i0, min(i0 + ch.FRAMES_PER_CHUNK, nf), 5):
            j1 = min(j0 + 5, i0 + ch.FRAMES_PER_CHUNK, nf)
            a24, b24 = j0 * gc.SAMPLES24_PER_FRAME, min(j1 * gc.SAMPLES24_PER_FRAME, len(x24))
            if b24 > a24:
                deltas.append(gc.Delta(float(send_t[j0]), 0, a24, b24))
    cond = ch.Cond(cid=rid, part="asr_context", path=path_name, erl=erl, d_ms=120, lang="zh")
    call = gc.Call(cid=rid, layout="tail", cond=cond, up24=up(echo8 + near_full + noise8),
                   up24_clean=up(near_full + noise8), deltas=deltas, cuts=[], responses=[uid],
                   near=[{"utt": rid, "start": t_near, "end": round(b / 8000, 4), "text_norm": "", "set": "tts",
                          "has_number": False}],
                   meta={"erl_eff": erl_eff, "bot": uid, "bot_end_s": round(end_s, 3), "t_near_s": round(t_near, 3)})
    return call, t_near, near_dur


def echo_clip(rid: str, x24: np.ndarray, aec: AecConfig, executor: ThreadPoolExecutor) -> tuple[np.ndarray, dict]:
    i = index_of(rid)
    path_name = f"m{i % 8 + 1}"
    erl = (6.0, 15.0)[(i // 8) % 2]
    call, t_near, dur = build_echo_call(rid, x24, i % 6, path_name, erl)
    res = asyncio.run(gg.run_call(call, aec, executor))
    a = max(0, int((t_near - PAD_S) * 16000))
    b = min(len(res["asr"]), int((t_near + dur + PAD_S) * 16000))
    meta = {"path": path_name, "erl_db": erl, **call.meta,
            "speex_aligned_at_s": res["summary"]["speex_aligned_at_s"], "speex_erle_db": res["summary"]["speex_erle_db"]}
    return res["asr"][a:b], meta


# ------------------------------------------------------------------------------------------ main


def onset_cut(x24: np.ndarray, rid: str) -> tuple[np.ndarray, int]:
    ms = (60, 100, 140)[index_of(rid) % 3]
    w = 480  # 20 ms
    n = len(x24) // w
    rms = np.sqrt((x24[: n * w].reshape(n, w) ** 2).mean(axis=1))
    first = int(np.argmax(rms > rms.max() * 10 ** (-25 / 20)))
    return x24[first * w + ms * 24:], ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    rows = [r for r in c.load_manifest() if not only or r["id"] in only]
    CLIPS.mkdir(parents=True, exist_ok=True)
    meta_path = CLIPS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    bank = NoiseBank()
    aec = load_aec()
    with ThreadPoolExecutor(1) as executor:
        for n, r in enumerate(rows):
            rid, deg = r["id"], r["degradation"]
            out = CLIPS / f"{rid}.wav"
            if out.exists() and not a.force:
                continue
            info: dict = {"degradation": deg}
            if r["set"] == "ascend":
                x16 = _shared(r["source"]["path_16k"])
                y = x16 if deg == "clean" else g711_chain(pad(x16, 16000), 16000)
                y = sc.to_i16(y)
            else:
                x24, sr = c.read_wav(c.DATA / "tts24k" / f"{rid}.wav")
                x24 = x24.astype(np.float64)
                if deg == "clean":
                    y = decode16(pad(gain_to(x24, 24000, TARGET_DBOV), 24000))
                elif deg == "g711":
                    y = g711_chain(pad(x24, 24000), 24000)
                elif deg == "noise":
                    kind = ("babble", "room")[index_of(rid) // 5 % 2]
                    snr = (10.0, 15.0, 20.0)[index_of(rid) // 10 % 3]
                    noise = bank.babble(rid) if kind == "babble" else bank.room(rid)
                    y = g711_chain(pad(x24, 24000), 24000, noise8=noise, snr_db=snr)
                    info.update(noise=kind, snr_db=snr)
                elif deg == "onset":
                    cut, ms = onset_cut(x24, rid)
                    y = g711_chain(pad(cut, 24000, head=False), 24000)
                    info["onset_cut_ms"] = ms
                elif deg == "echo":
                    y, extra = echo_clip(rid, x24, aec, executor)
                    info.update(extra)
                else:
                    raise ValueError(deg)
            c.write_wav(out, np.asarray(y, dtype=np.int16), 16000)
            info["duration_s"] = round(len(y) / 16000, 2)
            meta[rid] = info
            if n % 20 == 0:
                print(n, rid, deg, info, flush=True)
                meta_path.write_text(json.dumps(meta, indent=1))
    meta_path.write_text(json.dumps(meta, indent=1))
    print("done", len(rows))


if __name__ == "__main__":
    main()

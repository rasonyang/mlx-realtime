"""Shared helpers for S11 part 2, the echo-cancellation simulation (M0, gate G7). Throwaway spike code.

Import this module before numpy in every entry script: it caps the numerical thread pools first
(VECLIB_MAXIMUM_THREADS / OMP_NUM_THREADS / OPENBLAS_NUM_THREADS = 2, as the B4 brief asks).

Contents: paths; environment capture and load averages; G.711 (stdlib audioop, test-data only);
AICC's 24k->8k downsampler and per-frame linear 8k->24k upsampler, ported from
ai-native-callcenter @8fabadc internal/media/resample.go; ITU-T G.168 Annex D echo paths (the
git-ignored JSON from A2, or A3's copy); the P.56 active level (ported from
spikes/datasets/a2_data.py); band-limited noise; the Silero VAD wrapper from spikes/cpu; FIR
helpers; statistics.
"""

from __future__ import annotations

import os

for _k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import wave  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # deprecated in 3.12: used only to build test signals, as A2 did

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
REPO = SPIKES.parent
DATA = SPIKES / "data"
RESULTS = SPIKES / "results"
CACHE = Path(os.environ.get("S11_CACHE", str(DATA / "s11")))  # git-ignored (spikes/data/)
MANIFEST = DATA / "manifest.jsonl"
G168_A2 = DATA / "echo" / "g168_paths.json"  # git-ignored, (c) ITU
G168_A3 = HERE / "g168_echo_paths.json"  # git-ignored by **/g168*.json, (c) ITU
SILERO_ONNX = SPIKES / "cpu" / "models" / "silero_vad.onnx"
SILERO_PY = SPIKES / "cpu" / "silero_vad.py"
SPEEX_LIB = os.environ.get("SPEEXDSP_LIB", "/opt/homebrew/opt/speexdsp/lib/libspeexdsp.1.dylib")

SR = 16000  # the gateway's canonical rate; every canceller runs here
SR_TEL = 8000
SR_TTS = 24000
FRAME = 160  # 10 ms at 16 kHz: the engine frame
TICK_S = 0.020  # AICC playout tick (voice/rtp.go FrameDuration)
TEL_FRAME = 160  # one 20 ms G.711 frame at 8 kHz (media.FrameSamples)
# G.711: a 0 dBm0 sine is 3.17 dB (mu-law) / 3.14 dB (A-law) below the full-scale sine, which is
# -3.01 dBov (0 dBov = full-scale square wave, the P.56/sv56 convention A2 used). So 0 dBm0 is
# about -6.2 dBov, and the G.168 far-end level of -20 dBm0 is about -26.2 dBov.
DBM0_TO_DBOV = -6.2

SILERO_ON, SILERO_OFF = 0.5, 0.35  # the gateway's server_vad threshold and Silero's own hysteresis
SILERO_WIN = 512  # 32 ms at 16 kHz


# ------------------------------------------------------------------------------------- environment


def version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return None


def loadavg() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def speex_available() -> bool:
    return Path(SPEEX_LIB).exists()


def environment() -> dict:
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cores_perf": sysctl("hw.perflevel0.physicalcpu"),
        "cores_eff": sysctl("hw.perflevel1.physicalcpu"),
        "versions": {d: version(d) for d in ("numpy", "livekit", "pywebrtc-audio", "onnxruntime", "soxr", "protobuf")},
        "speexdsp_lib": SPEEX_LIB if speex_available() else None,
        "speexdsp_version": "1.2.1 (Homebrew Cellar)" if speex_available() else None,
        "threads_env": {k: os.environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "loadavg_1_5_15": loadavg(),
        "aicc_source": "ai-native-callcenter @8fabadc (read-only): voice/rtp.go Run, media/resample.go, aicall/framer.go",
        "g168_source": str(g168_file().relative_to(REPO)),
    }


# ------------------------------------------------------------------------------------- numbers / io


def to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64)), -32768, 32767).astype(np.int16)


def db(x: float, floor: float = 1e-20) -> float:
    return 10.0 * math.log10(max(float(x), floor))


def stats(xs, nd: int = 4) -> dict:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), nd), "p95": round(float(np.percentile(a, 95)), nd),
            "p99": round(float(np.percentile(a, 99)), nd), "max": round(float(a.max()), nd),
            "min": round(float(a.min()), nd), "mean": round(float(a.mean()), nd)}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16), w.getframerate()


def write_wav(path: Path, x16: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(x16, dtype="<i2").tobytes())


def wav_bytes(x16: np.ndarray, rate: int) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(x16, dtype="<i2").tobytes())
    return buf.getvalue()


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default) + "\n")
    return p


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------- G.711


def g711_roundtrip(x16: np.ndarray, law: str) -> np.ndarray:
    """int16 -> G.711 encode -> decode (stdlib audioop, as A2's telephone copies; S7 matched its LUTs)."""
    pcm = np.asarray(x16, dtype="<i2").tobytes()
    if law == "ulaw":
        out = audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2)
    elif law == "alaw":
        out = audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2)
    else:
        raise ValueError(law)
    return np.frombuffer(out, dtype="<i2").astype(np.int16)


# -------------------------------------------------------------------------------- AICC resamplers


def aicc_down_taps(factor: int = 3, taps: int = 32) -> np.ndarray:
    """media.NewDownsampler(3, 32): an odd 33-tap windowed sinc, cutoff 0.5/factor, Hamming, unity DC gain."""
    if taps % 2 == 0:
        taps += 1
    cutoff = 0.5 / factor
    i = np.arange(taps, dtype=np.float64)
    x = i - (taps - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(x == 0, 2 * cutoff, np.sin(2 * np.pi * cutoff * x) / (np.pi * x))
    h *= 0.54 - 0.46 * np.cos(2 * np.pi * i / (taps - 1))
    return h / h.sum()


AICC_DOWN = aicc_down_taps()


def aicc_downsample(x24: np.ndarray) -> np.ndarray:
    """AICC's downlink 24 kHz -> 8 kHz on one continuous stream (fresh filter history):
    out[k] = clamp16(round(sum_j h[j] * x[3k - j])), which is what Downsampler.Process computes frame by
    frame (its phase and history carry across chunks)."""
    y = np.convolve(np.asarray(x24, dtype=np.float64), AICC_DOWN)[: len(x24)]
    return to_i16(y[0::3])


def aicc_upsample(x8: np.ndarray, factor: int = 3, frame: int = TEL_FRAME) -> np.ndarray:
    """AICC's uplink 8 kHz -> 24 kHz: media.Upsample applied to each 20 ms RTP frame on its own
    (aicall/session.go pumpCallerAudio): linear interpolation, and the frame's last sample held for
    `factor` samples because it has no successor inside the frame."""
    n = len(x8) // frame * frame
    f = np.asarray(x8[:n], dtype=np.float64).reshape(-1, frame)
    nxt = np.concatenate([f[:, 1:], f[:, -1:]], axis=1)
    step = (nxt - f) / factor
    out = f[:, :, None] + step[:, :, None] * np.arange(factor, dtype=np.float64)[None, None, :]
    return to_i16(out.reshape(-1))


def soxr_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """One-shot soxr HQ in float32, as the gateway's ResampleStream (S7: bit-identical to one-shot)."""
    import soxr

    y = soxr.resample(np.asarray(x, dtype=np.float32) / 32768.0, sr_in, sr_out, quality="HQ")
    return y.astype(np.float64) * 32768.0


# --------------------------------------------------------------------------------- G.168 echo paths


def g168_file() -> Path:
    return G168_A2 if G168_A2.exists() else G168_A3


def load_g168() -> dict[str, dict]:
    """{name: {coefficients, K_css, min_erl_db, length}} from A2's JSON (or A3's)."""
    p = g168_file()
    js = json.loads(p.read_text())
    out = {}
    if isinstance(js.get("models"), list):  # A2 layout
        for m in js["models"]:
            out[m["name"]] = {"m": np.asarray(m["coefficients"], dtype=np.float64), "K_css": float(m["K_css"]),
                              "min_erl_db": float(m["min_erl_css_db"]), "peak_index": int(m["peak_index"])}
    else:  # A3 layout
        for name, m in js["models"].items():
            st = m["stated"]
            c = np.asarray(m["coefficients"], dtype=np.float64)
            out[name] = {"m": c, "K_css": float(st["Ki_css_or_white_noise_table_D1a"]),
                         "min_erl_db": float(st["min_erl_db_for_css_table_D1a"]), "peak_index": int(np.argmax(np.abs(c)))}
    return out


def echo_path(model: dict, erl_db: float) -> tuple[np.ndarray, float]:
    """Eq. D-1 without the bulk delay: g(k) = 10^(-ERL/20) * K_css * m(k). ERL is raised to the model's
    minimum (Table D.1a) when the requested value is below it; the effective ERL is returned."""
    erl = max(float(erl_db), model["min_erl_db"])
    return 10 ** (-erl / 20) * model["K_css"] * model["m"], erl


# ------------------------------------------------------------------------------------------- levels


def _fftconv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[: len(a)]


def p56_active_level(x: np.ndarray, fs: int) -> tuple[float, float]:
    """Active speech level in dBov and activity factor: ITU-T P.56 method B as in G.191 sv56.
    Ported verbatim from spikes/datasets/a2_data.py (A2), so levels match the manifest's.
    `x` is float in [-1, 1)."""
    t_const, hangover, margin = 0.03, 0.2, 15.9
    g = np.exp(-1.0 / (t_const * fs))
    hang = int(np.floor(hangover * fs + 0.5))
    k = (1 - g) * g ** np.arange(int(np.ceil(12 * t_const * fs)))
    q = _fftconv(_fftconv(np.abs(x), k), k)
    n = len(x)
    idx = np.arange(n)
    sq = float(np.sum(x * x))
    levels, diffs, counts = [], [], []
    for j in range(15):
        c = 2.0 ** (j - 14)
        last = np.maximum.accumulate(np.where(q >= c, idx, -(10 ** 12)))
        a = int(np.count_nonzero(idx - last <= hang))
        if a == 0:
            break
        levels.append(10 * np.log10(sq / a))
        diffs.append(levels[-1] - 20 * np.log10(c))
        counts.append(a)
    for j, d in enumerate(diffs):
        if d <= margin:
            if j == 0:
                return levels[0], counts[0] / n
            t = (diffs[j - 1] - margin) / (diffs[j - 1] - d)
            level = levels[j - 1] + t * (levels[j] - levels[j - 1])
            return float(level), float(10 ** ((10 * np.log10(sq / n) - level) / 10))
    raise ValueError("no speech activity found (signal too quiet)")


def dbm0_to_dbov(dbm0: float) -> float:
    return dbm0 + DBM0_TO_DBOV


# ------------------------------------------------------------------------------------------ filters


def lowpass_taps(cutoff_hz: float, fs: int, taps: int) -> np.ndarray:
    """Linear-phase windowed-sinc low-pass (Blackman), unity DC gain."""
    i = np.arange(taps, dtype=np.float64)
    x = i - (taps - 1) / 2
    fc = cutoff_hz / fs
    h = 2 * fc * np.sinc(2 * fc * x) * np.blackman(taps)
    return h / h.sum()


def bandpass_taps(lo_hz: float, hi_hz: float, fs: int, taps: int) -> np.ndarray:
    hi = lowpass_taps(hi_hz, fs, taps)
    lo = lowpass_taps(lo_hz, fs, taps)
    return hi - lo


class StreamFir:
    """Causal FIR applied frame by frame with carried history (what the gateway would run per 10 ms)."""

    def __init__(self, taps: np.ndarray):
        self.h = np.asarray(taps, dtype=np.float64)
        self.hist = np.zeros(len(self.h) - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.hist, np.asarray(x, dtype=np.float64)])
        y = np.convolve(buf, self.h, mode="valid")
        self.hist = buf[len(buf) - (len(self.h) - 1):]
        return y


# Pre-filter (A3 design point 8): low-pass both signals at 3.8 kHz, because AICC's linear-interpolation
# upsampling folds images of the 0-4 kHz band into 4-8 kHz, and those are not linear in the reference.
PREFILTER_TAPS = lowpass_taps(3800.0, SR, 63)  # 63 taps: group delay 31 samples (1.94 ms) on both paths


def band_noise(n: int, fs: int, dbm0: float, rng: np.random.Generator, lo: float = 300.0, hi: float = 3400.0) -> np.ndarray:
    """Band-limited white Gaussian noise (G.168 test 9 style) at a power level in dBm0 (RMS, not active)."""
    h = bandpass_taps(lo, hi, fs, 255)
    w = rng.standard_normal(n + len(h))
    y = np.convolve(w, h, mode="valid")[:n]
    target_rms = 32768.0 * 10 ** (dbm0_to_dbov(dbm0) / 20)
    return y * (target_rms / np.sqrt(np.mean(y * y)))


# ------------------------------------------------------------------------------------------- Silero


def _silero_module():
    spec = importlib.util.spec_from_file_location("s7_silero_vad", SILERO_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def silero_session():
    """onnxruntime CPU session as S7 ran Silero: sequential, 1 intra-op thread, spinning off."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(SILERO_ONNX), sess_options=so, providers=["CPUExecutionProvider"])


class Silero:
    """One continuous Silero stream per signal (the spikes/cpu numpy wrapper, ported from pipecat),
    with the gateway's model-state reset policy (src/mlx_realtime/vad/silero.py, config.yaml
    turn.vad.model_reset_interval_s = 2.0): reset once >= 2 s of audio has passed since the last
    reset, the speech gate is quiet (no candidate, not speaking; gate.py with threshold 0.5,
    min_speech_ms 150, silence_duration_ms 200) and the last 5 windows scored < 0.1. `in_speech` is the
    gate state before the current window, as session/feeder.py passes it."""

    _mod = None

    def __init__(self, session=None, reset_interval_s: float | None = 2.0):
        if Silero._mod is None:
            Silero._mod = _silero_module()
        self.session = session or silero_session()
        self.reset_interval_s = reset_interval_s
        self.resets = 0

    def probs(self, x16: np.ndarray) -> np.ndarray:
        m = Silero._mod.SileroOnnxModel(self.session)
        x = np.asarray(x16, dtype=np.float32) / 32768.0
        nw = len(x) // SILERO_WIN
        out = np.empty(nw, dtype=np.float32)
        reset_w = None if self.reset_interval_s is None else self.reset_interval_s * SR / SILERO_WIN
        since = low = 0
        speaking, cand, speech_end = False, None, 0
        self.resets = 0
        for i in range(nw):
            in_speech = speaking or cand is not None
            p = float(m(x[i * SILERO_WIN:(i + 1) * SILERO_WIN], SR)[0, 0])
            out[i] = p
            if reset_w is not None:  # SileroStream._maybe_reset
                since += 1
                low = low + 1 if p < 0.1 else 0
                if since >= reset_w and not in_speech and low >= 5:
                    m.reset_states()
                    since = 0
                    self.resets += 1
            start, end = 32 * i, 32 * (i + 1)  # SpeechGate.on_frame, ms
            if speaking:
                if p >= SILERO_OFF:
                    speech_end = end
                elif end - speech_end >= 200:
                    speaking = False
            elif p >= SILERO_ON or (cand is not None and p >= SILERO_OFF):
                if cand is None:
                    cand = start
                if end - cand >= 150:
                    speaking, speech_end, cand = True, end, None
            else:
                cand = None
        return out


def vad_runs(p: np.ndarray, min_speech_ms: float) -> list[tuple[int, int, int]]:
    """Hysteresis runs (enter p >= 0.5, stay while p >= 0.35) as (start_win, confirm_win, end_win);
    confirm_win is where the run reaches min_speech_ms, i.e. where speech_started would be emitted
    (-1 if the run is shorter). Window i covers [32 i, 32 i + 32) ms."""
    need = max(1, math.ceil(min_speech_ms / 32.0))
    runs, i, n = [], 0, len(p)
    while i < n:
        if p[i] >= SILERO_ON:
            j = i
            while j < n and p[j] >= SILERO_OFF:
                j += 1
            confirm = i + need - 1 if j - i >= need else -1
            runs.append((i, confirm, j))
            i = j
        else:
            i += 1
    return runs

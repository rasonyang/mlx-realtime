"""Shared helpers for the S7 spike scripts (M0, throwaway).

Paths, environment capture, load-average guard, percentiles, ONNX session
construction, audio I/O for the downloaded test set, and the telephone
degradation chain. Nothing here is production code.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA = HERE / "data"
RESULTS = HERE.parent / "results"

ST_FILES = {
    "v3.2-cpu": "smart-turn-v3.2-cpu.onnx",  # int8
    "v3.2-gpu": "smart-turn-v3.2-gpu.onnx",  # fp32
}
SILERO_FILE = "silero_vad.onnx"
TEST_SET_DIR = DATA / "smart-turn-data-v3.2-test"
ZH_MANIFEST = TEST_SET_DIR / "zho_manifest.json"

LOAD_LIMIT = 4.0


# ---------------------------------------------------------------- environment


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return None


def environment() -> dict:
    """Versions and machine facts recorded with every result file (SPIKE.md §1)."""
    import onnxruntime as ort

    env = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "cores_logical": os.cpu_count(),
        "cores_perf": _sysctl("hw.perflevel0.physicalcpu"),
        "cores_eff": _sysctl("hw.perflevel1.physicalcpu"),
        "mem_bytes": _sysctl("hw.memsize"),
        "iogpu_wired_limit_mb": _sysctl("iogpu.wired_limit_mb"),
        "mx_device_info": "n/a: the gateway/CPU environment is MLX-free by design",
        "onnxruntime": ort.__version__,
        "ort_providers": ort.get_available_providers(),
        "numpy": np.__version__,
        "soxr": _version("soxr"),
        "soundfile": _version("soundfile"),
        "huggingface_hub": _version("huggingface-hub"),
        "pyarrow": _version("pyarrow"),
        "transformers": _version("transformers"),
        # The gateway environment is MLX-free by design; recorded as absent on purpose.
        "mlx": _version("mlx"),
        "mlx_lm": _version("mlx-lm"),
        "mlx_audio": _version("mlx-audio"),
        "openai": _version("openai"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    }
    manifest = MODELS / "MANIFEST.json"
    if manifest.exists():
        env["models_manifest"] = json.loads(manifest.read_text())
    return env


# ------------------------------------------------------------- load average


def loadavg1() -> float:
    return round(os.getloadavg()[0], 2)


def wait_for_load(limit: float = LOAD_LIMIT, max_wait_s: float = 300.0, poll_s: float = 10.0) -> dict:
    """Wait (bounded) until the 1-minute load average is at or below `limit`."""
    t0 = time.monotonic()
    first = loadavg1()
    cur = first
    while cur > limit and time.monotonic() - t0 < max_wait_s:
        time.sleep(poll_s)
        cur = loadavg1()
    return {"load_at_request": first, "load_at_start": cur, "waited_s": round(time.monotonic() - t0, 1)}


@contextmanager
def load_guard(record: dict, limit: float = LOAD_LIMIT, max_wait_s: float = 300.0):
    """Record the 1-min load average before and after a timed run, waiting first if it is high.

    The "after" value includes this process's own busy threads, so it is an upper bound on
    outside load. A run is flagged when the load before it was still above `limit`.
    """
    w = wait_for_load(limit, max_wait_s)
    record["load_before"] = w["load_at_start"]
    record["load_wait_s"] = w["waited_s"]
    try:
        yield record
    finally:
        record["load_after"] = loadavg1()
        record["load_flag"] = record["load_before"] > limit


# ---------------------------------------------------------------- statistics


def stats_ms(ns: list[int] | np.ndarray) -> dict:
    """p50/p95/p99/max/mean in milliseconds from nanosecond samples."""
    a = np.asarray(ns, dtype=np.float64) / 1e6
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
    }


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(f"wrote {path}")
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ sessions


def ort_session(model: Path, intra: int = 1, spinning: bool = True):
    """CPUExecutionProvider session, sequential, inter-op 1 (the Smart Turn reference settings)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = intra
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    flag = "1" if spinning else "0"
    so.add_session_config_entry("session.intra_op.allow_spinning", flag)
    so.add_session_config_entry("session.inter_op.allow_spinning", flag)
    return ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])


def model_path(name: str) -> Path:
    p = MODELS / name
    if not p.exists():
        sys.exit(f"missing {p}: run `uv run python fetch_models.py` first")
    return p


# --------------------------------------------------------------------- audio

SR = 16000
WINDOW_S = 8


def last_8s_front_padded(audio: np.ndarray) -> np.ndarray:
    """The Smart Turn input window: the last 8 s, zero-padded at the front (reference audio_utils)."""
    n = SR * WINDOW_S
    if audio.size > n:
        return audio[-n:]
    if audio.size < n:
        return np.pad(audio, (n - audio.size, 0))
    return audio


def load_zh_rows(manifest: Path = ZH_MANIFEST) -> list[dict]:
    """Rows of the downloaded Chinese subset, audio as float32 at 16 kHz (resampled with soxr HQ if needed)."""
    import soundfile as sf
    import soxr

    if not manifest.exists():
        sys.exit(f"missing {manifest}: run `uv run python fetch_testdata.py` first")
    meta = json.loads(manifest.read_text())
    rows = []
    for r in meta["rows"]:
        path = manifest.parent / "zho" / r["file"]
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1).astype(np.float32)
        r = dict(r)
        r["native_sr"] = int(sr)
        if sr != SR:
            audio = soxr.resample(audio, sr, SR, quality="HQ").astype(np.float32)
        r["audio"] = audio
        rows.append(r)
    return rows


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def telephone_degrade(audio16k: np.ndarray, quality: str = "HQ") -> tuple[np.ndarray, np.ndarray]:
    """16 kHz float -> 8 kHz (soxr) -> G.711 mu-law round trip (stdlib audioop) -> 16 kHz (soxr).

    Returns (audio_8k_after_mulaw as float32, audio_16k_degraded as float32).
    audioop is deprecated in 3.12 and is used here only to make test data.
    """
    import warnings

    import soxr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop

    x8 = soxr.resample(audio16k.astype(np.float32), SR, 8000, quality=quality)
    pcm = to_int16(x8).tobytes()
    mu = audioop.lin2ulaw(pcm, 2)
    back = np.frombuffer(audioop.ulaw2lin(mu, 2), dtype=np.int16).astype(np.float32) / 32768.0
    x16 = soxr.resample(back, 8000, SR, quality=quality).astype(np.float32)
    return back, x16

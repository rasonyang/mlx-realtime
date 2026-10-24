"""Shared helpers of the transcription spike: paths, WAV I/O, the manifest, text normalization.

Normalization is asr_context's (spikes/asr_context/common.py): the repository's
``bench.scoring.norm_text``, then every whitespace removed and the Chinese digit characters
零一二三四五六七八九〇幺 mapped to 0-9. CER is the character edit distance over the normalized
strings, pooled (total edits / total reference characters); Latin letters and digits are
characters too.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"  # git-ignored
RESULTS = REPO / "spikes" / "results"
sys.path.insert(0, str(REPO))  # bench/ is not part of the mlx_realtime package
from bench.scoring import norm_text  # noqa: E402

RATE = 24000
FRAME = 480  # 20 ms at 24 kHz


def load_calls() -> list[dict]:
    return [json.loads(line) for line in (HERE / "calls.jsonl").read_text().splitlines() if line]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").copy()


def write_wav(path: Path, x: np.ndarray, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(x, dtype="<i2").tobytes())


_DIGITS = str.maketrans("零一二三四五六七八九〇幺", "0123456789" + "01")


def norm(text: str) -> str:
    return "".join(norm_text(text).translate(_DIGITS).split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance; rows vectorised with numpy (a long leg is thousands of characters)."""
    if not a or not b:
        return max(len(a), len(b))
    bv = np.frombuffer(b.encode("utf-32-le"), dtype="<u4")
    idx = np.arange(len(b) + 1)
    prev = idx.copy()
    for i, ch in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (bv != ord(ch)), prev[1:] + 1)
        cur = np.minimum.accumulate(cur - idx) + idx  # cur[j] = min(cur[j], cur[j-1] + 1)
        prev = cur
    return int(prev[-1])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[max(0, min(len(v) - 1, round(q * len(v) + 0.5) - 1))]

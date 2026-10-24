"""Shared helpers: paths, WAV I/O, the manifest, text normalization and the metrics.

Metric definitions (all on the manifest ``text`` as reference):

Normalization ``norm(text)``: the repository's ``bench.scoring.norm_text`` (NFKC, lower-case,
punctuation and symbols to spaces, whitespace next to CJK removed), then every remaining
whitespace removed, and the Chinese digit characters 零一二三四五六七八九〇幺 mapped to 0-9 (a
code read out digit by digit is written either way: "八七九零" = "8790"). "Wi-Fi 7" -> "wifi7",
"NovaNet 是做什么的？" -> "novanet是做什么的". Numbers are otherwise not verbalized: "5G 畅享 199"
heard as "5G畅享一百九十九" is a miss.

Term hit: a term counts as hit when ``norm(term)`` is a substring of ``norm(hypothesis)``.
Term hit rate = hits / term occurrences (pooled; a two-term utterance counts two). Orders are
pooled with the rest and also reported per category.

CER: character edit distance over ``norm(...)`` strings, pooled (total edits / total reference
characters). Latin letters and digits are characters too.

Non-term CER: the term occurrences are masked out of the reference AND the hypothesis.
  1. ``norm(text)`` is aligned with ``norm(hyp)`` by Levenshtein DP (substitution, deletion and
     insertion cost 1; ties prefer the diagonal, then deletion).
  2. The reference characters of each term occurrence form a span (located in ``norm(text)``).
     The hypothesis side of a span is everything between the nearest exactly-matched reference
     characters on its left and right (so a garble longer or shorter than the term is removed
     whole, and an error on the neighbouring words still counts unless it sits between the term
     and an exact-match anchor).
  3. Both sides are removed. Non-term CER = edit distance(masked ref, masked hyp) / len(masked ref),
     pooled. A term misheard as other words costs nothing here.
For utterances without terms (ASCEND) it equals CER.
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "data"  # git-ignored (data/.gitignore)
RESULTS = REPO / "spikes" / "results"
sys.path.insert(0, str(REPO))  # bench/ is not part of the mlx_realtime package
from bench.scoring import norm_text  # noqa: E402


def shared_data() -> Path:
    """spikes/data of this checkout, or $MLXRT_MAIN_DATA (the main checkout's, from a worktree)."""
    candidates = [REPO / "spikes" / "data"]
    if os.environ.get("MLXRT_MAIN_DATA"):
        candidates.append(Path(os.environ["MLXRT_MAIN_DATA"]))
    for p in candidates:
        if (p / "manifest.jsonl").exists():
            return p
    raise SystemExit("spikes/data not found: set MLXRT_MAIN_DATA")


def load_manifest(path: Path | None = None) -> list[dict]:
    return [json.loads(line) for line in (path or HERE / "corpus.jsonl").read_text().splitlines() if line]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").copy(), f.getframerate()


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(x, dtype="<i2").tobytes())


# ----------------------------------------------------------------------------- text and metrics


_DIGITS = str.maketrans("零一二三四五六七八九〇幺", "0123456789" + "01")


def norm(text: str) -> str:
    return "".join(norm_text(text).translate(_DIGITS).split())


def term_hit(term: str, hyp: str) -> bool:
    return norm(term) in norm(hyp)


def _dp(a: str, b: str) -> list[list[int]]:
    prev = list(range(len(b) + 1))
    table = [prev]
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (ai != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        table.append(cur)
        prev = cur
    return table


def edit_distance(a: str, b: str) -> int:
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j - 1] + (a[i - 1] != b[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def align_map(ref: str, hyp: str) -> list[list[int]]:
    """For every reference index the hypothesis indices aligned to it (match or substitution), plus
    ``ins[i]``: hypothesis indices inserted after reference position i (i = -1 .. len(ref)-1) in
    the second element. Returned as (per_ref, ins_after)."""
    table = _dp(ref, hyp)
    per_ref: list[list[int]] = [[] for _ in ref]
    ins_after: dict[int, list[int]] = {}
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            per_ref[i - 1].append(j - 1)
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            i -= 1
        else:
            ins_after.setdefault(i - 1, []).append(j - 1)
            j -= 1
    return per_ref, ins_after  # type: ignore[return-value]


def term_spans(ref: str, terms: list[str]) -> list[tuple[int, int]]:
    """[start, end) of every term occurrence in the normalized reference (left to right, no overlap)."""
    spans: list[tuple[int, int]] = []
    taken = [False] * len(ref)
    for term in terms:
        t = norm(term)
        start = 0
        while True:
            k = ref.find(t, start)
            if k < 0:
                break
            if not any(taken[k:k + len(t)]):
                spans.append((k, k + len(t)))
                for q in range(k, k + len(t)):
                    taken[q] = True
            start = k + 1
    return sorted(spans)


def masked_pair(text: str, hyp: str, terms: list[str]) -> tuple[str, str]:
    ref, h = norm(text), norm(hyp)
    spans = term_spans(ref, terms)
    if not spans:
        return ref, h
    per_ref, _ = align_map(ref, h)

    def exact(i: int) -> int | None:  # hypothesis index of reference char i when it matched exactly
        return per_ref[i][0] if per_ref[i] and h[per_ref[i][0]] == ref[i] else None

    drop_ref = set()
    drop_hyp = set()
    for a, b in spans:
        drop_ref.update(range(a, b))
        left = next((exact(i) for i in range(a - 1, -1, -1) if exact(i) is not None), -1)
        right = next((exact(i) for i in range(b, len(ref)) if exact(i) is not None), len(h))
        drop_hyp.update(range(left + 1, right))  # everything between the exactly matched neighbours
    return ("".join(c for i, c in enumerate(ref) if i not in drop_ref),
            "".join(c for i, c in enumerate(h) if i not in drop_hyp))


def score(text: str, hyp: str, terms: list[str]) -> dict:
    ref, h = norm(text), norm(hyp)
    mref, mhyp = masked_pair(text, hyp, terms)
    return {"ref_chars": len(ref), "edits": edit_distance(ref, h),
            "nt_ref_chars": len(mref), "nt_edits": edit_distance(mref, mhyp),
            "terms": len(terms), "hits": sum(term_hit(t, hyp) for t in terms)}

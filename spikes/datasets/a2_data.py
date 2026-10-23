#!/usr/bin/env python3
"""M0 work package A2: public test audio, telephone copies, manifest, and S11 echo material.

Run every step from spikes/datasets/ (the uv project), passing the data and results directories:

    uv run python -I a2_data.py fleurs  --data ../data
    uv run python -I a2_data.py ascend  --data ../data
    uv run python -I a2_data.py build   --data ../data
    uv run python -I a2_data.py g168    --data ../data
    uv run python -I a2_data.py aec     --data ../data
    uv run python -I a2_data.py summary --data ../data --results ../results

Downloads are untrusted data. They are only parsed (TSV, tar headers, WAV, parquet, PDF, CSV) and
never executed. Archive members are read one at a time and written under names this script chooses;
nothing is extracted with tarfile.extract*. Every download is pinned to a revision or commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import platform
import posixpath
import re
import shutil
import statistics
import sys
import tarfile
import time
import unicodedata
import urllib.parse
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # stdlib G.711; deprecated since 3.11 (PEP 594), removed in Python 3.13.

# --------------------------------------------------------------------------------------------------
# Pinned sources
# --------------------------------------------------------------------------------------------------

FLEURS = {
    "name": "FLEURS",
    "repo": "google/fleurs",
    "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
    "license": "CC-BY-4.0",
    "url": "https://huggingface.co/datasets/google/fleurs",
    "split": "test",
}
FLEURS_LANGS = {"zh": "cmn_hans_cn", "en": "en_us"}
# test.tar.gz stores its members sorted by file name (checked while streaming). The candidate pool is
# the first K members, so only a prefix of each archive is downloaded (see README).
FLEURS_PREFIX_MEMBERS = 140
FLEURS_EDGES = [1.0, 5.0, 6.5, 8.0, 10.0001]

ASCEND = {
    "name": "ASCEND",
    "repo": "CAiRE/ASCEND",
    "revision": "737e9800ae31be9932ba8464c80366559bd28424",
    "file": "main/test-00000-of-00001.parquet",
    "license": "CC-BY-SA-4.0",
    "url": "https://huggingface.co/datasets/CAiRE/ASCEND",
    "split": "test",
}
# Only these parquet row groups (100 rows each) are read. Rows are ordered by session and speaker, so
# groups 4, 9 and 11 were chosen to cover both test speakers (spk3, spk17) and all four test sessions
# with enough code-switched candidates, for about 23 MB of audio instead of the 106 MB split (README).
ASCEND_ROW_GROUPS = [4, 9, 11]
ASCEND_EDGES = [1.0, 2.5, 4.0, 6.0, 10.0001]
FILLERS = {"uh", "um", "umm", "oh", "ok", "okay", "yeah", "yes", "no", "hmm", "mm", "ah", "er", "eh", "huh", "wow"}

N_PER_SET = 42
PRIORITY_CAP = 12  # at most this many "has a number" picks are forced in per set
ORDER_TAG = "mlx-realtime-a2"  # mixed into the hash that orders candidates

G168 = {
    "recommendation": "ITU-T G.168",
    "title": "Digital network echo cancellers",
    "edition": "04/2015 (edition 8.0), in force",
    "url": "https://www.itu.int/rec/T-REC-G.168-201504-I/en",
    "pdf_url": "https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-G.168-201504-I!!PDF-E&type=items",
    "pdf_name": "T-REC-G.168-201504-I!!PDF-E.pdf",
    "pdf_sha256": "f093a2d77714dca5fc22c7c3cfaae97c3315ade839b61826dd45c3d0285ceba2",
    "cor_edition": "G.168 (2015) Cor. 1 (12/2022) (edition 8.1), in force; full reprint, 'reference error corrections'",
    "cor_url": "https://www.itu.int/rec/T-REC-G.168-202212-I!Cor1/en",
    "cor_pdf_url": "https://www.itu.int/rec/dologin.asp?lang=e&id=T-REC-G.168-202212-I!Cor1!PDF-E&type=items",
    "cor_pdf_name": "T-REC-G.168-202212-I!Cor1!PDF-E.pdf",
    "cor_pdf_sha256": "2b50b0b1fc249ed692f6e35f93409d594c5baffc8d14a57cd5465c6bba2c1365",
}
# Table D.1a (CSS / white-noise input) and Table D.1b (tone input), transcribed from the PDF and
# re-checked against the page text by step_g168: model -> (K_i, minimum ERL in dB).
G168_K_CSS = {1: (1.39e-5, 6.0), 2: (1.44e-5, 6.55), 3: (1.52e-5, 6.0), 4: (1.77e-5, 6.0),
              5: (9.33e-6, 6.0), 6: (1.51e-5, 6.0), 7: (2.33e-5, 11.06), 8: (1.33e-5, 9.27)}
G168_K_TONE = {1: (1.22e-5, 6.0), 2: (6.78e-6, 6.0), 3: (9.66e-6, 6.0), 4: (1.07e-5, 6.0),
               5: (7.05e-6, 6.0), 6: (8.60e-6, 6.0), 7: (6.58e-6, 6.0), 8: (4.58e-6, 6.0)}
G168_MODEL_INFO = {
    1: ("D.2", "network hybrid simulator", "short dispersion"),
    2: ("D.2", "network hybrid simulator", "median-short dispersion"),
    3: ("D.2", "network hybrid simulator", "median-long dispersion"),
    4: ("D.2", "network hybrid simulator", "long dispersion"),
    5: ("D.3", "measured, North American network", "single reflection, dispersion about 6 ms; closest to the mean measured magnitude response; most frequent"),
    6: ("D.3", "measured, North American network", "single reflection, dispersion about 9 ms (spectral peak near 250 Hz)"),
    7: ("D.3", "measured, North American network", "double reflection, dispersion about 6 ms; notches at the Test 10A/10B tone frequencies"),
    8: ("D.3", "measured, North American network", "double reflection; alternative to m7 for Tests 10A/10B (no notches there)"),
}

AEC = {
    "repo": "microsoft/AEC-Challenge",
    "commit": "8f861f0f8ffca26a3d1d16f17f2594ea89abe7ac",
    "url": "https://github.com/microsoft/AEC-Challenge",
    "dataset": "datasets/synthetic (ICASSP 2021+ synthetic set: 10,000 x 10 s examples at 16 kHz)",
    "code_license": "MIT",
    "data_license": (
        "Provided 'under the original terms that Microsoft received such datasets' (repo README): clean speech "
        "from LibriVox (public domain) and the Edinburgh 56-speaker noisy speech DB (CC-BY-4.0, DataShare 10283/2791); "
        "noise from AudioSet (CC-BY-4.0), Freesound (CC0 only) and DEMAND (CC-BY-SA-3.0)."
    ),
    "n_linear": 10,
    "n_nonlinear": 10,
}
AEC_KINDS = {
    "farend_speech": "farend_speech/farend_speech_fileid_{i}.wav",
    "echo_signal": "echo_signal/echo_fileid_{i}.wav",
    "nearend_speech": "nearend_speech/nearend_speech_fileid_{i}.wav",
    "nearend_mic_signal": "nearend_mic_signal/nearend_mic_fileid_{i}.wav",
}

USER_AGENT = "mlx-realtime-m0-spike/0.1 (dataset fetch)"
SR = 16000
TEL_SR = 8000

# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def log(*parts: object) -> None:
    print(*parts, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hash_order(key: str) -> str:
    return hashlib.sha256(f"{ORDER_TAG}:{key}".encode()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class KeepAliveHTTPS:
    """Plain HTTPS GETs over one kept-alive connection per host (stdlib only, no credentials sent)."""

    def __init__(self) -> None:
        self.conns: dict[str, http.client.HTTPSConnection] = {}
        self.bytes_in = 0

    def get(self, url: str, retries: int = 5) -> bytes:
        for attempt in range(retries):
            hops = 0
            current = url
            try:
                while True:
                    u = urllib.parse.urlsplit(current)
                    conn = self.conns.get(u.netloc)
                    if conn is None:
                        conn = http.client.HTTPSConnection(u.netloc, timeout=180)
                        self.conns[u.netloc] = conn
                    target = u.path + (f"?{u.query}" if u.query else "")
                    conn.request("GET", target, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
                    resp = conn.getresponse()
                    body = resp.read()
                    if resp.status in (301, 302, 303, 307, 308):
                        hops += 1
                        if hops > 5:
                            raise RuntimeError(f"too many redirects for {url}")
                        current = urllib.parse.urljoin(current, resp.getheader("Location") or "")
                        continue
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status} for {current}")
                    self.bytes_in += len(body)
                    return body
            except (http.client.HTTPException, OSError) as exc:
                for c in self.conns.values():
                    c.close()
                self.conns.clear()
                if attempt == retries - 1:
                    raise
                log(f"  retry {attempt + 1} for {url}: {type(exc).__name__}: {exc}")
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def close(self) -> None:
        for c in self.conns.values():
            c.close()


# --------------------------------------------------------------------------------------------------
# Text: normalization for CER/WER, and a number heuristic
# --------------------------------------------------------------------------------------------------

CJK = "㐀-䶿一-鿿豈-﫿"
_SPACE_NEXT_TO_CJK = re.compile(rf"(?<=[{CJK}])\s+|\s+(?=[{CJK}])")


def norm_text(text: str) -> str:
    """Reference text for CER/WER/MER scoring (rule documented in README).

    1. Unicode NFKC (full-width -> half-width), and U+2018/U+2019 -> ASCII apostrophe.
    2. Lower-case.
    3. Every punctuation (P*) or symbol (S*) character becomes a space, except an apostrophe
       between two letters ("what's" stays).
    4. Collapse whitespace; drop whitespace next to a CJK character, keep one space between
       Latin/digit tokens. Numbers are NOT verbalized (see has_number).
    """
    s = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    s = s.lower()
    out = []
    for i, ch in enumerate(s):
        cat = unicodedata.category(ch)
        if ch == "'" and 0 < i < len(s) - 1 and s[i - 1].isalpha() and s[i + 1].isalpha():
            out.append(ch)
        elif cat[0] in "PS":
            out.append(" ")
        else:
            out.append(ch)
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    return _SPACE_NEXT_TO_CJK.sub("", s)


_EN_NUMBER = re.compile(
    r"\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"twentieth|hundredth|percent|dozen|dollars?|euros?|pounds?|yuan|cents?|january|february|april|"
    r"june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_ZH_NUMBER = re.compile(
    r"[零〇二两三四五六七八九十百千][零〇一二两三四五六七八九十百千万亿]"
    r"|一[十百千万亿]"
    r"|[一二两三四五六七八九十]+[月号日年岁块元]"
    r"|[二两三四五六七八九十百千万亿]+[个位间门次本件条张台天周点]"
)


def has_digit(text: str) -> bool:
    return re.search(r"[0-9０-９]", text) is not None


def has_number(text: str) -> bool:
    """Heuristic: digits, Chinese numerals in number-like patterns, English number/month/currency words."""
    return has_digit(text) or bool(_ZH_NUMBER.search(text)) or bool(_EN_NUMBER.search(text))


# --------------------------------------------------------------------------------------------------
# Deterministic stratified selection
# --------------------------------------------------------------------------------------------------


def _quotas(counts: list[int], total: int) -> list[int]:
    q = [0] * len(counts)
    remaining = total
    active = [i for i, c in enumerate(counts) if c > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        for i in list(active):
            take = min(share, counts[i] - q[i], remaining)
            q[i] += take
            remaining -= take
            if q[i] >= counts[i]:
                active.remove(i)
            if remaining == 0:
                break
    return q


def select(cands: list[dict], n: int, edges: list[float], order=None) -> list[dict]:
    """Equal-share length bins (deficits redistributed); number-bearing items first (capped);
    then round-robin over groups (gender or speaker); ties broken by `order` (default: a fixed hash)."""
    order = order or (lambda c: hash_order(c["key"]))
    nb = len(edges) - 1
    bins: list[list[dict]] = [[] for _ in range(nb)]
    for c in cands:
        for b in range(nb):
            if edges[b] <= c["dur"] < edges[b + 1]:
                bins[b].append(c)
                break
    for b in bins:
        b.sort(key=order)
    q = _quotas([len(b) for b in bins], n)
    chosen: list[list[dict]] = [[] for _ in range(nb)]
    taken: set[str] = set()
    prio_left = PRIORITY_CAP
    progress = True
    while prio_left > 0 and progress:
        progress = False
        for b in range(nb):
            if prio_left == 0 or len(chosen[b]) >= q[b]:
                continue
            nxt = next((c for c in bins[b] if c["prio"] and c["key"] not in taken), None)
            if nxt is not None:
                chosen[b].append(nxt)
                taken.add(nxt["key"])
                prio_left -= 1
                progress = True
    for b in range(nb):
        groups = sorted({c["group"] for c in bins[b]})
        queues = {g: [c for c in bins[b] if c["group"] == g and c["key"] not in taken] for g in groups}
        while len(chosen[b]) < q[b] and any(queues.values()):
            counts = {g: sum(1 for c in chosen[b] if c["group"] == g) for g in groups}
            g = min((g for g in groups if queues[g]), key=lambda g: (counts[g], g))
            c = queues[g].pop(0)
            chosen[b].append(c)
            taken.add(c["key"])
    return [c for b in range(nb) for c in chosen[b]]


# --------------------------------------------------------------------------------------------------
# Step: FLEURS (zh, en) - TSV + prefix of test.tar.gz
# --------------------------------------------------------------------------------------------------


class _ResumableHTTPStream(io.RawIOBase):
    """Sequential reader over an HTTP GET that resumes with a Range request after a network error."""

    def __init__(self, session, url: str, headers: dict, label: str) -> None:
        super().__init__()
        self.session, self.url, self.headers, self.label = session, url, headers, label
        self.pos = 0
        self.buf = b""
        self.cm = None
        self.it = None
        self.t0 = time.time()
        self.next_log = 8 << 20
        self.retries = 0
        self._open()

    def _open(self) -> None:
        h = dict(self.headers)
        h["Range"] = f"bytes={self.pos}-"
        self.cm = self.session.stream("GET", self.url, headers=h, follow_redirects=True, timeout=180)
        resp = self.cm.__enter__()
        if resp.status_code != 206:
            raise RuntimeError(f"{self.label}: expected HTTP 206 for a range request, got {resp.status_code}")
        self.it = resp.iter_bytes(1 << 16)

    def _close_http(self) -> None:
        if self.cm is not None:
            try:
                self.cm.__exit__(None, None, None)
            except Exception:
                pass
        self.cm = None
        self.it = None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
            except Exception as exc:  # network error: resume at the current offset
                self.retries += 1
                if self.retries > 8:
                    raise
                log(f"  {self.label}: {type(exc).__name__}; resuming at byte {self.pos}")
                self._close_http()
                time.sleep(5)
                self._open()
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        self.pos += n
        if self.pos >= self.next_log:
            rate = self.pos / max(time.time() - self.t0, 1e-6) / 1e6
            log(f"  {self.label}: {self.pos / 1e6:.0f} MB read ({rate:.2f} MB/s)")
            self.next_log += 8 << 20
        return n

    def close(self) -> None:
        self._close_http()
        super().close()


def _read_fleurs_tsv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) != 7:
                raise ValueError(f"{path}: unexpected TSV row with {len(r)} fields")
            if not re.fullmatch(r"\d+\.wav", r[1]):
                raise ValueError(f"{path}: unexpected file name {r[1]!r}")
            rows.append({"sentence_id": r[0], "file": r[1], "raw_transcription": r[2], "transcription": r[3],
                         "num_samples": int(r[5]), "gender": r[6]})
    return rows


def _verify_hub_file(api, repo: str, revision: str, rel: str, local: Path | None) -> dict:
    info = api.get_paths_info(repo, [rel], repo_type="dataset", revision=revision)[0]
    out = {"path": rel, "size": info.size, "blob_id": info.blob_id}
    if info.lfs is not None:
        out["lfs_sha256"] = info.lfs.sha256
    if local is not None:
        data = local.read_bytes()
        out["sha256"] = sha256_bytes(data)
        ok = (out["sha256"] == info.lfs.sha256) if info.lfs is not None else (git_blob_sha1(data) == info.blob_id)
        if not ok or len(data) != info.size:
            raise RuntimeError(f"{repo}@{revision}:{rel}: downloaded file does not match the Hub metadata")
        out["verified"] = True
    return out


def step_fleurs(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, get_session, hf_hub_download, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    api = HfApi()
    raw = args.data / "raw" / "fleurs"
    for set_name, lang in FLEURS_LANGS.items():
        out_dir = raw / lang / "test"
        sel_path = raw / lang / "selection.json"
        if sel_path.exists() and not args.force:
            log(f"[fleurs] {lang}: selection exists, skipping (use --force to redo)")
            continue
        tsv_rel = f"data/{lang}/test.tsv"
        tar_rel = f"data/{lang}/audio/test.tar.gz"
        tsv_local = Path(hf_hub_download(FLEURS["repo"], tsv_rel, repo_type="dataset",
                                         revision=FLEURS["revision"], local_dir=raw / "hub"))
        tsv_info = _verify_hub_file(api, FLEURS["repo"], FLEURS["revision"], tsv_rel, tsv_local)
        tar_info = _verify_hub_file(api, FLEURS["repo"], FLEURS["revision"], tar_rel, None)
        rows = _read_fleurs_tsv(tsv_local)
        by_file = {r["file"]: r for r in rows}
        archive_order = sorted(by_file)
        pool = archive_order[:FLEURS_PREFIX_MEMBERS]

        # File names are random 64-bit ids, so archive order is already a random order: prefer earlier
        # members (one recording per sentence) and the stream can stop at the last selected member.
        cands, seen_sentences = [], set()
        for f in pool:
            r = by_file[f]
            dur = r["num_samples"] / SR
            if not (1.0 <= dur <= 10.0) or r["sentence_id"] in seen_sentences:
                continue
            seen_sentences.add(r["sentence_id"])
            cands.append({"key": f, "dur": dur, "prio": has_digit(r["raw_transcription"]), "group": r["gender"], "row": r})
        chosen = select(cands, N_PER_SET, FLEURS_EDGES, order=lambda c: c["key"])
        wanted = {c["key"] for c in chosen}
        last_index = max(archive_order.index(f) for f in wanted)
        log(f"[fleurs] {lang}: pool={len(pool)} candidates={len(cands)} selected={len(chosen)}; "
            f"streaming test.tar.gz up to member #{last_index + 1}")

        url = hf_hub_url(FLEURS["repo"], tar_rel, repo_type="dataset", revision=FLEURS["revision"])
        stream = _ResumableHTTPStream(get_session(), url, build_hf_headers(), f"{lang}/test.tar.gz")
        got: dict[str, tuple[bytes, int]] = {}
        index = 0
        try:
            with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    base = posixpath.basename(member.name)
                    if member.name != f"test/{base}" or base != archive_order[index]:
                        raise RuntimeError(f"archive order differs from sorted TSV names at #{index}: {member.name!r}")
                    if base in wanted:
                        if member.size > 64 << 20:
                            raise RuntimeError(f"unexpectedly large member {member.name!r}")
                        data = tf.extractfile(member).read()
                        got[base] = (data, index)
                    index += 1
                    if len(got) == len(wanted):
                        break
        finally:
            compressed_read = stream.pos
            stream.close()
        missing = wanted - set(got)
        if missing:
            raise RuntimeError(f"members not found in the archive prefix: {sorted(missing)[:5]}")

        out_dir.mkdir(parents=True, exist_ok=True)
        selected = []
        for c in chosen:
            data, idx = got[c["key"]]
            r = c["row"]
            info = sf.info(io.BytesIO(data))
            if info.samplerate != SR or info.channels != 1 or info.frames != r["num_samples"]:
                raise RuntimeError(f"{c['key']}: sr={info.samplerate} ch={info.channels} frames={info.frames} "
                                   f"(TSV num_samples={r['num_samples']})")
            (out_dir / c["key"]).write_bytes(data)
            selected.append({
                "file": c["key"], "archive_index": idx, "sentence_id": r["sentence_id"],
                "raw_transcription": r["raw_transcription"], "fleurs_transcription": r["transcription"],
                "num_samples": r["num_samples"], "duration_s": r["num_samples"] / SR, "gender": r["gender"],
                "subtype": info.subtype, "sha256": sha256_bytes(data),
            })
        write_json(sel_path, {
            "source": FLEURS, "config": lang, "set": set_name,
            "tsv": tsv_info, "archive": tar_info,
            "archive_prefix": {"members_in_pool": len(pool), "members_read": index,
                               "compressed_bytes_read": compressed_read},
            "pool_candidates": len(cands), "selected": selected,
        })
        log(f"[fleurs] {lang}: saved {len(selected)} WAVs; read {compressed_read / 1e6:.1f} MB of "
            f"{tar_info['size'] / 1e6:.1f} MB")


# --------------------------------------------------------------------------------------------------
# Step: ASCEND (code-switched) - selected parquet row groups only
# --------------------------------------------------------------------------------------------------


def _latin_words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z']*", text.lower())


def step_ascend(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    raw = args.data / "raw" / "ascend"
    sel_path = raw / "selection.json"
    if sel_path.exists() and not args.force:
        log("[ascend] selection exists, skipping (use --force to redo)")
        return
    file_info = _verify_hub_file(HfApi(), ASCEND["repo"], ASCEND["revision"], ASCEND["file"], None)
    fs = HfFileSystem()
    hub_path = f"datasets/{ASCEND['repo']}@{ASCEND['revision']}/{ASCEND['file']}"
    small_cols = ["id", "path", "transcription", "duration", "language", "original_speaker_id", "session_id", "topic"]
    # cache_type="none": every read is one HTTP range request of exactly the bytes pyarrow asks for
    # (pre_buffer coalesces adjacent column chunks); a read-ahead cache would pull megabytes of audio.
    with fs.open(hub_path, "rb", cache_type="none") as f:
        pf = pq.ParquetFile(f, pre_buffer=True)
        md = pf.metadata
        if md.num_rows != 1315:
            raise RuntimeError(f"unexpected row count {md.num_rows}")
        rows = []
        for g in ASCEND_ROW_GROUPS:
            for i, r in enumerate(pf.read_row_group(g, columns=small_cols).to_pylist()):
                r["row_group"], r["row_in_group"] = g, i
                rows.append(r)
        cands, seen = [], set()
        for r in rows:
            text = r["transcription"]
            tn = norm_text(text)
            words = _latin_words(tn)
            cjk = re.findall(rf"[{CJK}]", tn)
            if r["language"] != "mixed" or not (1.0 <= r["duration"] <= 10.0) or "[unk]" in text.lower():
                continue
            if not cjk or not [w for w in words if w not in FILLERS] or len(cjk) + len(words) < 4 or tn in seen:
                continue
            seen.add(tn)
            cands.append({"key": r["id"], "dur": r["duration"], "prio": has_number(text),
                          "group": f"spk{r['original_speaker_id']}", "row": r})
        chosen = select(cands, N_PER_SET, ASCEND_EDGES)
        wanted = {c["key"] for c in chosen}
        groups = sorted({c["row"]["row_group"] for c in chosen})
        audio_bytes_read = 0
        audio: dict[str, bytes] = {}
        for g in groups:
            for c in range(md.row_group(g).num_columns):
                cc = md.row_group(g).column(c)
                if cc.path_in_schema.startswith("audio."):
                    audio_bytes_read += cc.total_compressed_size
            t = pf.read_row_group(g, columns=["id", "audio"]).to_pylist()
            for r in t:
                if r["id"] in wanted:
                    audio[r["id"]] = r["audio"]["bytes"]
    log(f"[ascend] candidates={len(cands)} selected={len(chosen)} row_groups={groups} "
        f"audio column bytes ~{audio_bytes_read / 1e6:.1f} MB")
    out_dir = raw / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for c in chosen:
        r = c["row"]
        data = audio[c["key"]]
        info = sf.info(io.BytesIO(data))
        if info.samplerate != SR or info.channels != 1 or abs(info.frames / SR - r["duration"]) > 0.02:
            raise RuntimeError(f"{c['key']}: sr={info.samplerate} ch={info.channels} frames={info.frames} dur={r['duration']}")
        name = f"{c['key']}.wav"
        (out_dir / name).write_bytes(data)
        selected.append({
            "file": name, "id": r["id"], "source_path": posixpath.basename(r["path"] or ""),
            "transcription": r["transcription"], "duration_s": info.frames / SR, "language": r["language"],
            "speaker": f"spk{r['original_speaker_id']}", "session_id": r["session_id"], "topic": r["topic"],
            "row_group": r["row_group"], "subtype": info.subtype, "sha256": sha256_bytes(data),
        })
    write_json(sel_path, {
        "source": ASCEND, "set": "mix", "parquet": file_info,
        "row_groups_read": ASCEND_ROW_GROUPS, "audio_row_groups": groups,
        "audio_column_bytes_read": audio_bytes_read, "pool_rows": len(rows), "pool_candidates": len(cands),
        "selected": selected,
    })
    log(f"[ascend] saved {len(selected)} WAVs")


# --------------------------------------------------------------------------------------------------
# Step: build - 16 kHz copies, telephone copies (G.711), manifest
# --------------------------------------------------------------------------------------------------


TARGET_LEVEL_DBOV = -26.0  # nominal active speech level for speech tests (ITU-T P.56 measure)
PEAK_LIMIT_DBFS = -1.0


def _fftconv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    n = len(x) + len(k) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(k, nfft), nfft)[: len(x)]


def p56_active_level(x: np.ndarray, fs: int) -> tuple[float, float]:
    """Active speech level in dBov (0 dBov = full-scale square wave) and activity factor, following
    ITU-T P.56 method B as in the G.191 `sv56` tool: |x| smoothed twice (time constant 0.03 s),
    thresholds 2^-14..2^0 with a 0.2 s hangover, margin 15.9 dB, linear interpolation between thresholds.
    The recursive smoother is applied as an FFT convolution with its impulse response cut at 12 time constants."""
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


def _to_int16(x: np.ndarray) -> tuple[np.ndarray, int]:
    y = np.round(np.asarray(x, dtype=np.float64) * 32768.0)
    clipped = int(np.count_nonzero((y > 32767) | (y < -32768)))
    return np.clip(y, -32768, 32767).astype("<i2"), clipped


def _levels(x16: np.ndarray) -> tuple[float, float]:
    x = x16.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return (round(20 * np.log10(max(rms, 1e-10)), 2), round(20 * np.log10(max(peak, 1e-10)), 2))


def telephone_copy(x16: np.ndarray, codec: str) -> tuple[bytes, np.ndarray, dict]:
    """16 kHz int16 -> 8 kHz (soxr HQ) -> int16 -> G.711 (audioop) -> decode -> 16 kHz (soxr HQ) -> int16."""
    x8f = soxr.resample(x16.astype(np.float64) / 32768.0, SR, TEL_SR, quality="HQ")
    x8, clip8 = _to_int16(x8f)
    pcm = x8.tobytes()
    if sys.byteorder != "little":
        raise RuntimeError("audioop works in native byte order; this script assumes little-endian")
    if codec == "ulaw":
        g711 = audioop.lin2ulaw(pcm, 2)
        dec = audioop.ulaw2lin(g711, 2)
    elif codec == "alaw":
        g711 = audioop.lin2alaw(pcm, 2)
        dec = audioop.alaw2lin(g711, 2)
    else:
        raise ValueError(codec)
    y8 = np.frombuffer(dec, dtype="<i2").astype(np.float64) / 32768.0
    y16, clip16 = _to_int16(soxr.resample(y8, TEL_SR, SR, quality="HQ"))
    return g711, y16, {"clipped_8k": clip8, "clipped_tel_16k": clip16, "samples_8k": len(g711)}


def _items_for_build(data: Path) -> list[dict]:
    items = []
    for set_name, lang in FLEURS_LANGS.items():
        sel = read_json(data / "raw" / "fleurs" / lang / "selection.json")
        for s in sel["selected"]:
            items.append({
                "set": set_name, "raw_path": data / "raw" / "fleurs" / lang / "test" / s["file"],
                "raw_sha256": s["sha256"], "duration_s": s["duration_s"], "text": s["raw_transcription"],
                "source": FLEURS["name"], "source_repo": FLEURS["repo"], "source_config": lang,
                "source_id": s["file"], "source_ref": f"sentence {s['sentence_id']}", "split": FLEURS["split"],
                "revision": FLEURS["revision"], "license": FLEURS["license"], "url": FLEURS["url"],
                "speaker": s["gender"].lower(),
            })
    sel = read_json(data / "raw" / "ascend" / "selection.json")
    for s in sel["selected"]:
        items.append({
            "set": "mix", "raw_path": data / "raw" / "ascend" / "test" / s["file"], "raw_sha256": s["sha256"],
            "duration_s": s["duration_s"], "text": s["transcription"], "source": ASCEND["name"],
            "source_repo": ASCEND["repo"], "source_config": "main", "source_id": s["id"],
            "source_ref": s["source_path"], "split": ASCEND["split"], "revision": ASCEND["revision"],
            "license": ASCEND["license"], "url": ASCEND["url"],
            "speaker": f"{s['speaker']}/ses{s['session_id']}",
        })
    return items


def step_build(args: argparse.Namespace) -> None:
    data = args.data
    items = _items_for_build(data)
    dirs = {k: data / "asr" / k for k in ("16k", "tel16k", "tel8k")}
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rows = []
    for set_name in ("zh", "en", "mix"):
        subset = sorted((it for it in items if it["set"] == set_name),
                        key=lambda it: (round(it["duration_s"], 3), it["source_id"]))
        for i, it in enumerate(subset, 1):
            uid = f"{set_name}_{i:03d}"
            raw_bytes = it["raw_path"].read_bytes()
            if sha256_bytes(raw_bytes) != it["raw_sha256"]:
                raise RuntimeError(f"{it['raw_path']}: checksum differs from the selection record")
            x, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=False)
            if sr != SR or x.ndim != 1:
                raise RuntimeError(f"{it['raw_path']}: sr={sr} ndim={x.ndim}")
            # One gain per utterance, applied to the original samples (float) before PCM16 conversion:
            # active speech level to TARGET_LEVEL_DBOV, unless that would put the peak above PEAK_LIMIT_DBFS.
            src_level, src_activity = p56_active_level(x, SR)
            src_peak_db = 20 * np.log10(max(float(np.max(np.abs(x))), 1e-10))
            gain_db = TARGET_LEVEL_DBOV - src_level
            peak_limited = src_peak_db + gain_db > PEAK_LIMIT_DBFS
            if peak_limited:
                gain_db = PEAK_LIMIT_DBFS - src_peak_db
            x16, clip_src = _to_int16(x * 10 ** (gain_db / 20))
            sf.write(dirs["16k"] / f"{uid}.wav", x16, SR, subtype="PCM_16")
            codec = "alaw" if i % 3 == 0 else "ulaw"
            g711, y16, stats = telephone_copy(x16, codec)
            (dirs["tel8k"] / f"{uid}.{codec}").write_bytes(g711)
            sf.write(dirs["tel16k"] / f"{uid}.wav", y16, SR, subtype="PCM_16")
            rms, peak = _levels(x16)
            level_16k, _ = p56_active_level(x16.astype(np.float64) / 32768.0, SR)
            row = {
                "id": uid, "set": set_name,
                "path_16k": f"asr/16k/{uid}.wav",
                "path_tel_16k": f"asr/tel16k/{uid}.wav",
                f"path_8k_{codec}": f"asr/tel8k/{uid}.{codec}",
                "codec_8k": codec,
                "duration_s": round(len(x16) / SR, 4),
                "text": it["text"], "text_norm": norm_text(it["text"]),
                "has_number": has_number(it["text"]),
                "source": it["source"], "source_repo": it["source_repo"], "source_config": it["source_config"],
                "source_id": it["source_id"], "source_ref": it["source_ref"],
                "split": it["split"], "revision": it["revision"], "license": it["license"], "url": it["url"],
                "speaker": it["speaker"],
                "active_level_dbov": round(level_16k, 2), "rms_dbfs": rms, "peak_dbfs": peak,
                "gain_db": round(gain_db, 2), "peak_limited": bool(peak_limited),
                "source_active_level_dbov": round(src_level, 2), "source_activity": round(src_activity, 3),
                "clipped_samples": {"src_to_pcm16": clip_src, **{k: v for k, v in stats.items() if k.startswith("clipped")}},
                "sha256_16k": sha256_file(dirs["16k"] / f"{uid}.wav"),
                "sha256_source": it["raw_sha256"],
            }
            rows.append(row)
        log(f"[build] {set_name}: {len(subset)} utterances")
    with open(data / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"[build] wrote {len(rows)} rows to {data / 'manifest.jsonl'}")


# --------------------------------------------------------------------------------------------------
# Step: g168 - echo path models from Annex D
# --------------------------------------------------------------------------------------------------

_NUM_TOKEN = re.compile(r"^[-–−]?\d+$")


def _printed_page(words: list[dict]) -> str | None:
    lines: dict[int, list[dict]] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    for top in sorted(lines, reverse=True)[:4] + sorted(lines)[:3]:
        text = " ".join(x["text"] for x in sorted(lines[top], key=lambda x: x["x0"]))
        if "Rec. ITU-T G.168" in text:
            m = re.match(r"^(\d+)\s+Rec\.", text) or re.search(r"\)\s+(\d+)$", text)
            if m:
                return m.group(1)
    return None


def _extract_tables(pdf_path: Path) -> dict[int, dict]:
    import pdfplumber

    tables: dict[int, dict] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pidx, page in enumerate(pdf.pages):
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
            for i, w in enumerate(words[:-1]):
                nxt = words[i + 1]
                m = re.fullmatch(r"D\.([2-9])", nxt["text"])
                if w["text"] != "Table" or not m or abs(nxt["top"] - w["top"]) > 2:
                    continue
                title_words = sorted((x for x in words if abs(x["top"] - w["top"]) <= 2), key=lambda x: x["x0"])
                title = " ".join(x["text"] for x in title_words)
                mm = re.search(r"impulse response of (?:the )?echo path model (\d)$", title)
                if not mm:
                    continue  # a reference in running text, not a table title
                table_no, model = int(m.group(1)), int(mm.group(1))
                if model != table_no - 1:
                    raise RuntimeError(f"page {pidx + 1}: Table D.{table_no} titled model {model}")
                below = [x for x in words if x["top"] > w["bottom"] and _NUM_TOKEN.match(x["text"])]
                lines: list[list[dict]] = []
                for x in sorted(below, key=lambda x: (x["top"], x["x0"])):
                    if lines and abs(lines[-1][0]["top"] - x["top"]) <= 2:
                        lines[-1].append(x)
                    else:
                        lines.append([x])
                rows: list[list[dict]] = []
                for line in lines:
                    if not rows:
                        if len(line) >= 4:
                            rows.append(line)
                        continue
                    gap = line[0]["top"] - rows[-1][0]["top"]
                    spacing = (rows[1][0]["top"] - rows[0][0]["top"]) if len(rows) > 1 else gap
                    if len(line) < 2 or gap > 1.5 * spacing:
                        break
                    rows.append(line)
                centers = sorted((x["x0"] + x["x1"]) / 2 for row in rows for x in row)
                cols: list[list[float]] = [[centers[0]]]
                for c in centers[1:]:
                    if c - cols[-1][-1] > 12:
                        cols.append([c])
                    else:
                        cols[-1].append(c)
                col_x = [sum(c) / len(c) for c in cols]
                grid = []
                for row in rows:
                    row = sorted(row, key=lambda x: x["x0"])
                    idx = [min(range(len(col_x)), key=lambda k: abs(col_x[k] - (x["x0"] + x["x1"]) / 2)) for x in row]
                    if idx != list(range(len(row))):
                        raise RuntimeError(f"Table D.{table_no}: row {len(grid)} does not fill columns left to right: {idx}")
                    grid.append([int(x["text"].replace("–", "-").replace("−", "-")) for x in row])
                n_cols = len(col_x)
                full = [len(r) == n_cols for r in grid]
                # Read in columns: only the last column may be short, and only at the bottom.
                if not all(len(r) in (n_cols, n_cols - 1) for r in grid) or full != sorted(full, reverse=True):
                    raise RuntimeError(f"Table D.{table_no}: unexpected ragged layout {[len(r) for r in grid]}")
                coeffs = [grid[r][c] for c in range(n_cols) for r in range(len(grid)) if c < len(grid[r])]
                tables[model] = {
                    "table": f"Table D.{table_no}", "title": title, "pdf_page_index": pidx + 1,
                    "printed_page": _printed_page(words), "rows": len(grid), "columns": n_cols,
                    "coefficients": coeffs,
                }
    return tables


def _k_tone(m: np.ndarray) -> float:
    spec = np.fft.fft(m.astype(np.float64), 8192)[:4096]  # Eq. D-3: f = 0..4095, 8192-point DFT
    return float(1.0 / np.max(np.abs(spec)))


def _check_k_tables(pdf_path: Path) -> dict:
    """Parse Tables D.1a and D.1b from the page text and compare them with the transcribed constants."""
    import pdfplumber

    row = re.compile(r"^([1-8]) (\d\.\d\d) \S+ 10[–−-](\d) (\d+(?:\.\d+)?)$")
    parsed: dict[str, dict[int, tuple[float, float]]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for label in ("D.1a", "D.1b"):
                title = f"Table {label} – Scaling factors"
                if title not in text:
                    continue
                vals = {}
                for line in text.split(title, 1)[1].splitlines():
                    m = row.match(line.strip())
                    if m:
                        vals[int(m.group(1))] = (float(f"{m.group(2)}e-{m.group(3)}"), float(m.group(4)))
                    elif vals:
                        break
                parsed[label] = vals
    ok_a = parsed.get("D.1a") == G168_K_CSS
    ok_b = parsed.get("D.1b") == G168_K_TONE
    return {"table_D1a_parsed": {f"m{k}": v for k, v in sorted(parsed.get("D.1a", {}).items())},
            "table_D1b_parsed": {f"m{k}": v for k, v in sorted(parsed.get("D.1b", {}).items())},
            "match": ok_a and ok_b}


def step_g168(args: argparse.Namespace) -> None:
    out_dir = args.data / "echo"
    src = out_dir / "g168_src"
    src.mkdir(parents=True, exist_ok=True)
    http = KeepAliveHTTPS()
    pdfs = {}
    for key_url, key_name, key_sha in (("pdf_url", "pdf_name", "pdf_sha256"), ("cor_pdf_url", "cor_pdf_name", "cor_pdf_sha256")):
        path = src / G168[key_name]
        if not path.exists():
            log(f"[g168] downloading {G168[key_url]}")
            path.write_bytes(http.get(G168[key_url]))
        digest = sha256_file(path)
        if digest != G168[key_sha]:
            raise RuntimeError(f"{path.name}: sha256 {digest} differs from the pinned {G168[key_sha]}")
        pdfs[key_name] = path
    http.close()

    main = _extract_tables(pdfs["pdf_name"])
    cor = _extract_tables(pdfs["cor_pdf_name"])
    if sorted(main) != list(range(1, 9)):
        raise RuntimeError(f"expected models m1..m8, found {sorted(main)}")
    k_check = _check_k_tables(pdfs["pdf_name"])
    models = []
    for i in range(1, 9):
        t = main[i]
        m = np.array(t["coefficients"], dtype=np.int64)
        k_tone_calc = _k_tone(m)
        k_tone_tab, min_erl_tone = G168_K_TONE[i]
        k_css, min_erl_css = G168_K_CSS[i]
        rel = abs(k_tone_calc - k_tone_tab) / k_tone_tab
        clause, origin, desc = G168_MODEL_INFO[i]
        peak = int(np.argmax(np.abs(m)))
        energy = np.cumsum(m.astype(np.float64) ** 2) / np.sum(m.astype(np.float64) ** 2)
        models.append({
            "name": f"m{i}", "clause": clause, "table": t["table"], "printed_page": t["printed_page"],
            "pdf_page_index": t["pdf_page_index"], "origin": origin, "description": desc,
            "length_samples": len(m), "length_ms": len(m) / 8.0,
            "peak_index": peak, "samples_to_99pct_energy": int(np.searchsorted(energy, 0.99)) + 1,
            "K_css": k_css, "min_erl_css_db": min_erl_css, "K_tone": k_tone_tab, "min_erl_tone_db": min_erl_tone,
            "check": {
                "K_tone_recomputed": float(f"{k_tone_calc:.6g}"), "K_tone_rel_error": float(f"{rel:.3g}"),
                "K_tone_matches_table_D1b_to_3_digits": f"{k_tone_calc:.2e}" == f"{k_tone_tab:.2e}",
                "identical_in_cor1_reprint": cor.get(i, {}).get("coefficients") == t["coefficients"],
                "table_layout": f"{t['rows']} rows x {t['columns']} columns, read in columns",
            },
            "coefficients": t["coefficients"],
        })
    result = {
        "source": {k: G168[k] for k in ("recommendation", "title", "edition", "url", "pdf_url", "pdf_sha256",
                                         "cor_edition", "cor_url", "cor_pdf_url", "cor_pdf_sha256")},
        "annex": "Annex D - Echo-path models for testing of speech echo cancellers (integral part of the Recommendation)",
        "sample_rate_hz": 8000,
        "units": "integer sequences m_i(k) as printed; scale with Eq. D-1 before use",
        "equation_D1": "g(k) = (10^(-ERL/20) * K_i) * m_i(k - delta)",
        "usage_notes": [
            "delta (pure echo path delay) and ERL are chosen per test; delta must keep g(k) inside the canceller's H register.",
            "Use K_css (Table D.1a) for CSS or speech-like input, K_tone (Table D.1b) for tones (e.g. Test 10).",
            "Minimum ERL: 6 dB for m1, m3-m6; 6.55 dB (m2), 11.06 dB (m7), 9.27 dB (m8) with K_css, so the scaled |G(f)| stays at or below 0 dB.",
            "Clause D.4: echo paths measured in France have characteristics equivalent to m1, m5 and m6 (no further tables).",
            "Clause 6.4.2.1 / Figure 7: g(k) models the end path including the hybrid and PCM codecs; Test 3/echo-path-change tests use m4, m7, m8 as Echo Path Model B (long dispersion).",
        ],
        "K_table_text_check": k_check,
        "extraction": {
            "tool": "pdfplumber word coordinates (column clustering), validated by recomputing K_tone (Eq. D-2/D-3) "
                    "for every model and comparing with Table D.1b; repeated on the Cor. 1 (12/2022) reprint. "
                    "Also cross-checked once (2026-10-22, by hand) against poppler `pdftotext -layout`: all 8 identical.",
            "all_K_tone_match": all(x["check"]["K_tone_matches_table_D1b_to_3_digits"] for x in models),
            "all_identical_in_cor1": all(x["check"]["identical_in_cor1_reprint"] for x in models),
        },
        "copyright": "ITU-T Recommendations are free to download but (c) ITU, all rights reserved. Keep this file out of git.",
        "models": models,
    }
    write_json(out_dir / "g168_paths.json", result)
    for x in models:
        log(f"[g168] {x['name']}: {x['length_samples']} taps ({x['length_ms']} ms), {x['table']} p.{x['printed_page']}, "
            f"K_tone recomputed {x['check']['K_tone_recomputed']:.6g} vs table {x['K_tone']:.3g} "
            f"(match={x['check']['K_tone_matches_table_D1b_to_3_digits']}), "
            f"cor1 identical={x['check']['identical_in_cor1_reprint']}")
    log(f"[g168] wrote {out_dir / 'g168_paths.json'}; K text check: {k_check.get('match')}")


# --------------------------------------------------------------------------------------------------
# Step: aec - Microsoft AEC Challenge synthetic subset
# --------------------------------------------------------------------------------------------------


def _parse_lfs_pointer(text: str) -> tuple[str, int]:
    m1 = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
    m2 = re.search(r"^size (\d+)$", text, re.M)
    if not text.startswith("version https://git-lfs.github.com/spec/v1") or not m1 or not m2:
        raise RuntimeError("not a git-lfs pointer")
    return m1.group(1), int(m2.group(1))


def step_aec(args: argparse.Namespace) -> None:
    out = args.data / "echo" / "aec_challenge"
    base_raw = f"https://raw.githubusercontent.com/{AEC['repo']}/{AEC['commit']}/datasets/synthetic/"
    base_media = f"https://media.githubusercontent.com/media/{AEC['repo']}/{AEC['commit']}/datasets/synthetic/"
    http = KeepAliveHTTPS()
    meta_bytes = http.get(base_raw + "meta.csv")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.csv").write_bytes(meta_bytes)
    meta = list(csv.DictReader(io.StringIO(meta_bytes.decode("utf-8"))))
    clean = [r for r in meta if r["split"] == "test" and r["is_farend_noisy"] == "0" and r["is_nearend_noisy"] == "0"]
    picked = []
    for nonlinear, n in (("0", AEC["n_linear"]), ("1", AEC["n_nonlinear"])):
        pool = sorted((r for r in clean if r["is_farend_nonlinear"] == nonlinear), key=lambda r: (int(r["ser"]), int(r["fileid"])))
        idx = sorted({round(k * (len(pool) - 1) / (n - 1)) for k in range(n)})
        picked += [pool[k] for k in idx]
    examples = []
    for r in picked:
        fid = int(r["fileid"])
        files = {}
        for kind, pattern in AEC_KINDS.items():
            rel = pattern.format(i=fid)
            dest = out / "synthetic" / rel
            oid, size = _parse_lfs_pointer(http.get(base_raw + rel).decode("utf-8"))
            if not (dest.exists() and sha256_file(dest) == oid):
                data = http.get(base_media + rel)
                if sha256_bytes(data) != oid or len(data) != size:
                    raise RuntimeError(f"{rel}: content does not match its LFS pointer")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            info = sf.info(str(dest))
            files[kind] = {"path": f"echo/aec_challenge/synthetic/{rel}", "sha256": oid, "bytes": size,
                           "sample_rate": info.samplerate, "channels": info.channels, "duration_s": info.frames / info.samplerate}
        examples.append({"fileid": fid, "ser_db": int(r["ser"]), "is_farend_nonlinear": r["is_farend_nonlinear"] == "1",
                         "is_farend_noisy": False, "is_nearend_noisy": False, "nearend_scale": float(r["nearend_scale"]),
                         "farend_speaker": r["farend_speaker"], "nearend_speaker": r["nearend_speaker"], "files": files})
        log(f"[aec] fileid {fid}: ser={r['ser']} nonlinear={r['is_farend_nonlinear']}")
    http.close()
    write_json(out / "subset.json", {
        "source": AEC, "meta_csv_sha256": sha256_bytes(meta_bytes), "meta_csv_git_blob": git_blob_sha1(meta_bytes),
        "selection": "split=test, is_farend_noisy=0, is_nearend_noisy=0; 10 linear + 10 nonlinear far-end examples, "
                     "evenly spaced over SER after sorting by (ser, fileid)",
        "n_examples": len(examples), "bytes_downloaded": http.bytes_in, "examples": examples,
    })
    log(f"[aec] wrote {len(examples)} examples; {http.bytes_in / 1e6:.1f} MB downloaded")


# --------------------------------------------------------------------------------------------------
# Step: summary - results JSON, versioned copies of README and manifest
# --------------------------------------------------------------------------------------------------


def _dur_stats(d: list[float]) -> dict:
    if not d:
        return {}
    q = statistics.quantiles(d, n=4) if len(d) > 1 else [d[0]] * 3
    return {"n": len(d), "total_s": round(sum(d), 2), "min_s": round(min(d), 2), "p25_s": round(q[0], 2),
            "median_s": round(q[1], 2), "mean_s": round(statistics.mean(d), 2), "p75_s": round(q[2], 2),
            "max_s": round(max(d), 2)}


def step_summary(args: argparse.Namespace) -> None:
    import pyarrow
    import pdfplumber
    import huggingface_hub

    data, results = args.data, args.results
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        for key in ("path_16k", "path_tel_16k", "path_8k_ulaw", "path_8k_alaw"):
            if key in r and not (data / r[key]).is_file():
                raise RuntimeError(f"{r['id']}: missing {r[key]}")
    sets = {}
    for s in ("zh", "en", "mix"):
        rs = [r for r in rows if r["set"] == s]
        sets[s] = {
            "count": len(rs), "duration": _dur_stats([r["duration_s"] for r in rs]),
            "has_number": sum(r["has_number"] for r in rs),
            "codec_8k": {c: sum(r["codec_8k"] == c for r in rs) for c in ("ulaw", "alaw")},
            "speakers": dict(sorted({sp: sum(r["speaker"] == sp for r in rs) for sp in {r["speaker"] for r in rs}}.items())),
            "source": rs[0]["source"] if rs else None, "source_repo": rs[0]["source_repo"] if rs else None,
            "config": rs[0]["source_config"] if rs else None, "split": rs[0]["split"] if rs else None,
            "revision": rs[0]["revision"] if rs else None, "license": sorted({r["license"] for r in rs}),
            "url": rs[0]["url"] if rs else None,
            "level": {
                "target_active_dbov": TARGET_LEVEL_DBOV,
                "active_dbov_min": min(r["active_level_dbov"] for r in rs),
                "active_dbov_median": statistics.median(r["active_level_dbov"] for r in rs),
                "active_dbov_max": max(r["active_level_dbov"] for r in rs),
                "gain_db_min": min(r["gain_db"] for r in rs), "gain_db_max": max(r["gain_db"] for r in rs),
                "peak_limited": sum(r["peak_limited"] for r in rs),
                "source_active_dbov_min": min(r["source_active_level_dbov"] for r in rs),
                "source_active_dbov_median": statistics.median(r["source_active_level_dbov"] for r in rs),
                "source_active_dbov_max": max(r["source_active_level_dbov"] for r in rs),
            },
            "clipped_samples_total": sum(sum(r["clipped_samples"].values()) for r in rs),
        }
    fleurs_sel = {lang: read_json(data / "raw" / "fleurs" / lang / "selection.json") for lang in FLEURS_LANGS.values()}
    ascend_sel = read_json(data / "raw" / "ascend" / "selection.json")
    g168_path = data / "echo" / "g168_paths.json"
    aec_path = data / "echo" / "aec_challenge" / "subset.json"
    echo = {}
    if g168_path.exists():
        g = read_json(g168_path)
        echo["g168"] = {
            "file": "spikes/data/echo/g168_paths.json", "edition": g["source"]["edition"],
            "also_checked": g["source"]["cor_edition"], "annex": g["annex"], "sample_rate_hz": g["sample_rate_hz"],
            "n_models": len(g["models"]),
            "models": [{k: m[k] for k in ("name", "table", "printed_page", "origin", "length_samples", "length_ms",
                                          "K_css", "min_erl_css_db", "K_tone")} | {"K_tone_check_ok": m["check"]["K_tone_matches_table_D1b_to_3_digits"]}
                       for m in g["models"]],
            "extraction": g["extraction"],
        }
    if aec_path.exists():
        a = read_json(aec_path)
        echo["aec_challenge"] = {
            "dir": "spikes/data/echo/aec_challenge/", "n_examples": a["n_examples"],
            "files_per_example": list(AEC_KINDS), "sample_rate_hz": 16000,
            "duration_s_each": 10.0, "linear": sum(not e["is_farend_nonlinear"] for e in a["examples"]),
            "nonlinear": sum(e["is_farend_nonlinear"] for e in a["examples"]),
            "ser_db": sorted(e["ser_db"] for e in a["examples"]),
            "repo": AEC["repo"], "commit": AEC["commit"], "code_license": AEC["code_license"],
            "data_license": AEC["data_license"], "bytes_downloaded": a["bytes_downloaded"],
        }
    summary = {
        "work_package": "M0 A2 - test audio, telephone copies, manifest, echo material",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "spikes/datasets/a2_data.py",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "soxr": soxr.__version__,
            "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__, "pyarrow": pyarrow.__version__,
            "pdfplumber": pdfplumber.__version__, "huggingface_hub": huggingface_hub.__version__,
        },
        "manifest": {"path": "spikes/data/manifest.jsonl", "versioned_copy": "spikes/datasets/manifest.jsonl",
                     "rows": len(rows)},
        "sets": sets,
        "total": {"count": len(rows), "duration": _dur_stats([r["duration_s"] for r in rows]),
                  "codec_8k": {c: sum(r["codec_8k"] == c for r in rows) for c in ("ulaw", "alaw")}},
        "level_normalization": (
            f"Each utterance gets one gain, applied to the original samples before PCM16 conversion: active speech "
            f"level (ITU-T P.56 method B, sv56-style) to {TARGET_LEVEL_DBOV} dBov, limited so the peak stays at or below "
            f"{PEAK_LIMIT_DBFS} dBFS. Both the 16 kHz and the telephone copy carry it; originals stay in raw/."
        ),
        "telephone_copy": {
            "chain": "16 kHz PCM16 (level-normalized) -> soxr.resample(16000->8000, quality='HQ') -> round/clip to int16 -> G.711 "
                     "encode (u-law, or A-law for every 3rd utterance per set) -> decode -> soxr.resample(8000->16000, 'HQ') -> int16",
            "g711_implementation": "Python 3.12 stdlib audioop (lin2ulaw/ulaw2lin, lin2alaw/alaw2lin, width 2); "
                                   "deprecated since 3.11 (PEP 594) and removed in 3.13",
            "no_telephone_bandpass": "No 300-3400 Hz (G.712) filter: band-limiting comes only from the 8 kHz resampler",
            "byte_streams": "asr/tel8k/<id>.ulaw|.alaw: headerless G.711, 8000 Hz, 1 byte per sample",
        },
        "downloads": {
            "fleurs": {lang: {"archive_bytes": s["archive"]["size"],
                              "compressed_bytes_read": s["archive_prefix"]["compressed_bytes_read"],
                              "members_read": s["archive_prefix"]["members_read"],
                              "tsv_bytes": s["tsv"]["size"]} for lang, s in fleurs_sel.items()},
            "ascend": {"parquet_bytes": ascend_sel["parquet"]["size"], "row_groups_read": ascend_sel["row_groups_read"],
                       "audio_column_bytes_read": ascend_sel["audio_column_bytes_read"]},
        },
        "licenses": {
            "FLEURS": f"{FLEURS['license']} ({FLEURS['url']}); attribution required",
            "ASCEND": f"{ASCEND['license']} ({ASCEND['url']}); attribution + share-alike: derived audio (telephone copies) "
                      "may only be redistributed under CC-BY-SA-4.0",
            "ITU-T G.168": "free download, (c) ITU all rights reserved; coefficients kept in git-ignored spikes/data",
            "AEC-Challenge": f"code {AEC['code_license']}; data: {AEC['data_license']}",
            "non_commercial_or_nd_used": False,
        },
        "echo": echo,
        "caveats": [
            f"FLEURS test has no clip under 3.6 s; the shortest zh/en picks are {sets['zh']['duration']['min_s']} s / "
            f"{sets['en']['duration']['min_s']} s. Short utterances come from ASCEND (min {sets['mix']['duration']['min_s']} s).",
            f"ASCEND's test split has only two speakers: {sets['mix']['speakers']}.",
            f"FLEURS en in-range utterances are mostly female; selected genders: {sets['en']['speakers']}.",
            "text_norm does not verbalize numbers (FLEURS zh writes '1990 年'); score has_number rows separately or apply ITN.",
            "ASCEND references keep fillers (呃, 嗯, um, uh) and letter-by-letter spellings.",
            "Telephone copies have no 300-3400 Hz G.712 filter; only the 8 kHz resampler band-limits them.",
            f"Source levels differ by about 50 dB (P.56 active level median: zh {sets['zh']['level']['source_active_dbov_median']}, "
            f"en {sets['en']['level']['source_active_dbov_median']}, mix {sets['mix']['level']['source_active_dbov_median']} dBov); "
            f"all copies are normalized to {TARGET_LEVEL_DBOV} dBov (gain_db per row). Use raw/ for source levels.",
            "AEC Challenge material is acoustic (loudspeaker-to-mic) echo of English LibriVox speech, not line echo.",
        ],
    }
    write_json(results / "a2_datasets.json", summary)
    here = Path(__file__).resolve().parent
    shutil.copyfile(data / "manifest.jsonl", here / "manifest.jsonl")
    readme_src = here / "MANIFEST.md"
    if readme_src.exists():
        shutil.copyfile(readme_src, data / "README.md")
    log(f"[summary] {len(rows)} rows; wrote {results / 'a2_datasets.json'}, copied manifest and README")


# --------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fleurs", "ascend", "build", "g168", "aec", "summary"])
    ap.add_argument("--data", type=Path, required=True, help="spikes/data directory (git-ignored)")
    ap.add_argument("--results", type=Path, default=None, help="spikes/results directory (summary step)")
    ap.add_argument("--force", action="store_true", help="redo a fetch step even if its selection exists")
    args = ap.parse_args()
    args.data = args.data.resolve()
    if args.results is not None:
        args.results = args.results.resolve()
    if args.step == "summary" and args.results is None:
        ap.error("summary needs --results")
    {"fleurs": step_fleurs, "ascend": step_ascend, "build": step_build, "g168": step_g168,
     "aec": step_aec, "summary": step_summary}[args.step](args)


if __name__ == "__main__":
    main()

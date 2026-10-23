"""S7: fetch only the Chinese rows of HF pipecat-ai/smart-turn-data-v3.2-test (CC BY 4.0).

    uv run python fetch_testdata.py [--verify-rowgroup]

Why this route: the dataset is 10 parquet shards (4.8 GB). Rows are ordered by id, so the 929
`language == "zho"` rows are spread over every 100-row group, and each group stores its audio
as one dictionary page (~15 MB). Reading only Chinese rows from parquet is therefore impossible;
whole shards would be ~480 MB for ~93 Chinese rows each.

Instead:
1. Range-read each shard's footer and the small metadata columns of every row group
   (id, language, endpoint_bool, midfiller, endfiller, synthetic, dataset) — about 1 KB per group.
2. For the Chinese rows, ask the dataset viewer API (/rows) for the row's audio asset, which is
   served as FLAC (lossless), and download it.
3. --verify-rowgroup: download one row group's audio column from the parquet itself (~15 MB) and
   check that every Chinese asset in it decodes to exactly the same samples as the original.

Output (untrusted data, git-ignored): data/smart-turn-data-v3.2-test/zho/<row_idx>.flac and
zho_manifest.json beside it. A copy of the manifest goes to spikes/results/s7_testdata.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEST_SET_DIR, save_json, sha256_file  # noqa: E402

REPO = "pipecat-ai/smart-turn-data-v3.2-test"
REVISION = "dfc179241fee65b0efe81ca37786f41ce26f38ec"
N_SHARDS = 10
SMALL_COLS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset"]
LANG = "zho"
OUT = TEST_SET_DIR / "zho"
MANIFEST = TEST_SET_DIR / "zho_manifest.json"

_tls = threading.local()
_stats = {"requests": 0, "bytes": 0}
_stats_lock = threading.Lock()


def client():
    """One keep-alive HTTP client per thread (huggingface_hub's httpx client class)."""
    import httpx2

    c = getattr(_tls, "client", None)
    if c is None:
        c = _tls.client = httpx2.Client(timeout=httpx2.Timeout(180.0, connect=60.0), follow_redirects=False)
    return c


def http_get(url: str, headers: dict | None = None, retries: int = 12) -> tuple[int, dict, bytes]:
    """GET with retries; HTTP 429 honours Retry-After (the dataset viewer API rate-limits)."""
    for attempt in range(retries):
        try:
            r = client().get(url, headers=headers or {})
            with _stats_lock:
                _stats["requests"] += 1
                _stats["bytes"] += len(r.content)
            if r.status_code == 429:
                ra = r.headers.get("retry-after", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else min(120.0, 5.0 * 2**attempt)
                print(f"  429 rate-limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.status_code, dict(r.headers), r.content
        except Exception as e:  # noqa: BLE001 - spike: retry anything transient
            if attempt == retries - 1:
                raise
            wait = 2 * (attempt + 1)
            print(f"  retry {attempt + 1} in {wait}s after {type(e).__name__}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up after {retries} attempts: {url.split('?')[0]}")


def viewer_headers() -> dict:
    """Authenticated requests get a higher dataset-viewer rate limit. The token is never printed."""
    from huggingface_hub.utils import build_hf_headers

    return build_hf_headers()


def resolve(shard: int) -> tuple[str, int]:
    """Signed CDN URL and size for a shard (one redirect from the pinned resolve URL)."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    status, headers, _ = http_get(url, {"Range": "bytes=0-0"})
    if status in (301, 302, 303, 307, 308):
        url = headers.get("location") or headers.get("Location")
        status, headers, _ = http_get(url, {"Range": "bytes=0-0"})
    if status != 206:
        raise RuntimeError(f"shard {shard}: range probe got HTTP {status}")
    size = int((headers.get("content-range") or headers["Content-Range"]).split("/")[1])
    return url, size


class RemoteFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests (exact ranges, small cache)."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.cache: dict[tuple[int, int], bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        start, end = self.pos, self.pos + n - 1
        for (a, b), data in self.cache.items():
            if a <= start and end <= b:
                self.pos += n
                return data[start - a : end - a + 1]
        status, _, data = http_get(self.url, {"Range": f"bytes={start}-{end}"})
        if status != 206 or len(data) != n:
            raise RuntimeError(f"range {start}-{end}: HTTP {status}, {len(data)} bytes")
        if n <= (1 << 20):
            self.cache[(start, end)] = data
        self.pos += n
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def scan_shard(shard: int) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    url, size = resolve(shard)
    pf = pq.ParquetFile(pa.PythonFile(RemoteFile(url, size), mode="r"), pre_buffer=True)
    md = pf.metadata
    rows, offset = [], 0
    for g in range(md.num_row_groups):
        t = pf.read_row_group(g, columns=SMALL_COLS).to_pylist()
        for k, r in enumerate(t):
            rows.append({**r, "shard": shard, "row_group": g, "row_in_group": k, "row_in_shard": offset + k})
        offset += len(t)
    print(f"shard {shard}: {md.num_rows} rows, {md.num_row_groups} groups, "
          f"{sum(r['language'] == LANG for r in rows)} {LANG}", flush=True)
    return {"shard": shard, "url": url, "size": size, "num_rows": md.num_rows, "rows": rows}


def rows_page(offset: int, length: int = 100) -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={REPO}&config=default&split=train&offset={offset}&length={length}"
    )
    status, _, body = http_get(url, viewer_headers())
    if status != 200:
        raise RuntimeError(f"/rows offset {offset}: HTTP {status}")
    return json.loads(body)["rows"]


def fetch_page_assets(page_offset: int, wanted: dict[int, dict]) -> list[dict]:
    """Download the FLAC assets of the wanted rows in one /rows page (skips pages already on disk)."""
    if all((OUT / f"{idx}.flac").exists() for idx in wanted):
        return [{**m, "row_idx": idx, "file": f"{idx}.flac", "asset_ext": "flac"} for idx, m in wanted.items()]
    got = []
    for r in rows_page(page_offset):
        idx = r["row_idx"]
        if idx not in wanted:
            continue
        meta = wanted[idx]
        if r["row"]["id"] != meta["id"]:
            raise RuntimeError(f"row {idx}: viewer id {r['row']['id']} != parquet id {meta['id']}")
        src = r["row"]["audio"][0]["src"]
        if f"/--/{REVISION}/--/" not in src:
            raise RuntimeError(f"row {idx}: asset is not from the pinned revision: {src.split('?')[0]}")
        dst = OUT / f"{idx}.flac"
        if not dst.exists():
            status, _, data = http_get(src)
            if status != 200:
                raise RuntimeError(f"row {idx}: asset HTTP {status}")
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dst)
        got.append({**meta, "row_idx": idx, "file": dst.name, "asset_ext": src.split("?")[0].rsplit(".", 1)[-1]})
    return got


def verify_rowgroup(shard_info: dict, manifest_rows: list[dict]) -> dict:
    """Compare the viewer assets with the original parquet audio for one row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    by_group: dict[int, list[dict]] = {}
    for m in manifest_rows:
        if m["shard"] == shard_info["shard"]:
            by_group.setdefault(m["row_group"], []).append(m)
    url, size = resolve(shard_info["shard"])  # signed URLs expire; resolve again
    pf0 = pq.ParquetFile(pa.PythonFile(RemoteFile(url, size), mode="r"))
    # The smallest group that holds at least two Chinese rows.
    sizes = {g: pf0.metadata.row_group(g).total_byte_size for g in by_group if len(by_group[g]) >= 2}
    g = min(sizes, key=sizes.get)
    t = pf0.read_row_group(g, columns=["audio", "id"]).to_pylist()
    checks = []
    for m in by_group[g]:
        orig = t[m["row_in_group"]]
        assert orig["id"] == m["id"]
        a, sra = sf.read(io.BytesIO(orig["audio"]["bytes"]), dtype="int16", always_2d=True)
        b, srb = sf.read(OUT / m["file"], dtype="int16", always_2d=True)
        checks.append({
            "row_idx": m["row_idx"],
            "same_bytes": orig["audio"]["bytes"] == (OUT / m["file"]).read_bytes(),
            "same_samples": bool(sra == srb and a.shape == b.shape and np.array_equal(a, b)),
            "sr": sra,
            "frames": int(a.shape[0]),
        })
    return {"shard": shard_info["shard"], "row_group": g, "group_bytes": sizes[g], "rows": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-rowgroup", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="parallel shard scans")
    ap.add_argument("--page-workers", type=int, default=3, help="parallel dataset-viewer pages")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keep_out = TEST_SET_DIR.parent / ".gitignore"  # downloaded data never goes into git
    if not keep_out.exists():
        keep_out.write_text("# Downloaded, untrusted test data (S7). Never commit.\n*\n")
    t0 = time.time()

    scan_cache = TEST_SET_DIR / "scan_cache.json"
    if scan_cache.exists():
        shards = json.loads(scan_cache.read_text())
    else:
        with cf.ThreadPoolExecutor(args.workers) as ex:
            shards = list(ex.map(scan_shard, range(N_SHARDS)))
        shards.sort(key=lambda s: s["shard"])
        scan_cache.write_text(json.dumps(shards))
    base, wanted, langs = 0, {}, {}
    for s in shards:
        for r in s["rows"]:
            idx = base + r["row_in_shard"]
            langs[r["language"]] = langs.get(r["language"], 0) + 1
            if r["language"] == LANG:
                wanted[idx] = r
        base += s["num_rows"]
    print(f"scan: {base} rows, {len(wanted)} {LANG}; {_stats['requests']} requests, "
          f"{_stats['bytes'] / 1e6:.2f} MB, {time.time() - t0:.0f}s", flush=True)

    pages = sorted({idx // 100 * 100 for idx in wanted})
    rows = []
    with cf.ThreadPoolExecutor(args.page_workers) as ex:
        futs = [ex.submit(fetch_page_assets, p, {i: wanted[i] for i in wanted if i // 100 * 100 == p}) for p in pages]
        for n, f in enumerate(cf.as_completed(futs), 1):
            rows.extend(f.result())
            if n % 20 == 0:
                print(f"  pages {n}/{len(pages)}, rows {len(rows)}, {_stats['bytes'] / 1e6:.1f} MB", flush=True)
    rows.sort(key=lambda r: r["row_idx"])
    for r in rows:
        r["sha256"] = sha256_file(OUT / r["file"])

    verification = None
    if args.verify_rowgroup:
        verification = verify_rowgroup(shards[0], rows)

    manifest = {
        "source": f"huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "license": "CC-BY-4.0 (dataset card and LICENSE file); background noise samples CC0 (freesound)",
        "language_field": "language (ISO 639-3); Chinese = 'zho'",
        "label_field": "endpoint_bool (True = turn complete)",
        "language_counts": dict(sorted(langs.items())),
        "rows": rows,
        "n_rows": len(rows),
        "positives": sum(bool(r["endpoint_bool"]) for r in rows),
        "audio_bytes": sum((OUT / r["file"]).stat().st_size for r in rows),
        "download": {"requests": _stats["requests"], "bytes": _stats["bytes"], "seconds": round(time.time() - t0)},
        "route": "parquet metadata columns by HTTP range + dataset-viewer /rows FLAC assets",
        "verification": verification,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    save_json("s7_testdata.json", manifest)
    print(f"done: {len(rows)} rows, {manifest['audio_bytes'] / 1e6:.1f} MB audio, "
          f"{_stats['bytes'] / 1e6:.1f} MB transferred in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

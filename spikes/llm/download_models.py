"""M0 / S0 (work package A1): fetch the M0 model repos and write spikes/results/models.json.

Run from the repository root:

    uv run --project spikes/llm python spikes/llm/download_models.py                    # essentials
    uv run --project spikes/llm python spikes/llm/download_models.py --include-deferred # everything
    uv run --project spikes/llm python spikes/llm/download_models.py --only <repo> ...  # update entries
    ... --no-xet   # HF_HUB_DISABLE_XET=1: same Hub/CDN, plain HTTP that resumes after drops in-process

Each downloaded repo is fetched into the default Hugging Face cache with exactly

    uv run --project spikes/llm hf download <repo>

retried on failure (hf resumes partial files and waits on the per-blob locks of any other hf process
fetching the same repo). The manifest records per repo: the commit SHA the Hub resolves `main` to, the
SHA the local cache ref points at, every file with its size (and LFS sha256), the total bytes, the wall
time of the download command, and whether the local snapshot matches the Hub file list exactly.

Bandwidth decision (orchestrator, 2026-10-22): the HF route through the local proxy runs at
~0.1-2.5 MB/s, so only the ESSENTIAL repos are downloaded by default. DEFERRED repos are recorded with
their Hub SHA and expected file list only, status "deferred (bandwidth)".

The manifest is rewritten after every repo, so it always shows the current state
(ok / in_progress / pending / deferred (bandwidth) / not_found / failed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, __version__ as hf_version, constants
from huggingface_hub.errors import RepositoryNotFoundError

ESSENTIAL = [
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",  # A1 audio smoke
    "mlx-community/Qwen3-ASR-1.7B-8bit",  # A1 audio smoke
    "mlx-community/Kokoro-82M-bf16",  # A1 audio smoke (TTS fallback)
    "mlx-community/Qwen3.6-35B-A3B-4bit",  # A1 LLM smoke, G2 first choice
]
DEFERRED = [
    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
]
DEFERRED_STATUS = "deferred (bandwidth)"

SPIKE_DIR = Path(__file__).resolve().parent
RESULTS = SPIKE_DIR.parent / "results"
MANIFEST = RESULTS / "models.json"
CACHE = Path(constants.HF_HUB_CACHE)
MAX_ATTEMPTS = 6


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def hub_info(api: HfApi, repo: str):
    """model_info with file metadata; retried, since the network is shared with other downloads."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return api.model_info(repo, files_metadata=True, timeout=60)
        except RepositoryNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - transient network errors
            log(f"{repo}: model_info attempt {attempt} failed: {exc!r}")
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"{repo}: model_info failed {MAX_ATTEMPTS} times")


def repo_dir(repo: str) -> Path:
    return CACHE / ("models--" + repo.replace("/", "--"))


def local_snapshot(repo: str) -> tuple[str | None, Path | None]:
    ref = repo_dir(repo) / "refs" / "main"
    if not ref.is_file():
        return None, None
    sha = ref.read_text().strip()
    return sha, repo_dir(repo) / "snapshots" / sha


def local_files(snapshot: Path) -> dict[str, int]:
    """Relative path -> size of the blob each snapshot symlink points at."""
    out: dict[str, int] = {}
    for p in sorted(snapshot.rglob("*")):
        if p.is_file():  # follows the symlink into blobs/
            out[p.relative_to(snapshot).as_posix()] = p.stat().st_size
    return out


def bytes_in_cache(repo: str) -> int:
    """All blob bytes already on disk for this repo, partial (.incomplete) files included."""
    blobs = repo_dir(repo) / "blobs"
    return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file()) if blobs.is_dir() else 0


def hub_fields(info) -> dict:
    files = sorted(info.siblings or [], key=lambda s: s.rfilename)
    total = sum(s.size or 0 for s in files)
    return {
        "sha_hub": info.sha,
        "last_modified_hub": info.last_modified.isoformat() if info.last_modified else None,
        "expected_total_bytes": total,
        "expected_total_gb": round(total / 1e9, 3),
        "expected_n_files": len(files),
    }


def file_list(sizes: dict[str, int], hub_files: dict) -> list[dict]:
    return [
        {
            "path": path,
            "size": size,
            "lfs_sha256": hub_files[path].lfs.sha256 if path in hub_files and hub_files[path].lfs else None,
        }
        for path, size in sizes.items()
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_download(repo: str, no_xet: bool) -> tuple[bool, int, float, str, str]:
    """Run the exact CLI command; return (ok, attempts, seconds, cli_path, error_tail)."""
    cmd = ["uv", "run", "--project", str(SPIKE_DIR), "hf", "download", repo]
    env = dict(os.environ)
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if no_xet:
        # Same Hub and CDN, plain HTTP transport: http_get resumes with a Range request after a
        # dropped connection inside the process, while a failed Xet reconstruction starts over.
        env["HF_HUB_DISABLE_XET"] = "1"
    t0 = time.perf_counter()
    err_tail = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        log(f"{repo}: uv run --project spikes/llm hf download (attempt {attempt})")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            return True, attempt, time.perf_counter() - t0, (lines[-1] if lines else ""), ""
        err_tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-5:])
        log(f"{repo}: attempt {attempt} rc={proc.returncode}: {err_tail}")
        time.sleep(min(60, 10 * attempt))
    return False, MAX_ATTEMPTS, time.perf_counter() - t0, "", err_tail


def describe_deferred(api: HfApi, repo: str) -> dict:
    entry: dict = {"repo": repo, "status": DEFERRED_STATUS}
    try:
        info = hub_info(api, repo)
    except RepositoryNotFoundError as exc:
        return entry | {"status": "not_found", "error": str(exc).splitlines()[0]}
    hub_files = {s.rfilename: s for s in info.siblings or []}
    entry |= hub_fields(info)
    entry["bytes_in_cache"] = bytes_in_cache(repo)
    entry["files_hub"] = file_list({k: s.size for k, s in sorted(hub_files.items())}, hub_files)
    return entry


DOWNLOAD_KEYS = ("download_started_at", "download_finished_at", "download_seconds", "download_attempts",
                 "bytes_in_cache_before_run", "transport")


def fetch(api: HfApi, repo: str, on_start, previous: dict | None, no_xet: bool) -> dict:
    entry: dict = {"repo": repo}
    try:
        info = hub_info(api, repo)
    except RepositoryNotFoundError as exc:
        log(f"{repo}: NOT FOUND")
        return entry | {"status": "not_found", "error": str(exc).splitlines()[0]}
    hub_files = {s.rfilename: s for s in info.siblings or []}
    expected = {k: s.size for k, s in hub_files.items()}
    entry |= hub_fields(info)

    pre_sha, pre_snap = local_snapshot(repo)
    entry["already_cached_before_run"] = bool(
        pre_sha == info.sha and pre_snap is not None and pre_snap.is_dir() and local_files(pre_snap) == expected
    )
    # Keep what earlier runs of this script recorded (a real first download, or unfinished attempts).
    previous = previous or {}
    if previous.get("first_download"):
        entry["first_download"] = previous["first_download"]
    elif previous.get("status") == "ok" and not previous.get("already_cached_before_run"):
        entry["first_download"] = {k: previous.get(k) for k in DOWNLOAD_KEYS} | {
            "transport": previous.get("transport", "xet (hf_xet)")
        }
    if previous.get("earlier_attempts"):
        entry["earlier_attempts"] = previous["earlier_attempts"]
    elif previous.get("status") in ("in_progress", "failed"):
        entry["earlier_attempts"] = {
            k: previous[k] for k in ("status", "download_started_at", "transport", "error") if k in previous
        }
    entry["transport"] = "http (HF_HUB_DISABLE_XET=1)" if no_xet else "xet (hf_xet)"
    entry["bytes_in_cache_before_run"] = bytes_in_cache(repo)
    entry["download_started_at"] = now_iso()
    on_start(entry | {"status": "in_progress"})

    ok, attempts, seconds, cli_path, err = run_download(repo, no_xet)
    entry |= {
        "download_finished_at": now_iso(),
        "download_attempts": attempts,
        "download_seconds": round(seconds, 1),
    }
    if not ok:
        return entry | {"status": "failed", "error": err}

    sha, snap = local_snapshot(repo)
    files = local_files(snap) if snap and snap.is_dir() else {}
    total = sum(files.values())
    entry |= {
        "status": "ok",
        "sha": sha,
        "sha_cli_path": Path(cli_path).name if cli_path else None,
        "sha_matches_hub": sha == info.sha,
        "complete": files == expected,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "n_files": len(files),
        "snapshot": str(snap),
        "files": file_list(files, hub_files),
    }
    if files != expected:
        entry["mismatch"] = {
            "missing": sorted(set(expected) - set(files)),
            "extra": sorted(set(files) - set(expected)),
            "size_differs": sorted(k for k in set(files) & set(expected) if files[k] != expected[k]),
        }
    # Integrity: hash every LFS file and compare with the sha256 the Hub lists for it.
    bad = []
    for f in entry["files"]:
        if f["lfs_sha256"]:
            f["sha256_ok"] = sha256_file(snap / f["path"]) == f["lfs_sha256"]
            if not f["sha256_ok"]:
                bad.append(f["path"])
    entry["lfs_sha256_verified"] = not bad
    if bad:
        entry["lfs_sha256_mismatch"] = bad
    log(f"{repo}: ok sha={sha} {total / 1e9:.2f} GB in {seconds:.0f}s complete={files == expected} "
        f"sha256_ok={not bad}")
    return entry


def write_manifest(entries: dict[str, dict]) -> None:
    order = ESSENTIAL + DEFERRED
    manifest = {
        "generated_at": now_iso(),
        "command": "uv run --project spikes/llm hf download <repo>",
        "huggingface_hub": hf_version,
        "hf_hub_cache": str(CACHE),
        "bandwidth_decision": (
            "HF route through the local proxy measured at ~0.1-2.5 MB/s on 2026-10-22; the orchestrator "
            "limited downloads to the essential repos. Deferred repos list Hub metadata only."
        ),
        "note": (
            "sha = commit the local cache ref 'main' points at after the download; sha_hub = what the Hub "
            "resolved 'main' to just before it. download_seconds is the wall time of this run's hf download "
            "on a link shared with other downloads; bytes_in_cache_before_run counts blob bytes (partial "
            "files included) that an earlier or concurrent hf process had already fetched. first_download "
            "keeps the timing of an earlier run that did the real download; earlier_attempts records a run "
            "that never finished. transport: hf_xet (default) or plain HTTP (HF_HUB_DISABLE_XET=1), both "
            "from huggingface.co and its CDN. lfs_sha256_verified: every LFS file hashed and compared "
            "with the Hub's sha256."
        ),
        "models": [entries[r] for r in order if r in entries]
        + [entries[r] for r in entries if r not in order],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-deferred", action="store_true", help="also download the deferred repos")
    parser.add_argument("--only", nargs="+", metavar="REPO",
                        help="download only these repos and keep the other entries of the existing manifest")
    parser.add_argument("--no-xet", action="store_true",
                        help="set HF_HUB_DISABLE_XET=1 for hf download (plain HTTP with in-process resume)")
    args = parser.parse_args()

    api = HfApi()
    previous: dict[str, dict] = (
        {m["repo"]: m for m in json.loads(MANIFEST.read_text())["models"]} if MANIFEST.is_file() else {}
    )
    if args.only:
        to_download = args.only
        entries: dict[str, dict] = dict(previous)
        keep = ("sha_hub", "last_modified_hub", "expected_total_bytes", "expected_total_gb", "expected_n_files")
        entries |= {
            r: {"repo": r, "status": "pending"} | {k: previous[r][k] for k in keep if k in previous.get(r, {})}
            for r in to_download
        }
    else:
        to_download = ESSENTIAL + (DEFERRED if args.include_deferred else [])
        entries = {r: {"repo": r, "status": "pending"} for r in to_download}
        if not args.include_deferred:
            for repo in DEFERRED:
                entries[repo] = describe_deferred(api, repo)
                log(f"{repo}: {entries[repo]['status']}")

    def on_start(entry: dict) -> None:
        entries[entry["repo"]] = entry
        write_manifest(entries)

    for repo in to_download:
        entries[repo] = fetch(api, repo, on_start, previous.get(repo), args.no_xet)
        write_manifest(entries)
    log(f"done; wrote {MANIFEST}")


if __name__ == "__main__":
    main()

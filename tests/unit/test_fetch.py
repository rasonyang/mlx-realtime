"""``mlxrt models``: the committed manifest, download and verification, the token rule, the MLX
prefetch through a backend's ``hf`` CLI, and the Hugging Face cache layout."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.fetch import (
    FetchError,
    fetch_files,
    fetch_mlx,
    main_ref_mismatch,
    read_manifest,
)
from mlx_realtime.fetch import _headers as auth_headers  # pyright: ignore[reportPrivateUsage]
from mlx_realtime.hub import hub_cache, snapshot_dir

REPO = Path(__file__).parents[2]
SPIKE_MANIFEST = REPO / "spikes" / "cpu" / "models" / "MANIFEST.json"


def test_the_committed_manifest_pins_the_m0_models() -> None:
    entries = {entry.file: entry for entry in read_manifest(REPO / "models")}
    assert set(entries) == {
        "silero_vad.onnx",
        "smart-turn-v3.2-cpu.onnx",
        "smart-turn-v3.2-gpu.onnx",
    }
    assert "/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/" in (
        entries["silero_vad.onnx"].url
    )
    for name in ("smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2-gpu.onnx"):
        assert entries[name].url.startswith(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d"
        )
    if SPIKE_MANIFEST.is_file():  # the hashes were copied from the M0 spike
        spike = json.loads(SPIKE_MANIFEST.read_text())["models"]
        assert {m["file"]: m["sha256"] for m in spike} == {
            name: entry.sha256 for name, entry in entries.items()
        }


@contextmanager
def file_server(files: dict[str, bytes], seen: list[dict[str, str]]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            seen.append(dict(self.headers))
            body = files.get(self.path.lstrip("/"))
            self.send_response(200 if body is not None else 404)
            self.end_headers()
            self.wfile.write(body or b"")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def manifest(directory: Path, base: str, files: dict[str, bytes], *, wrong: str = "") -> None:
    models = [
        {
            "file": name,
            "url": f"{base}/{name}",
            "bytes": len(data),
            "sha256": "0" * 64 if name == wrong else hashlib.sha256(data).hexdigest(),
        }
        for name, data in files.items()
    ]
    (directory / "MANIFEST.json").write_text(json.dumps({"models": models}))


def test_files_are_downloaded_once_and_verified(tmp_path: Path) -> None:
    files = {"a.onnx": b"model a", "b.onnx": b"model b" * 1000}
    seen: list[dict[str, str]] = []
    reports: list[str] = []
    with file_server(files, seen) as base:
        manifest(tmp_path, base, files)
        fetch_files(tmp_path, {"HF_TOKEN": "hf_secret"}, reports.append)
        assert (tmp_path / "b.onnx").read_bytes() == files["b.onnx"]
        (tmp_path / "a.onnx").write_bytes(b"tampered")
        fetch_files(tmp_path, {}, reports.append)
    assert reports == [
        "a.onnx: downloading 0.0 MB",
        "a.onnx: downloaded and verified",
        "b.onnx: downloading 0.0 MB",
        "b.onnx: downloaded and verified",
        "a.onnx: downloading 0.0 MB",  # the tampered copy is replaced
        "a.onnx: downloaded and verified",
        "b.onnx: verified",
    ]
    assert all("Authorization" not in headers for headers in seen)  # not huggingface.co
    assert not any("secret" in report for report in reports)


def test_a_wrong_hash_leaves_no_file(tmp_path: Path) -> None:
    files = {"a.onnx": b"model a"}
    with file_server(files, []) as base:
        manifest(tmp_path, base, files, wrong="a.onnx")
        with pytest.raises(FetchError, match="sha256"):
            fetch_files(tmp_path, {}, lambda _: None)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["MANIFEST.json"]


def test_the_token_goes_to_huggingface_only() -> None:
    environ = {"HF_TOKEN": "hf_x"}
    assert auth_headers("https://huggingface.co/x/resolve/r/f.onnx", environ) == {
        "Authorization": "Bearer hf_x"
    }
    assert auth_headers("https://raw.githubusercontent.com/x", environ) == {}
    assert auth_headers("https://huggingface.co.evil.example/x", environ) == {}
    assert auth_headers("https://huggingface.co/x", {}) == {}


def test_hub_cache_follows_the_huggingface_environment(tmp_path: Path) -> None:
    assert hub_cache({"HF_HUB_CACHE": str(tmp_path)}) == tmp_path
    assert hub_cache({"HF_HOME": str(tmp_path)}) == tmp_path / "hub"
    assert hub_cache({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "huggingface" / "hub"
    assert hub_cache({}) == Path.home() / ".cache" / "huggingface" / "hub"


FAKE_UV = """\
import os, pathlib, sys
args = sys.argv[1:]
assert args[:4] == ["run", "--locked", "--project", args[3]] and args[4:6] == ["hf", "download"]
repo, revision = args[6], args[8]
assert "HF_HUB_OFFLINE" not in os.environ
cache = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + repo.replace("/", "--"))
snapshot = cache / "snapshots" / "c0ffee" / "config.json"
snapshot.parent.mkdir(parents=True, exist_ok=True)
snapshot.write_text("{}")
(cache / "refs").mkdir(exist_ok=True)
(cache / "refs" / revision).write_text("c0ffee")
with open(os.environ["CALLS"], "a") as calls:
    calls.write(" ".join(args) + "\\n")
"""


def test_mlx_repositories_are_fetched_through_each_backends_hf(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text(f"#!{sys.executable}\n{FAKE_UV}")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    cache, calls = tmp_path / "hub", tmp_path / "calls.txt"
    backends = {
        "llm": {
            "project": "backends/llm",
            "command": ["mlx_lm.server"],
            "port": 8101,
            "model": {"repo": "org/llm", "revision": "main"},
        },
        "asr": {
            "project": "backends/audio",
            "command": ["mlx_audio.server"],
            "port": 8102,
            "fetch": [{"repo": "org/asr", "revision": "main"}],
        },
    }
    config = validate_config({"backends": backends})
    environ = {"HF_HUB_CACHE": str(cache), "CALLS": str(calls), "HF_HUB_OFFLINE": "1"}
    reports: list[str] = []
    fetch_mlx(config.backends, environ, reports.append, uv=str(uv))
    assert calls.read_text().splitlines() == [
        "run --locked --project backends/llm hf download org/llm --revision main",
        "run --locked --project backends/audio hf download org/asr --revision main",
    ]
    assert snapshot_dir(cache, "org/asr", "main") == cache / "models--org--asr/snapshots/c0ffee"
    assert reports == ["llm: org/llm @ main", "asr: org/asr @ main"]


def test_a_name_resolving_elsewhere_than_the_pin_is_reported(tmp_path: Path) -> None:
    repo_dir = tmp_path / "models--org--asr"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("b" * 40)
    assert main_ref_mismatch(tmp_path, "org/asr", "b" * 40) is None
    warning = main_ref_mismatch(tmp_path, "org/asr", "a" * 40)
    assert warning is not None
    assert "refs/main" in warning

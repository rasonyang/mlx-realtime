"""``mlxrt models``: download and verify the model files.

- The ONNX files the gateway runs on CPU (Silero VAD, Smart Turn) are listed, with their source
  revision, license and sha256, in the committed ``models/MANIFEST.json``. A file already
  present is re-verified, not downloaded again.
- With ``--mlx``, the MLX repositories pinned in ``config.yaml`` (``backends.*.model`` and
  ``backends.*.fetch``) are downloaded by revision through each backend's own ``hf`` CLI, so
  the backends can start offline.

``HF_TOKEN`` is sent to huggingface.co when it is set, and never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx2

from mlx_realtime.config import BackendsConfig
from mlx_realtime.hub import hub_cache, ref, resolve, snapshot_dir

MANIFEST = "MANIFEST.json"

type Report = Callable[[str], None]


class FetchError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ModelFile:
    file: str
    url: str
    sha256: str
    bytes: int


def read_manifest(directory: Path) -> list[ModelFile]:
    try:
        document = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        return [
            ModelFile(entry["file"], entry["url"], entry["sha256"], int(entry["bytes"]))
            for entry in document["models"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FetchError(f"cannot read {directory / MANIFEST}: {error}") from error


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch_files(
    directory: Path, environ: Mapping[str, str], report: Report, *, timeout_s: float = 60.0
) -> None:
    """Make every manifest file present and verified; download the missing or wrong ones."""
    with httpx2.Client(follow_redirects=True, timeout=timeout_s) as client:
        for entry in read_manifest(directory):
            target = directory / entry.file
            if target.is_file() and sha256_of(target) == entry.sha256:
                report(f"{entry.file}: verified")
                continue
            report(f"{entry.file}: downloading {entry.bytes / 1e6:.1f} MB")
            _download(client, entry, target, _headers(entry.url, environ))
            report(f"{entry.file}: downloaded and verified")


def _headers(url: str, environ: Mapping[str, str]) -> dict[str, str]:
    token = environ.get("HF_TOKEN")
    host = urlsplit(url).hostname or ""
    if token and (host == "huggingface.co" or host.endswith(".huggingface.co")):
        return {"Authorization": f"Bearer {token}"}
    return {}


def _download(
    client: httpx2.Client, entry: ModelFile, target: Path, headers: dict[str, str]
) -> None:
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with client.stream("GET", entry.url, headers=headers) as response:
            if response.status_code != 200:
                raise FetchError(f"{entry.file}: HTTP {response.status_code} from {entry.url}")
            with partial.open("wb") as file:
                for block in response.iter_bytes():
                    digest.update(block)
                    file.write(block)
    except httpx2.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{entry.file}: download failed: {error}") from error
    if digest.hexdigest() != entry.sha256:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{entry.file}: sha256 {digest.hexdigest()} != {entry.sha256}")
    partial.replace(target)


def fetch_mlx(
    backends: BackendsConfig, environ: Mapping[str, str], report: Report, *, uv: str
) -> None:
    """Download every pinned repository through its backend's ``hf`` CLI, then check it."""
    cache = hub_cache(environ)
    for name, backend in backends.processes().items():
        served = [backend.model] if backend.model is not None else []
        for pin in [*served, *backend.fetch]:
            report(f"{name}: {pin.repo} @ {pin.revision}")
            command = [uv, "run", "--locked", "--project", backend.project]
            command += ["hf", "download", pin.repo, "--revision", pin.revision]
            env = {**os.environ, **environ}
            env.pop("HF_HUB_OFFLINE", None)
            if subprocess.run(command, env=env, check=False).returncode != 0:
                raise FetchError(f"{name}: hf download {pin.repo} failed")
            if snapshot_dir(cache, pin.repo, pin.revision) is None:
                raise FetchError(f"{name}: {pin.repo} @ {pin.revision} is not in {cache}")
            if pin not in served:
                warning = main_ref_mismatch(cache, pin.repo, pin.revision)
                if warning:
                    report(f"{name}: warning: {warning}")


def main_ref_mismatch(cache: Path, repo: str, revision: str) -> str | None:
    """Models requested by name resolve ``main`` offline; say so when it is not the pin."""
    pinned, main = resolve(cache, repo, revision), ref(cache, repo, "main")
    if pinned == main:
        return None
    return (
        f"{repo}: the backend loads it by name, which resolves to refs/main ({main}) offline, "
        f"not to the pinned {revision}"
    )

"""The local Hugging Face cache, read as huggingface_hub 2.2 lays it out (the gateway does not
depend on huggingface_hub; the backends' ``hf`` CLI writes the cache).

    $HF_HUB_CACHE/models--<org>--<name>/snapshots/<commit>/...   the files of one revision
    $HF_HUB_CACHE/models--<org>--<name>/refs/<branch or tag>     the commit it pointed at
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")


def hub_cache(environ: Mapping[str, str]) -> Path:
    """``HF_HUB_CACHE``, else the legacy ``HUGGINGFACE_HUB_CACHE``, else ``$HF_HOME/hub``."""
    home = environ.get("HF_HOME") or os.path.join(
        environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "huggingface",
    )
    legacy = environ.get("HUGGINGFACE_HUB_CACHE") or os.path.join(home, "hub")
    return Path(os.path.expandvars(os.path.expanduser(environ.get("HF_HUB_CACHE") or legacy)))


def repo_dir(cache: Path, repo: str) -> Path:
    return cache / ("models--" + repo.replace("/", "--"))


def ref(cache: Path, repo: str, name: str) -> str | None:
    """The commit a branch or tag pointed at when it was last downloaded."""
    path = repo_dir(cache, repo) / "refs" / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def resolve(cache: Path, repo: str, revision: str) -> str | None:
    """The commit of ``revision``: itself if it is a commit hash, else what its ref recorded."""
    return revision if _COMMIT.fullmatch(revision) else ref(cache, repo, revision)


def snapshot_dir(cache: Path, repo: str, revision: str) -> Path | None:
    """The local snapshot of ``repo`` at ``revision``, if it has been downloaded."""
    commit = resolve(cache, repo, revision)
    if commit is None:
        return None
    path = repo_dir(cache, repo) / "snapshots" / commit
    return path if (path / "config.json").is_file() else None

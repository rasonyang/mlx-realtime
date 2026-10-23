"""S7: fetch the ONNX models into ./models and write models/MANIFEST.json (sha256 + provenance).

    uv run python fetch_models.py

Pinned sources:
- Silero VAD v6.2.3: github.com/snakers4/silero-vad at commit 5cd7945 (tag v6.2.3),
  src/silero_vad/data/silero_vad.onnx, MIT. Integrity: git blob sha1 from the GitHub tree.
- Smart Turn v3.2: HF pipecat-ai/smart-turn-v3 at f766f81d, BSD-2-Clause. Integrity: HF LFS sha256.
Existing files are re-verified, not re-downloaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, sha256_file  # noqa: E402

SILERO = {
    "file": "silero_vad.onnx",
    "source": "github.com/snakers4/silero-vad",
    "tag": "v6.2.3",
    "revision": "5cd7945676eb32225748052e2e6a0580e4686a08",
    "path": "src/silero_vad/data/silero_vad.onnx",
    "license": "MIT",
    "git_blob_sha1": "80c5592ef1f4c9ede3e357bbd02eb863358a6a9d",
}
ST_REPO = "pipecat-ai/smart-turn-v3"
ST_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
ST_FILES = {
    # file: expected LFS sha256 (HF model_info at ST_REVISION)
    "smart-turn-v3.2-cpu.onnx": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    "smart-turn-v3.2-gpu.onnx": "ab8dc64b88713f90b571c15b714bd1330e6c883cad8763dacf65c9376dc539be",
}


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_silero() -> dict:
    dst = MODELS / SILERO["file"]
    url = f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO['revision']}/{SILERO['path']}"
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dst)
    blob = git_blob_sha1(dst)
    if blob != SILERO["git_blob_sha1"]:
        sys.exit(f"{dst}: git blob sha1 {blob} != {SILERO['git_blob_sha1']}")
    return {**SILERO, "url": url, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def fetch_smart_turn() -> list[dict]:
    from huggingface_hub import hf_hub_download

    out = []
    for name, want in ST_FILES.items():
        dst = MODELS / name
        if not dst.exists():
            hf_hub_download(ST_REPO, name, revision=ST_REVISION, local_dir=MODELS)
        got = sha256_file(dst)
        if got != want:
            sys.exit(f"{dst}: sha256 {got} != {want}")
        out.append(
            {
                "file": name,
                "source": f"huggingface.co/{ST_REPO}",
                "revision": ST_REVISION,
                "license": "BSD-2-Clause",
                "precision": "int8" if "cpu" in name else "fp32",
                "bytes": dst.stat().st_size,
                "sha256": got,
            }
        )
    return out


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    entries = [fetch_silero(), *fetch_smart_turn()]
    manifest = {"fetched": time.strftime("%Y-%m-%d"), "script": "spikes/cpu/fetch_models.py", "models": entries}
    (MODELS / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for e in entries:
        print(f"{e['file']:28s} {e['bytes']:>10d}  sha256 {e['sha256']}")


if __name__ == "__main__":
    main()

"""A1 / S0: record the environment of the three spike projects in spikes/results/a1_env.json.

Run from the repository root:

    uv run --project spikes/llm python spikes/llm/record_env.py

Records: `uv --version`; the Python each project runs; key package versions from each uv.lock; the
`--help` text of mlx_lm.server and mlx_audio.server as installed; `mx.device_info()`; and
`sysctl iogpu.wired_limit_mb` (SPIKE.md §1).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import tomllib
from pathlib import Path

import mlx.core as mx

SPIKES = Path(__file__).resolve().parents[1]
PROJECTS = ("llm", "audio", "sdk")
KEY_PACKAGES = (
    "mlx", "mlx-lm", "mlx-audio", "mlx-metal", "transformers", "numpy", "openai", "websockets",
    "huggingface-hub", "hf-xet", "tokenizers", "safetensors", "misaki", "spacy", "en-core-web-sm",
    "phonemizer-fork", "espeakng-loader", "soxr", "fastapi", "uvicorn", "pydantic",
)


def run(*cmd: str) -> str:
    # Drop this project's VIRTUAL_ENV so a nested `uv run --project <other>` does not warn about it.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env).stdout.strip()


def main() -> None:
    env: dict = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "uv_version": run("uv", "--version"),
        "sysctl_iogpu.wired_limit_mb": run("sysctl", "-n", "iogpu.wired_limit_mb"),
        "hw.memsize": run("sysctl", "-n", "hw.memsize"),
        "macos": run("sw_vers", "-productVersion"),
        "mx_device_info": mx.device_info(),
        "projects": {},
    }
    for name in PROJECTS:
        proj = SPIKES / name
        lock = tomllib.loads((proj / "uv.lock").read_text())
        versions = {p["name"]: p["version"] for p in lock["package"]}
        pyproject = tomllib.loads((proj / "pyproject.toml").read_text())
        env["projects"][name] = {
            "python_pin": (proj / ".python-version").read_text().strip(),
            "python": run("uv", "run", "--project", str(proj), "python", "-c",
                          "import sys; print(sys.version.split()[0])"),
            "dependencies": pyproject["project"]["dependencies"],
            "uv_sources": pyproject.get("tool", {}).get("uv", {}).get("sources"),
            "locked_packages": len(lock["package"]),
            "key_versions": {k: versions[k] for k in KEY_PACKAGES if k in versions},
        }
    env["mlx_lm_server_help"] = run("uv", "run", "--project", str(SPIKES / "llm"), "mlx_lm.server", "--help")
    help_audio = run("uv", "run", "--project", str(SPIKES / "audio"), "python", "-m", "mlx_audio.server", "--help")
    env["mlx_audio_server_help"] = "\n".join(
        ln for ln in help_audio.splitlines() if "pkg_resources" not in ln
    )
    out = SPIKES / "results" / "a1_env.json"
    out.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in env.items() if not k.endswith("_help")}, indent=1)[:3000])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

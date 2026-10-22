"""``mlxrt dev``: the commands it builds for each backend, the port guard, and the run loop
(prefixed output, exit when a child exits, signal forwarding, the kill after grace), driven
with small Python children instead of model servers."""

from __future__ import annotations

import asyncio
import io
import os
import signal
import socket
import sys
from pathlib import Path

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.supervisor import (
    ProcessSpec,
    SupervisorError,
    backend_specs,
    busy_ports,
    gateway_spec,
    supervise,
)

LLM_REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
COMMIT = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
BACKENDS = {
    "env": {"HF_HUB_OFFLINE": "1"},
    "llm": {
        "project": "backends/llm",
        "command": ["mlx_lm.server"],
        "port": 8101,
        "model": {"repo": LLM_REPO, "revision": COMMIT},
        "args": ["--chat-template-args", '{"enable_thinking":false}', "--max-tokens", "512"],
    },
    "asr": {
        "project": "backends/audio",
        "command": ["mlx_audio.server"],
        "port": 8102,
        "fetch": [{"repo": "mlx-community/Qwen3-ASR-1.7B-8bit", "revision": "main"}],
        "args": ["--log-dir", "/tmp/asr"],
        "env": {"EXTRA": "1"},
    },
}


def snapshot(cache: Path, repo: str, commit: str) -> Path:
    path = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / commit
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def test_backend_commands(tmp_path: Path) -> None:
    served = snapshot(tmp_path, LLM_REPO, COMMIT)
    config = validate_config({"backends": BACKENDS})
    llm, asr = backend_specs(config.backends, uv="/bin/uv", cache=tmp_path)
    assert llm == ProcessSpec(
        "llm",
        (
            "/bin/uv", "run", "--locked", "--project", "backends/llm", "mlx_lm.server",
            "--host", "127.0.0.1", "--port", "8101", "--model", str(served),
            "--chat-template-args", '{"enable_thinking":false}', "--max-tokens", "512",
        ),
        {"HF_HUB_OFFLINE": "1"},
    )  # fmt: skip
    assert asr.argv == (
        "/bin/uv", "run", "--locked", "--project", "backends/audio", "mlx_audio.server",
        "--host", "127.0.0.1", "--port", "8102", "--log-dir", "/tmp/asr",
    )  # fmt: skip
    assert asr.env == {"HF_HUB_OFFLINE": "1", "EXTRA": "1"}
    gateway = gateway_spec(Path("config.yaml"), python="/py")
    assert gateway.argv == ("/py", "-m", "mlx_realtime", "--config", "config.yaml", "serve")


def test_a_served_model_must_be_downloaded(tmp_path: Path) -> None:
    config = validate_config({"backends": BACKENDS})
    with pytest.raises(SupervisorError, match=r"mlxrt models --mlx"):
        backend_specs(config.backends, uv="uv", cache=tmp_path)


def test_a_branch_revision_resolves_through_its_ref(tmp_path: Path) -> None:
    served = snapshot(tmp_path, LLM_REPO, COMMIT)
    refs = tmp_path / ("models--" + LLM_REPO.replace("/", "--")) / "refs"
    refs.mkdir()
    (refs / "main").write_text(COMMIT)
    llm = {**BACKENDS["llm"], "model": {"repo": LLM_REPO, "revision": "main"}}
    config = validate_config({"backends": {"llm": llm}})
    (spec,) = backend_specs(config.backends, uv="uv", cache=tmp_path)
    assert str(served) in spec.argv


def test_ports_in_use_are_reported() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        taken = listener.getsockname()[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        assert busy_ports([("tts", "127.0.0.1", taken), ("llm", "127.0.0.1", free)]) == [
            f"tts: 127.0.0.1:{taken} is already in use"
        ]


def child(name: str, code: str) -> ProcessSpec:
    return ProcessSpec(name, (sys.executable, "-c", code), {"CHILD": name})


SLEEPER = (
    "import os, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, lambda *_: (print('term', flush=True), sys.exit(0)))\n"
    "print('up', os.environ['CHILD'], flush=True)\n"
    "time.sleep(60)\n"
)


async def wait_for(out: io.StringIO, text: str) -> None:
    async with asyncio.timeout(10):
        while text not in out.getvalue():  # noqa: ASYNC110 (output written by the supervisor)
            await asyncio.sleep(0.01)


async def test_output_is_prefixed_and_the_first_exit_stops_everything() -> None:
    out = io.StringIO()
    specs = [
        child("llm", SLEEPER),
        child("gateway", "import sys; print('bye', flush=True); sys.exit(3)"),
    ]
    code = await supervise(specs, out, grace_s=5)
    lines = out.getvalue().splitlines()
    assert code == 3
    assert "gateway | bye" in lines
    assert "mlxrt   | gateway exited with code 3; stopping the others" in lines
    assert "llm     | term" in lines  # stopped with SIGTERM, not killed


async def test_a_clean_exit_still_counts_as_a_failure() -> None:
    assert await supervise([child("gateway", "pass")], io.StringIO(), grace_s=1) == 1


async def test_sigterm_is_forwarded_to_every_child() -> None:
    out = io.StringIO()
    running = asyncio.create_task(supervise([child("llm", SLEEPER), child("asr", SLEEPER)], out))
    await wait_for(out, "up llm")
    await wait_for(out, "up asr")
    os.kill(os.getpid(), signal.SIGTERM)  # the supervisor's handler is installed by now
    assert await running == 0
    lines = out.getvalue().splitlines()
    assert "mlxrt | SIGTERM: stopping every process" in lines
    assert {"llm   | term", "asr   | term"} <= set(lines)


async def test_a_child_that_ignores_the_stop_is_killed_after_grace() -> None:
    stubborn = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('up', flush=True)\n"
        "time.sleep(60)\n"
    )
    out = io.StringIO()
    specs = [child("tts", stubborn), child("gateway", "import time; time.sleep(0.5)")]
    async with asyncio.timeout(10):
        code = await supervise(specs, out, grace_s=0.3)
    assert code == 1

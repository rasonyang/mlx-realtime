"""Start the ASR and TTS backends of config.yaml (not the LLM) the way ``mlxrt dev`` does.

    uv run --locked python spikes/live_echo/hold_backends.py        # until Ctrl-C

``bursts.py`` starts them itself when nothing listens on their ports (``backends_running``).
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "src"))

from mlx_realtime.config import load_config  # noqa: E402
from mlx_realtime.hub import hub_cache  # noqa: E402
from mlx_realtime.supervisor import backend_specs  # noqa: E402


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@contextlib.contextmanager
def backends_running(log_dir: Path, timeout_s: float = 300.0) -> Iterator[None]:
    """ASR and TTS up (started here unless already listening); stopped on exit if started here."""
    config = load_config(REPO / "config.yaml", os.environ)
    specs = backend_specs(
        config.backends.model_copy(update={"llm": None}), uv=os.environ.get("UV", "uv"),
        cache=hub_cache(os.environ),
    )
    ports = {name: b.port for name, b in config.backends.processes().items() if name != "llm"}
    log_dir.mkdir(parents=True, exist_ok=True)
    started: list[subprocess.Popen[bytes]] = []
    try:
        for spec in specs:
            if _listening(ports[spec.name]):
                print(f"{spec.name}: already listening on {ports[spec.name]}")
                continue
            log = (log_dir / f"{spec.name}.log").open("wb")
            started.append(
                subprocess.Popen(
                    spec.argv, cwd=REPO, env={**os.environ, **spec.env}, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                )
            )
        deadline = time.monotonic() + timeout_s
        while not all(_listening(p) for p in ports.values()):
            if time.monotonic() > deadline or any(p.poll() is not None for p in started):
                raise SystemExit(f"backends did not come up; see {log_dir}")
            time.sleep(1.0)
        yield
    finally:
        import signal

        for process in started:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        for process in started:
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    with backends_running(Path(__file__).parent / "captures" / "hold" / "logs"):
        print("ASR and TTS are up; Ctrl-C to stop")
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                time.sleep(3600)

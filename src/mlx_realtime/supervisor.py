"""``mlxrt dev``: the development supervisor (ARCHITECTURE §2, Supervision).

It starts every configured backend with ``uv run --locked --project backends/<role> …``, then
the gateway, and:

- prefixes every output line with the process name;
- forwards SIGINT and SIGTERM to every child, and stops them all;
- exits when any child exits (there is no restart loop in development).

Each child runs in its own session, so a Ctrl-C reaches it once, through the supervisor; ``uv
run`` forwards the signal to the process it runs. A child still running ``grace_s`` after the
stop request is killed with its whole process group.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from mlx_realtime.config import BackendsConfig
from mlx_realtime.hub import snapshot_dir

SUPERVISOR = "mlxrt"
_LINE_LIMIT = 1 << 20


class SupervisorError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str]  # added to the supervisor's own environment


def backend_specs(backends: BackendsConfig, *, uv: str, cache: Path) -> list[ProcessSpec]:
    """``uv run --locked --project <project> <command> --host … --port … [--model …] <args>``.

    A served model is passed as its local snapshot directory, so the backend runs exactly the
    pinned revision.
    """
    specs: list[ProcessSpec] = []
    for name, backend in backends.processes().items():
        argv = [uv, "run", "--locked", "--project", backend.project, *backend.command]
        argv += ["--host", backend.host, "--port", str(backend.port)]
        if backend.model is not None:
            path = snapshot_dir(cache, backend.model.repo, backend.model.revision)
            if path is None:
                pin = f"{backend.model.repo} @ {backend.model.revision}"
                raise SupervisorError(f"{name}: {pin} is not downloaded; run `mlxrt models --mlx`")
            argv += ["--model", str(path)]
        argv += backend.args
        specs.append(ProcessSpec(name, tuple(argv), {**backends.env, **backend.env}))
    return specs


def busy_ports(endpoints: Iterable[tuple[str, str, int]]) -> list[str]:
    """The ``(name, host, port)`` endpoints something already listens on. Checked before any
    process starts, so a port in use never leaves half a stack running."""
    busy: list[str] = []
    for name, host, port in endpoints:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                busy.append(f"{name}: {host}:{port} is already in use")
    return busy


def gateway_spec(config_path: Path, *, python: str) -> ProcessSpec:
    argv = (python, "-m", "mlx_realtime", "--config", str(config_path), "serve")
    return ProcessSpec("gateway", argv, {})


class _Child:
    def __init__(self, spec: ProcessSpec, process: asyncio.subprocess.Process) -> None:
        self.spec = spec
        self.process = process

    @classmethod
    async def start(cls, spec: ProcessSpec) -> _Child:
        process = await asyncio.create_subprocess_exec(
            *spec.argv,
            env={**os.environ, **spec.env},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            limit=_LINE_LIMIT,
        )
        return cls(spec, process)

    async def pump(self, out: TextIO, width: int) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        while line := await stream.readline():
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            out.write(f"{self.spec.name:<{width}} | {text}\n")
            out.flush()

    def signal(self, signum: int) -> None:
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.send_signal(signum)

    def kill_group(self) -> None:
        if self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGKILL)


async def supervise(specs: Sequence[ProcessSpec], out: TextIO, *, grace_s: float = 10.0) -> int:
    """Run ``specs`` until one exits or a stop is requested; return the exit code."""
    width = max([len(SUPERVISOR), *(len(spec.name) for spec in specs)])

    def say(message: str) -> None:
        out.write(f"{SUPERVISOR:<{width}} | {message}\n")
        out.flush()

    loop = asyncio.get_running_loop()
    requested: list[int] = []
    stop = asyncio.Event()

    def on_signal(signum: int) -> None:
        requested.append(signum)
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, on_signal, signum)
    children: list[_Child] = []
    try:
        async with asyncio.TaskGroup() as tasks:
            for spec in specs:
                say(f"starting {spec.name}: {' '.join(spec.argv)}")
                child = await _Child.start(spec)
                children.append(child)
                tasks.create_task(child.pump(out, width))
            exits = {
                tasks.create_task(child.process.wait(), name=child.spec.name): child
                for child in children
            }
            stopping = tasks.create_task(stop.wait())
            await asyncio.wait({*exits, stopping}, return_when=asyncio.FIRST_COMPLETED)
            finished = [child for task, child in exits.items() if task.done()]
            if finished:
                first = finished[0]
                code = first.process.returncode or 0
                say(f"{first.spec.name} exited with code {code}; stopping the others")
                result = 1 if code == 0 else (128 - code if code < 0 else code)
                forward = signal.SIGTERM
            else:
                forward = requested[0]
                say(f"{signal.Signals(forward).name}: stopping every process")
                result = 0
            stopping.cancel()
            await _stop_all(exits, forward, grace_s)
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)
        for child in children:
            child.kill_group()
    return result


async def _stop_all(exits: Mapping[asyncio.Task[int], _Child], signum: int, grace_s: float) -> None:
    """Signal every child, then kill the process groups of those still running after grace."""
    for child in exits.values():
        child.signal(signum)
    _, pending = await asyncio.wait(exits, timeout=grace_s)
    for task in pending:
        exits[task].kill_group()
    if pending:
        await asyncio.wait(pending)

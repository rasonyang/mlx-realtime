"""The ``mlxrt`` command.

- ``serve`` runs the gateway.
- ``dev`` is what ``make dev`` runs: a supervisor that starts the configured model backends and
  the gateway (``supervisor.py``).
- ``models`` downloads and verifies the model files (``fetch.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mlx_realtime.config import Config, ConfigError, load_config
from mlx_realtime.fetch import FetchError, fetch_files, fetch_mlx, main_ref_mismatch
from mlx_realtime.hub import hub_cache
from mlx_realtime.logs import configure_logging
from mlx_realtime.server.app import GatewayServer
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.supervisor import (
    SupervisorError,
    backend_specs,
    busy_ports,
    gateway_spec,
    supervise,
)
from mlx_realtime.vad.onnx import ModelFileMissing

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlxrt", description="Local OpenAI Realtime gateway.")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="configuration file (config.yaml)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the gateway")
    commands.add_parser("dev", help="run the model backends and the gateway, for development")
    models = commands.add_parser("models", help="download and verify the model files")
    models.add_argument(
        "--dir", type=Path, default=Path("models"), help="ONNX model directory (models)"
    )
    models.add_argument(
        "--mlx", action="store_true", help="also download the MLX repositories pinned in the config"
    )
    args = parser.parse_args(argv)
    match args.command:
        case "serve":
            _serve(args.config)
        case "dev":
            raise SystemExit(_dev(args.config))
        case _:
            raise SystemExit(_models(args.config, args.dir, mlx=args.mlx))


def _load(path: Path) -> Config:
    try:
        return load_config(path, os.environ)
    except ConfigError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None


def _serve(path: Path) -> None:
    config = _load(path)
    try:
        gateway = Gateway(config)
    except (ConfigError, ModelFileMissing) as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
    configure_logging(config.observability.log_level)
    logger.info(
        "starting gateway", extra={"host": config.gateway.host, "port": config.gateway.port}
    )
    GatewayServer(gateway).run()


def _dev(path: Path) -> int:
    config = _load(path)
    cache = hub_cache(os.environ)
    try:
        specs = backend_specs(config.backends, uv=os.environ.get("UV", "uv"), cache=cache)
    except SupervisorError as error:
        print(error, file=sys.stderr)
        return 2
    endpoints = [(name, b.host, b.port) for name, b in config.backends.processes().items()]
    endpoints.append(("gateway", config.gateway.host, config.gateway.port))
    if busy := busy_ports(endpoints):
        print("\n".join(busy), file=sys.stderr)
        return 2
    for name, backend in config.backends.processes().items():
        for pin in backend.fetch:
            if warning := main_ref_mismatch(cache, pin.repo, pin.revision):
                print(f"{name}: warning: {warning}", file=sys.stderr)
    specs.append(gateway_spec(path, python=sys.executable))
    return asyncio.run(supervise(specs, sys.stdout))


def _models(path: Path, directory: Path, *, mlx: bool) -> int:
    def report(message: str) -> None:
        print(message, flush=True)

    try:
        fetch_files(directory, os.environ, report)
        if mlx:
            uv = os.environ.get("UV", "uv")
            fetch_mlx(_load(path).backends, os.environ, report, uv=uv)
    except FetchError as error:
        print(error, file=sys.stderr)
        return 1
    return 0

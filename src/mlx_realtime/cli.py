"""``mlxrt serve`` runs the gateway. ``mlxrt dev`` is what ``make dev`` runs.

In M1 every provider is a mock, so ``dev`` starts the gateway alone; from M2 on it becomes the
supervisor that also starts the model backends (ARCHITECTURE §2).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mlx_realtime.config import ConfigError, load_config
from mlx_realtime.logs import configure_logging
from mlx_realtime.providers.registry import build_providers
from mlx_realtime.server.app import GatewayServer
from mlx_realtime.server.gateway import Gateway

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mlxrt", description="Local OpenAI Realtime gateway.")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="configuration file (config.yaml)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="run the gateway")
    commands.add_parser("dev", help="development run (M1: the gateway with mock providers)")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, os.environ)
        gateway = Gateway(config, build_providers(config.providers))
    except ConfigError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from None
    configure_logging(config.observability.log_level)
    logger.info(
        "starting gateway",
        extra={"command": args.command, "host": config.gateway.host, "port": config.gateway.port},
    )
    GatewayServer(gateway).run()

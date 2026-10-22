"""Run AICC's own Realtime client harness against a gateway (ARCHITECTURE §11, AICC acceptance).

AICC (ai-native-callcenter) attaches through its ``gateway`` profile (D11). Its test package
``internal/provider`` holds the live harness: ``probe`` (handshake, an opening spoken turn, audio
back) and ``measureResponseCreated``. This script adds ``zz_mlx_live_test.go`` (next to it) to
that package with ``go test -overlay``, so the AICC checkout is never modified, points it at the
gateway, and checks AICC's deadlines on the result:

- the handshake within 3 s (AICC's own dial and session.updated waits);
- ``response.created`` within 3 s of each ``response.create`` (every sample measured);
- the first audio within 3 s of ``response.created``, and no 2 s gap between audio deltas: AICC's
  watchdog abandons such a turn and logs ``response abandoned``, which fails the run;
- the opening and a mid-call line spoken word for word, with a flow voice (D16, D19).

Usage (the gateway must be running, or let the script start one with ``--serve``)::

    uv run --locked python tests/aicc/probe.py --aicc-repo ~/src/ai-native-callcenter \\
        --url ws://127.0.0.1:8000/v1/realtime
    uv run --locked python tests/aicc/probe.py --aicc-repo ~/src/ai-native-callcenter \\
        --serve config.mock.yaml          # a gateway of its own, with mock providers

The API key goes to AICC as ``REALTIME_API_KEY`` (the gateway profile's variable); without one
the wrapper sends ``local``. Needs Go and AICC's module dependencies (``go mod download``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WRAPPER = HERE / "zz_mlx_live_test.go"
PACKAGE = Path("internal") / "provider"
DEADLINE_S = 3.0

_DURATION = r"((?:\d+(?:\.\d+)?(?:h|ms|us|µs|ns|m|s))+)"
_SAMPLE = re.compile(rf"request -> response\.created {_DURATION}")
_SUMMARY = re.compile(rf"n=(\d+)/(\d+) median={_DURATION} p95={_DURATION} max={_DURATION}")
_FIRST_AUDIO = re.compile(rf"first audio after {_DURATION}")
_RESULT = re.compile(r"^--- (PASS|FAIL|SKIP): (\S+)")
_GIVEN_UP = ("response abandoned", "unanswered", "never answered")
_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
_PART = re.compile(r"(\d+(?:\.\d+)?)(h|ms|us|µs|ns|m|s)")


def go_duration(text: str) -> float:
    """Seconds in a Go ``time.Duration`` string such as ``1m2.5s``, ``350ms`` or ``0s``."""
    parts = _PART.findall(text)
    if not parts or "".join(number + unit for number, unit in parts) != text:
        raise ValueError(f"not a Go duration: {text!r}")
    return sum(float(number) * _UNITS[unit] for number, unit in parts)


@dataclass
class Report:
    results: dict[str, str] = field(default_factory=dict[str, str])
    created_s: list[float] = field(default_factory=list[float])
    created_max_s: float | None = None
    unanswered: int = 0
    first_audio_s: list[float] = field(default_factory=list[float])
    given_up: list[str] = field(default_factory=list[str])

    def read(self, line: str) -> None:
        if match := _RESULT.match(line.strip()):
            self.results[match.group(2)] = match.group(1)
        if match := _SAMPLE.search(line):
            self.created_s.append(go_duration(match.group(1)))
        if match := _SUMMARY.search(line):
            answered, asked = int(match.group(1)), int(match.group(2))
            self.unanswered += asked - answered
            self.created_max_s = go_duration(match.group(5))
        if match := _FIRST_AUDIO.search(line):
            self.first_audio_s.append(go_duration(match.group(1)))
        if any(marker in line for marker in _GIVEN_UP):
            self.given_up.append(line.strip())

    def failures(self, exit_code: int) -> list[str]:
        problems: list[str] = []
        if exit_code != 0:
            problems.append(f"go test exited with {exit_code}")
        failed = sorted(name for name, result in self.results.items() if result == "FAIL")
        if failed:
            problems.append(f"failed: {', '.join(failed)}")
        if not self.results or all(result == "SKIP" for result in self.results.values()):
            problems.append("no test ran")
        late = [sample for sample in self.created_s if sample >= DEADLINE_S]
        if late:
            problems.append(f"response.created later than {DEADLINE_S:g} s: {late}")
        if self.unanswered:
            problems.append(f"{self.unanswered} response.create left unanswered")
        if self.given_up:
            problems.append("AICC gave up on a turn: " + " | ".join(self.given_up))
        return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--aicc-repo", type=Path, required=True, help="an AICC checkout")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="the gateway, e.g. ws://127.0.0.1:8000/v1/realtime")
    target.add_argument(
        "--serve", type=Path, metavar="CONFIG", help="start a gateway with this config first"
    )
    parser.add_argument(
        "--api-key", default=None, help="REALTIME_API_KEY for AICC (default: local)"
    )
    parser.add_argument("--model", default=None, help="the ?model= AICC asks for")
    parser.add_argument("--run", default="^TestMLX", help="go test -run pattern")
    args = parser.parse_args(argv)

    aicc: Path = args.aicc_repo.expanduser().resolve()
    if not (aicc / PACKAGE / "live_test.go").is_file():
        print(f"{aicc} has no {PACKAGE}/live_test.go: not an AICC checkout", file=sys.stderr)
        return 2
    go = shutil.which("go")
    if go is None:
        print("go is not on PATH", file=sys.stderr)
        return 2
    gateway: subprocess.Popen[bytes] | None = None
    try:
        url: str = args.url
        if args.serve is not None:
            gateway, url = serve(args.serve)
        print(f"AICC harness against {url}", flush=True)
        exit_code, report = run_harness(go, aicc, url, args)
    finally:
        if gateway is not None:
            stop(gateway)
    problems = report.failures(exit_code)
    summarize(report, problems)
    return 1 if problems else 0


def run_harness(go: str, aicc: Path, url: str, args: argparse.Namespace) -> tuple[int, Report]:
    environ = {**os.environ, "MLX_REALTIME_URL": url}
    if args.api_key is not None:
        environ["REALTIME_API_KEY"] = args.api_key
    if args.model is not None:
        environ["MLX_REALTIME_MODEL"] = args.model
    report = Report()
    with tempfile.TemporaryDirectory(prefix="mlxrt-aicc-") as scratch:
        overlay = Path(scratch) / "overlay.json"
        target = str(aicc / PACKAGE / WRAPPER.name)
        overlay.write_text(json.dumps({"Replace": {target: str(WRAPPER)}}), encoding="utf-8")
        command = [go, "test", "-count=1", f"-overlay={overlay}", "-run", args.run, "-v"]
        command.append("./" + PACKAGE.as_posix() + "/")
        process = subprocess.Popen(
            command,
            cwd=aicc,
            env=environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            report.read(line)
        return process.wait(), report


def serve(config: Path) -> tuple[subprocess.Popen[bytes], str]:
    """``mlxrt serve`` with ``config`` on a free port of 127.0.0.1, once it is ready."""
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    environ = {**os.environ, "MLXRT__GATEWAY__HOST": "127.0.0.1", "MLXRT__GATEWAY__PORT": str(port)}
    command = ["uv", "run", "--locked", "mlxrt", "--config", str(config), "serve"]
    log = Path(tempfile.gettempdir()) / f"mlxrt-aicc-gateway-{port}.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            command, cwd=REPO, env=environ, stdout=output, stderr=subprocess.STDOUT
        )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the gateway exited with {process.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1) as answer:
                if answer.status == 200:
                    print(f"gateway ready on port {port} (log: {log})", flush=True)
                    return process, f"ws://127.0.0.1:{port}/v1/realtime"
        except OSError:
            pass
        time.sleep(0.2)
    stop(process)
    raise SystemExit(f"the gateway was not ready within 180 s; see {log}")


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def summarize(report: Report, problems: list[str]) -> None:
    print("\n== AICC acceptance summary ==")
    for name, result in sorted(report.results.items()):
        print(f"  {result:4}  {name}")
    if report.created_s:
        worst = max(report.created_s)
        print(f"  response.created: {len(report.created_s)} samples, worst {worst * 1000:.0f} ms")
    if report.first_audio_s:
        print(f"  first audio after session start: {report.first_audio_s[0] * 1000:.0f} ms")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print("  OK" if not problems else "  FAILED")


if __name__ == "__main__":
    raise SystemExit(main())

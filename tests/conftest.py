from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.integration.invariants import assert_causal
from tests.support import Event, normalize

GOLDEN_DIR = Path(__file__).parent / "golden"

type GoldenCheck = Callable[[str, Sequence[tuple[str, Event]]], None]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="rewrite tests/golden/*.jsonl from this run instead of comparing",
    )


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenCheck:
    """Check a recorded trace's causal invariants, then compare it with
    ``tests/golden/<name>.jsonl`` (ids normalized)."""
    update = bool(request.config.getoption("--update-golden"))

    def check(name: str, trace: Sequence[tuple[str, Event]]) -> None:
        assert_causal([event for direction, event in trace if direction == "server"])
        path = GOLDEN_DIR / f"{name}.jsonl"
        lines = [
            json.dumps(entry, ensure_ascii=False, sort_keys=True) for entry in normalize(trace)
        ]
        if update:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        assert path.exists(), f"{path} is missing; run pytest --update-golden"
        assert lines == path.read_text(encoding="utf-8").splitlines()

    return check

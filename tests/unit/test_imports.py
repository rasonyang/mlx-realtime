"""The layer rules of ARCHITECTURE §4.1, checked on the import graph of ``src/``."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / "src" / "mlx_realtime"


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path: Path) -> set[str]:
    """Every module ``path`` imports, relative imports resolved."""
    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


MODULES = {module_name(path): imports_of(path) for path in sorted(PACKAGE.rglob("*.py"))}


def internal(imports: set[str]) -> set[str]:
    return {name for name in imports if name.startswith("mlx_realtime.")}


def layer(module: str) -> str:
    return module.split(".")[1] if module.count(".") else ""


def test_the_graph_covers_every_layer() -> None:
    assert {layer(module) for module in MODULES} >= {
        "server",
        "protocol",
        "session",
        "providers",
        "audio",
        "text",
        "vad",
        "config",
        "logs",
        "cli",
    }


@pytest.mark.parametrize("module", sorted(MODULES))
def test_nothing_imports_mlx(module: str) -> None:
    banned = {
        name for name in MODULES[module] if name.split(".")[0] in {"mlx", "mlx_lm", "mlx_audio"}
    }
    assert not banned
    assert not {
        name
        for name in MODULES[module]
        if name.startswith("mlx") and not name.startswith("mlx_realtime")
    }


def test_the_gateway_environment_has_no_mlx() -> None:
    for name in ("mlx", "mlx_lm", "mlx_audio"):
        assert importlib.util.find_spec(name) is None


FORBIDDEN = {
    "session": {"protocol", "server"},
    "providers": {"session", "protocol", "server"},
}
LEAVES = {"audio", "text", "vad"}


@pytest.mark.parametrize("module", sorted(MODULES))
def test_layer_rules(module: str) -> None:
    own = layer(module)
    targets = {layer(name) for name in internal(MODULES[module])} - {own}
    assert not targets & FORBIDDEN.get(own, set()), f"{module} imports {targets & FORBIDDEN[own]}"
    if own in LEAVES:
        assert not targets, f"leaf module {module} imports {targets}"

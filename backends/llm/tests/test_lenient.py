"""The lenient qwen3_coder parser and its installation into mlx_lm.server (mlx-lm 0.32.2).

The captured outputs are M0's (``tests/data/s1_toolparse_qwen3.6.json``, copied from B1's
``spikes/results/s1_toolparse_qwen3.6.json``): 7 calls the stock parser dropped and 23 it parsed.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx_lm
import pytest
from mlx_lm import server, tokenizer_utils
from mlx_lm.tool_parsers import qwen3_coder

from mlxrt_llm import serve
from mlxrt_llm.lenient import declared_names, is_lenient, lenient, rewrite

DATA = json.loads((Path(__file__).parent / "data" / "s1_toolparse_qwen3.6.json").read_text())
TOOLS: list[dict[str, Any]] = DATA["tools"]
NAMES = declared_names(TOOLS)
STOCK = qwen3_coder.parse_tool_call
REPO = Path(__file__).resolve().parents[3]
QWEN36_SNAPSHOT = Path(
    os.environ.get(
        "MLXRT_QWEN36_SNAPSHOT",
        Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
        "/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46",
    )
)


def body(generated: str) -> str:
    """What the server hands the parser: the text between ``<tool_call>`` and ``</tool_call>``."""
    return generated.split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0]


def rendered(call: dict[str, Any]) -> str:
    """A call's body as the Qwen3.6 chat template renders it (and the model writes it)."""
    out = f"\n<function={call['name']}>\n"
    for name, value in call["arguments"].items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        out += f"<parameter={name}>\n{text}\n</parameter>\n"
    return out + "</function>\n"


@pytest.fixture
def restore_parser() -> Iterator[None]:
    yield
    qwen3_coder.parse_tool_call = STOCK


@pytest.mark.parametrize("generated", DATA["dropped"])
def test_the_stock_parser_drops_every_captured_call(generated: str) -> None:
    with pytest.raises(ValueError, match="No function provided"):
        STOCK(body(generated), TOOLS)


@pytest.mark.parametrize("generated", DATA["dropped"])
def test_the_lenient_parser_recovers_every_captured_call(generated: str) -> None:
    call = lenient(STOCK)(body(generated), TOOLS)
    assert call["name"] == "transfer_to_agent"
    arguments = call["arguments"]
    assert arguments["queue"] == "mobile_support"
    assert isinstance(arguments["reason"], str)
    assert arguments["reason"]
    assert isinstance(arguments["summary"], str)
    assert arguments["summary"]
    assert arguments["slots"] == {}  # typed from the schema (object), as for a well-formed call


@pytest.mark.parametrize("call", DATA["parsed"])
def test_well_formed_calls_parse_exactly_as_before(call: dict[str, Any]) -> None:
    text = rendered(call)
    assert STOCK(text, TOOLS) == call
    assert lenient(STOCK)(text, TOOLS) == call


def test_b1s_results_when_present() -> None:
    """Every call B1's spike results record as dropped is recovered (the copy above is the
    same data; this reads the spike files themselves when the checkout has them)."""
    files = sorted((REPO / "spikes" / "results").glob("s1_toolparse_*.json"))
    if not files:
        pytest.skip("no spikes/results/s1_toolparse_*.json in this checkout")
    for path in files:
        for generated in json.loads(path.read_text())["dropped_raw_from_log"]:
            assert lenient(STOCK)(body(generated), TOOLS)["name"] in NAMES


def test_an_undeclared_name_is_not_invented_into_a_call() -> None:
    text = "\n<lookup_balance>\n<parameter=accountId>\n42\n</parameter>\n</function>\n"
    assert rewrite(text, NAMES) is None
    with pytest.raises(ValueError, match="No function provided"):
        lenient(STOCK)(text, TOOLS)


def test_a_call_closed_with_its_own_name() -> None:
    text = "\n<repair_status>\n<parameter=rma>\nRMA1001\n</parameter>\n</repair_status>\n"
    assert lenient(STOCK)(text, TOOLS) == {"name": "repair_status", "arguments": {"rma": "RMA1001"}}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("\n<repair_status>\n<parameter=rma>\nRMA10", id="cut-inside-the-call"),
        pytest.param("\n<repair_status>\n", id="no-parameters-no-close"),
        pytest.param("I will call repair_status now.", id="prose"),
        pytest.param("", id="empty"),
    ],
)
def test_what_cannot_be_repaired_fails_as_before(text: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 (the stock parser's own error)
        lenient(STOCK)(text, TOOLS)


def test_rewrite_leaves_a_function_tag_alone() -> None:
    text = "\n<function=hangup>\n</function>\n"
    assert rewrite(text, NAMES) is None
    assert rewrite("\n<hangup>\n</function>\n", NAMES) == text


@pytest.mark.parametrize(
    "tools",
    [None, [], "tools", [{"type": "function"}], [{"function": {"name": ""}}], [{"function": 3}]],
)
def test_declared_names_of_malformed_tool_lists(tools: object) -> None:
    assert declared_names(tools) == frozenset()


def test_the_server_formats_a_recovered_call() -> None:
    """mlx_lm.server's own formatter: the stock parser's call is dropped, the lenient one's
    streams as a complete call."""
    text = body(DATA["dropped"][0])
    assert server.ToolCallFormatter(STOCK, TOOLS, streaming=True)([text]) == []
    (call,) = server.ToolCallFormatter(lenient(STOCK), TOOLS, streaming=True)([text])
    assert call["type"] == "function"
    assert call["index"] == 0
    assert call["function"]["name"] == "transfer_to_agent"
    assert json.loads(call["function"]["arguments"])["queue"] == "mobile_support"


def test_install_patches_the_parser_the_server_loads(restore_parser: None) -> None:
    serve.install()
    serve.install()  # idempotent: wrapped once
    patched = importlib.import_module("mlx_lm.tool_parsers.qwen3_coder").parse_tool_call
    assert is_lenient(patched)
    assert patched.__wrapped__ is STOCK
    assert patched(body(DATA["dropped"][0]), TOOLS)["name"] == "transfer_to_agent"


@pytest.mark.skipif(not QWEN36_SNAPSHOT.is_dir(), reason="the Qwen3.6 snapshot is not downloaded")
def test_a_loaded_qwen36_tokenizer_carries_the_lenient_parser(restore_parser: None) -> None:
    """The hook as the server meets it: loading the model's tokenizer picks qwen3_coder from
    the chat template and reads its (now lenient) parse_tool_call."""
    serve.install()
    tokenizer = tokenizer_utils.load(QWEN36_SNAPSHOT)
    assert is_lenient(tokenizer.tool_parser)
    assert tokenizer.tool_call_start == "<tool_call>"


def test_install_refuses_another_mlx_lm_version(
    monkeypatch: pytest.MonkeyPatch, restore_parser: None
) -> None:
    monkeypatch.setattr(mlx_lm, "__version__", "0.33.0")
    with pytest.raises(serve.PatchError, match=r"mlx-lm is 0\.33\.0"):
        serve.install()
    assert qwen3_coder.parse_tool_call is STOCK


def test_install_refuses_a_changed_hook_site(
    monkeypatch: pytest.MonkeyPatch, restore_parser: None
) -> None:
    def load(*args: object, **kwargs: object) -> None:
        """A tokenizer loader that picks its parser some other way."""

    monkeypatch.setattr(tokenizer_utils, "load", load)
    with pytest.raises(serve.PatchError, match="no longer reads"):
        serve.install()
    assert qwen3_coder.parse_tool_call is STOCK


def test_install_refuses_a_parser_that_changed(
    monkeypatch: pytest.MonkeyPatch, restore_parser: None
) -> None:
    def accepts_everything(model_output: str, tools: Any = None) -> dict[str, Any]:
        return {"name": "lookup", "arguments": {}}

    monkeypatch.setattr(qwen3_coder, "parse_tool_call", accepts_everything)
    with pytest.raises(serve.PatchError, match="accepts a call opened with <NAME>"):
        serve.install()


def test_main_exits_before_the_server_on_a_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], restore_parser: None
) -> None:
    monkeypatch.setattr(mlx_lm, "__version__", "0.31.0")
    monkeypatch.setattr(server, "main", lambda: pytest.fail("the server must not start"))
    with pytest.raises(SystemExit) as exited:
        serve.main()
    assert exited.value.code == 2
    assert "refusing to start" in capsys.readouterr().err

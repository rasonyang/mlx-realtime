"""``python -m mlxrt_llm.serve``: ``mlx_lm.server`` with the lenient tool-call parser.

The command line is ``mlx_lm.server``'s own, passed through unchanged, plus two flags of ours
(``checkpoints.py``): ``--tools-checkpoint`` also saves a prompt-cache checkpoint where the tool
definitions end, ``--shared-prefix-checkpoint`` one where a new system message stops matching a
recent one. Before the server starts,
``install()`` replaces ``mlx_lm.tool_parsers.qwen3_coder.parse_tool_call`` with
``lenient.lenient(original)``. That is the one hook it needs: when the server loads a model,
``tokenizer_utils.load`` imports the tool-parser module the chat template calls for and reads
its ``parse_tool_call`` then, so every request's ``ToolCallFormatter`` gets the lenient parser.

The patch is written against mlx-lm 0.32.2 and refuses to run on anything else: a different
version, a hook site that no longer reads the parser the same way, or a parser that does not
behave as the patch expects stops the backend at start with an error that says what changed.
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import Any, cast

from mlxrt_llm import checkpoints
from mlxrt_llm.lenient import Parser, is_lenient, lenient

SUPPORTED_MLX_LM = "0.32.2"

# What the patch relies on, as it reads in mlx-lm 0.32.2's source.
_LOAD_IMPORTS_PARSER = 'importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")'
_LOAD_READS_PARSER = "tool_parser = tool_module.parse_tool_call"
_FORMATTER_CALLS_PARSER = "self._tool_parser(tool_text, self._tools)"

# A call as Qwen3.6 writes it when it drops ``function=`` (M0 S1), and what it must parse to.
_SELF_TEST_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}},
        },
    }
]
_SELF_TEST_TEXT = "\n<lookup>\n<parameter=count>\n3\n</parameter>\n</function>\n"
_SELF_TEST_RESULT = {"name": "lookup", "arguments": {"count": 3}}


class PatchError(RuntimeError):
    """mlx-lm is not what the patch was written against; the backend must not start."""


def install() -> None:
    """Replace qwen3_coder's ``parse_tool_call`` with the lenient parser (idempotent)."""
    import mlx_lm
    from mlx_lm import server, tokenizer_utils
    from mlx_lm.tool_parsers import qwen3_coder

    problems: list[str] = []
    version = str(cast(object, mlx_lm.__version__))
    if version != SUPPORTED_MLX_LM:
        problems.append(f"mlx-lm is {version}; the patch was written against {SUPPORTED_MLX_LM}")
    load_source = _source(tokenizer_utils, "load")
    if _LOAD_IMPORTS_PARSER not in load_source or _LOAD_READS_PARSER not in load_source:
        problems.append("tokenizer_utils.load no longer reads the parser module's parse_tool_call")
    formatter_source = _source(server.ToolCallFormatter, "__call__")
    if _FORMATTER_CALLS_PARSER not in formatter_source:
        problems.append("server.ToolCallFormatter no longer calls the parser with the tools")
    markers = (qwen3_coder.tool_call_start, qwen3_coder.tool_call_end)
    if markers != ("<tool_call>", "</tool_call>"):
        problems.append(f"qwen3_coder's call markers changed: {markers}")
    if problems:
        raise PatchError(_refusal(problems))

    current = cast(Parser, qwen3_coder.parse_tool_call)
    if is_lenient(current):
        return
    patched = lenient(current)
    try:
        current(_SELF_TEST_TEXT, _SELF_TEST_TOOLS)
    except ValueError:
        pass
    else:
        problems.append("the stock parser accepts a call opened with <NAME>; review the patch")
    quiet = logging.getLogger("mlxrt_llm")
    quiet.disabled = True  # the self-test's recovery is not a call the model made
    try:
        result: object = patched(_SELF_TEST_TEXT, _SELF_TEST_TOOLS)
    except ValueError as error:
        result = f"ValueError: {error}"
    finally:
        quiet.disabled = False
    if result != _SELF_TEST_RESULT:
        problems.append(f"the lenient parser returned {result!r}, not {_SELF_TEST_RESULT!r}")
    if problems:
        raise PatchError(_refusal(problems))
    qwen3_coder.parse_tool_call = patched


def _source(owner: object, name: str) -> str:
    """The source of ``owner.name``, as installed (mlx-lm ships its .py files)."""
    return inspect.getsource(getattr(owner, name))


def _refusal(problems: list[str]) -> str:
    return (
        "mlxrt_llm: refusing to start mlx_lm.server with the lenient tool-call parser:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
        + "\nReview backends/llm/src/mlxrt_llm/ against the installed mlx-lm."
    )


TOOLS_CHECKPOINT_FLAG = "--tools-checkpoint"
SHARED_PREFIX_CHECKPOINT_FLAG = "--shared-prefix-checkpoint"


def main() -> None:
    ours = {TOOLS_CHECKPOINT_FLAG, SHARED_PREFIX_CHECKPOINT_FLAG}
    flags = ours & set(sys.argv[1:])
    sys.argv[1:] = [argument for argument in sys.argv[1:] if argument not in ours]
    tools = TOOLS_CHECKPOINT_FLAG in flags
    shared_prefix = SHARED_PREFIX_CHECKPOINT_FLAG in flags
    try:
        install()
        if tools or shared_prefix:
            checkpoints.install(tools=tools, shared_prefix=shared_prefix)
    except (PatchError, checkpoints.CheckpointPatchError) as error:
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    installed = ["lenient qwen3_coder tool-call parser"]
    if tools:
        installed.append("tool-definitions cache checkpoint")
    if shared_prefix:
        installed.append("shared-prefix cache checkpoint")
    print(
        f"mlxrt_llm: {', '.join(installed)} (mlx-lm {SUPPORTED_MLX_LM})",
        file=sys.stderr,
        flush=True,
    )
    from mlx_lm.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()

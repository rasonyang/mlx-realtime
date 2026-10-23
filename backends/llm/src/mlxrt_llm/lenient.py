"""A lenient wrapper around mlx-lm's qwen3_coder tool-call parser (M0 S1, G2).

Qwen3.6 sometimes opens a call with the bare function name::

    <tool_call>
    <transfer_to_agent>
    <parameter=queue>
    mobile_support
    </parameter>
    </function>
    </tool_call>

instead of ``<function=transfer_to_agent>``. mlx-lm's qwen3_coder parser finds no
``<function=…</function>`` block and raises ``ValueError("No function provided.")``; the server
logs it and drops the call, so the stream ends with ``finish_reason: tool_calls`` and no call
(M0: 11% of realistic transfer turns).

``lenient(parse)`` wraps the stock parser. Only when the stock parser fails, the call text has
no ``<function=`` tag, and it opens with ``<NAME>`` for a NAME the request declared as a tool,
is that tag rewritten to ``<function=NAME>`` (and a closing ``</NAME>``, if the model wrote one
instead of ``</function>``, to ``</function>``); the stock parser then runs on the result. In
every other case the stock parser's result or error stands, so a well-formed call parses
exactly as before and an undeclared name is never invented into a call.

The server hands the parser the text between ``<tool_call>`` and ``</tool_call>``, and the
request's ``tools`` list.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

logger = logging.getLogger("mlxrt_llm")

type Parser = Callable[[str, Any], Any]
"""``parse_tool_call(model_output, tools)``: a dict (or list of dicts) or ``ValueError``."""

# A bare opening tag: ``<NAME>``, NAME without spaces, ``=``, ``/`` or angle brackets.
_BARE_OPEN = re.compile(r"\A(\s*)<([^\s<>=/]+)>")


def declared_names(tools: object) -> frozenset[str]:
    """The function names of an OpenAI ``tools`` list:
    ``[{"type": "function", "function": {"name": …}}, …]``."""
    if not isinstance(tools, list):
        return frozenset()
    names: set[str] = set()
    for tool in cast(list[object], tools):
        if not isinstance(tool, Mapping):
            continue
        function = cast(Mapping[str, object], tool).get("function")
        if not isinstance(function, Mapping):
            continue
        name = cast(Mapping[str, object], function).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def rewrite(text: str, names: frozenset[str]) -> str | None:
    """``text`` with its bare ``<NAME>`` opening tag rewritten to ``<function=NAME>``.

    ``None`` when there is nothing to rewrite: the text already has a ``<function=`` tag, it
    does not open with a tag, the tag is not a declared name, or the call is never closed (by
    ``</function>`` or ``</NAME>``), as when generation stopped inside it.
    """
    if "<function=" in text:
        return None
    match = _BARE_OPEN.match(text)
    if match is None or match.group(2) not in names:
        return None
    name = match.group(2)
    fixed = f"{match.group(1)}<function={name}>{text[match.end() :]}"
    if "</function>" in fixed:
        return fixed
    closing = f"</{name}>"
    at = fixed.rfind(closing)
    if at < 0:
        return None
    return f"{fixed[:at]}</function>{fixed[at + len(closing) :]}"


def lenient(parse: Parser) -> Parser:
    """``parse`` that also accepts a call opened with a declared ``<NAME>`` (module docstring)."""

    def parse_tool_call(model_output: str, tools: Any = None) -> Any:
        try:
            return parse(model_output, tools)
        except ValueError:
            fixed = rewrite(model_output, declared_names(tools))
            if fixed is None:
                raise
        result = parse(fixed, tools)
        logger.warning("tool call written as <NAME> instead of <function=NAME>: recovered")
        return result

    parse_tool_call.__wrapped__ = parse  # pyright: ignore[reportFunctionMemberAccess]
    parse_tool_call.__mlxrt_lenient__ = True  # pyright: ignore[reportFunctionMemberAccess]
    return parse_tool_call


def is_lenient(parse: object) -> bool:
    """Whether ``parse`` is a parser this module wrapped."""
    return bool(getattr(parse, "__mlxrt_lenient__", False))

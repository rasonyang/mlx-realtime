"""The Qwen3.6 chat template (tests/fixtures/templates), rendered as mlx_lm.server renders it,
so tests can check offline that a message list is one the real template accepts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE = Path(__file__).parent / "fixtures" / "templates" / "qwen3_6_chat_template.jinja"

type Json = dict[str, Any]


def render(messages: Sequence[Json], tools: Sequence[Json] = ()) -> str:
    """The prompt mlx_lm.server builds: tool-call arguments parsed to objects (its
    ``convert_chat``), then the template with the request's tools, a generation prompt and
    thinking off. A shape the template rejects raises ``TemplateError``."""
    converted = json.loads(json.dumps(list(messages)))
    for message in converted:
        for entry in message.get("tool_calls", []):
            entry["function"]["arguments"] = json.loads(entry["function"]["arguments"])
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)  # pyright: ignore[reportUnknownLambdaType]
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        messages=converted,
        tools=list(tools) or None,
        add_generation_prompt=True,
        enable_thinking=False,
    )

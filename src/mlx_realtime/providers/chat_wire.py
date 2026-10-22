"""The chat-completions wire format as mlx_lm.server 0.32.2 speaks it (ARCHITECTURE §5.2, §5.3).

Requests: the conversation is adapted to what the Qwen chat template accepts.

- All leading system messages become one system message, first.
- A system message later in the conversation becomes a user message in ``system_note``.
- Adjacent assistant messages merge: text, then tool calls, as one assistant message.
- A conversation with no user message (a greeting asked for with ``response.create``) gets a
  synthetic user cue at its end.

Responses: one ``chat.completion.chunk`` per SSE event. ``delta.content`` is text,
``delta.reasoning`` is dropped (thinking is off), and ``delta.tool_calls`` holds complete calls:
the server parses each call (qwen3_coder) before sending it, so a call that does not check out
is a bad response, never a partial call. The usage chunk (``choices: []``) precedes
``data: [DONE]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from mlx_realtime.providers.base import (
    FinishReason,
    LLMDone,
    Message,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolChoice,
    ToolDef,
    Usage,
)

type Json = dict[str, object]

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


@dataclass(frozen=True, slots=True)
class Sampling:
    max_completion_tokens: int
    temperature: float
    top_p: float


def request_body(
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
    *,
    tool_choice: ToolChoice,
    max_output_tokens: int | None,
    sampling: Sampling,
    model: str | None,
    cue: str,
    system_note: str,
) -> Json:
    limit = sampling.max_completion_tokens
    body: Json = {
        "messages": chat_messages(messages, cue=cue, system_note=system_note),
        "stream": True,
        # Without include_usage the server fails mid-stream (ARCHITECTURE §5.3).
        "stream_options": {"include_usage": True},
        "max_completion_tokens": limit
        if max_output_tokens is None
        else min(limit, max_output_tokens),
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        # Off server-wide too (--chat-template-args): neither setting alone can turn it on.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # The server ignores tool_choice, so it is enforced here: "none" sends no tools at all.
    if tool_choice == "auto" and tools:
        body["tools"] = [_tool_spec(tool) for tool in tools]
    if model is not None:
        body["model"] = model  # exactly the server's --model, or another model gets loaded
    return body


def chat_messages(messages: Sequence[Message], *, cue: str, system_note: str) -> list[Json]:
    leading: list[str] = []
    rest = list(messages)
    while rest and rest[0].role == "system":
        leading.append(rest.pop(0).content)
    adapted: list[Json] = []
    if leading:
        adapted.append({"role": "system", "content": "\n\n".join(leading)})
    for message in rest:
        match message.role:
            case "system":
                adapted.append(
                    {"role": "user", "content": system_note.format(text=message.content)}
                )
            case "assistant":
                _add_assistant(adapted, message)
            case "tool":
                adapted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
            case "user":
                adapted.append({"role": "user", "content": message.content})
    if not any(entry["role"] == "user" for entry in adapted):
        adapted.append({"role": "user", "content": cue})
    return adapted


def _add_assistant(adapted: list[Json], message: Message) -> None:
    calls = [_call_spec(call) for call in message.tool_calls]
    previous = adapted[-1] if adapted else None
    if previous is not None and previous["role"] == "assistant":
        texts = [text for text in (str(previous["content"]), message.content) if text]
        previous["content"] = "\n".join(texts)
        if calls:
            previous["tool_calls"] = [*cast(list[Json], previous.get("tool_calls", [])), *calls]
        return
    entry: Json = {"role": "assistant", "content": message.content}
    if calls:
        entry["tool_calls"] = calls
    adapted.append(entry)


def _call_spec(call: ToolCall) -> Json:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments},
    }


def _tool_spec(tool: ToolDef) -> Json:
    function: Json = {"name": tool.name}
    if tool.description is not None:
        function["description"] = tool.description
    if tool.parameters is not None:
        function["parameters"] = dict(tool.parameters)
    return {"type": "function", "function": function}


class ChunkReader:
    """Reads the stream's chunks in order and keeps what ``finish`` needs."""

    def __init__(self) -> None:
        self.finish_reason: FinishReason | None = None
        self.usage: Usage | None = None
        self.calls = 0
        self.reasoning_chunks = 0

    def read(self, data: str) -> list[TextDelta | ToolCall]:
        """The deltas of one ``data:`` payload; calls come first, as the server sends them."""
        chunk = _object(_loads(data), "chunk")
        if "usage" in chunk and chunk["usage"] is not None:
            self.usage = _usage(_object(chunk["usage"], "usage"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = _object(cast(list[object], choices)[0], "choice")
        reason = choice.get("finish_reason")
        if reason is not None:
            if not isinstance(reason, str) or reason not in _FINISH_REASONS:
                raise _bad(f"unknown finish_reason {reason!r}")
            self.finish_reason = _FINISH_REASONS[reason]
        delta = _object(choice.get("delta", {}), "delta")
        deltas: list[TextDelta | ToolCall] = []
        raw_calls = delta.get("tool_calls")
        if raw_calls is not None:
            if not isinstance(raw_calls, list):
                raise _bad("tool_calls is not a list")
            deltas += [_tool_call(raw) for raw in cast(list[object], raw_calls)]
            self.calls += len(deltas)
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise _bad("content is not a string")
            if content:
                deltas.append(TextDelta(content))
        if delta.get("reasoning"):
            self.reasoning_chunks += 1  # dropped: thinking is off
        return deltas

    def finish(self) -> LLMDone:
        """The end of the stream, at ``[DONE]``."""
        if self.finish_reason is None:
            raise _bad("the stream ended without a finish_reason")
        if self.finish_reason == "tool_calls" and self.calls == 0:
            raise _bad(
                "finish_reason tool_calls without a call: the server dropped a malformed call"
            )
        return LLMDone(self.finish_reason, self.usage)


def _tool_call(raw: object) -> ToolCall:
    call = _object(raw, "tool call")
    function = _object(call.get("function"), "tool call function")
    call_id, name, arguments = call.get("id"), function.get("name"), function.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise _bad("a tool call without an id")
    if not isinstance(name, str) or not name:
        raise _bad("a tool call without a name")
    if not isinstance(arguments, str) or not isinstance(_loads(arguments), dict):
        raise _bad(f"tool call {name!r}: arguments are not a JSON object")
    return ToolCall(call_id, name, arguments)


def _usage(usage: Json) -> Usage:
    details = usage.get("prompt_tokens_details")
    cached = _object(details, "prompt_tokens_details").get("cached_tokens", 0) if details else 0
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not all(isinstance(value, int) for value in (prompt, completion, cached)):
        raise _bad("usage counts are not integers")
    return Usage(cast(int, prompt), cast(int, completion), cast(int, cached))


def _loads(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError as error:
        raise _bad(f"invalid JSON: {error}") from None


def _object(value: object, what: str) -> Json:
    if not isinstance(value, Mapping):
        raise _bad(f"{what} is not an object")
    return {str(key): item for key, item in cast(Mapping[object, object], value).items()}


def _bad(message: str) -> ProviderError:
    return ProviderError("bad_response", f"llm stream: {message}")

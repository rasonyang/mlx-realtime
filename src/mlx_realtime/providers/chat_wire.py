"""The chat-completions wire format as mlx_lm.server 0.32.2 speaks it (ARCHITECTURE §5.2, §5.3).

Requests: the conversation is adapted to what the Qwen chat template accepts (``Shaping``).

- All leading system messages become one system message, first.
- A system message later in the conversation becomes a user message in ``system_note``.
- Adjacent assistant messages merge: text, then tool calls, as one assistant message. Text that
  follows a call is not merged here; it is an interlude (below).
- Tool results follow their call (M5). Each assistant message with tool calls is followed at
  once by the results of its calls, in call order, so the template renders
  ``<tool_call>`` then ``<tool_response>`` with nothing between. What the conversation holds
  between a call and its result is the call's interlude:

  - an assistant text (AICC's wait line) before any user message of the interlude is merged into
    the call's message, before the call (``interlude: merge``, the default), or placed after
    the results (``interlude: after_output``);
  - more calls before any user message join the call's message (parallel calls);
  - everything else (user notes, caller speech, later assistant text) keeps its order and comes
    after the results.

  A call still unanswered when the request is built gets a placeholder result
  (``unanswered: pending``, the default), is replayed without a result (``keep``), or is left
  out (``drop``). A result whose call is not in the conversation is dropped.
- A conversation with no user message (a greeting asked for with ``response.create``) gets a
  synthetic user cue, right after the system message: it opens the conversation, before anything
  the assistant already said.

Responses: one ``chat.completion.chunk`` per SSE event. ``delta.content`` is text,
``delta.reasoning`` is dropped (thinking is off), and ``delta.tool_calls`` holds complete calls:
the server parses each call (qwen3_coder) before sending it, so a call that does not check out
is a bad response, never a partial call. The usage chunk (``choices: []``) precedes
``data: [DONE]``.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

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
type Interlude = Literal["merge", "after_output"]
type Unanswered = Literal["pending", "keep", "drop"]

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


@dataclass(frozen=True, slots=True)
class Shaping:
    """How the conversation is adapted to the Qwen template (the module docstring)."""

    cue: str
    system_note: str
    interlude: Interlude
    unanswered: Unanswered
    pending_output: str  # the placeholder result of an unanswered call


def request_body(
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
    *,
    tool_choice: ToolChoice,
    max_output_tokens: int | None,
    sampling: Sampling,
    model: str | None,
    shaping: Shaping,
) -> Json:
    limit = sampling.max_completion_tokens
    body: Json = {
        "messages": chat_messages(messages, shaping),
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


def chat_messages(messages: Sequence[Message], shaping: Shaping) -> list[Json]:
    leading: list[str] = []
    rest = list(messages)
    while rest and rest[0].role == "system":
        leading.append(rest.pop(0).content)
    notes = [
        Message("user", shaping.system_note.format(text=message.content))
        if message.role == "system"
        else message
        for message in rest
    ]
    shaped = _merge_adjacent(_place_results(_merge_adjacent(notes), shaping))
    adapted: list[Json] = []
    if leading:
        adapted.append({"role": "system", "content": "\n\n".join(leading)})
    if not any(message.role == "user" for message in shaped):
        # The template needs a user message. The cue opens the conversation, before anything
        # the assistant already said (a greeting, a call it made).
        adapted.append({"role": "user", "content": shaping.cue})
    adapted += [_entry(message) for message in shaped]
    return adapted


def _merge_adjacent(messages: Sequence[Message]) -> list[Message]:
    """Adjacent assistant messages become one, text before calls; text after a call stays apart
    (it is the call's interlude, which ``_place_results`` shapes)."""
    merged: list[Message] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (
            message.role == "assistant"
            and previous is not None
            and previous.role == "assistant"
            and (not previous.tool_calls or not message.content)
        ):
            merged[-1] = _join(previous, message)
        else:
            merged.append(message)
    return merged


def _join(first: Message, second: Message, *, calls: bool = True) -> Message:
    text = "\n".join(part for part in (first.content, second.content) if part)
    tool_calls = first.tool_calls + second.tool_calls if calls else first.tool_calls
    return replace(first, content=text, tool_calls=tool_calls)


def _place_results(messages: Sequence[Message], shaping: Shaping) -> list[Message]:
    """Each call message followed at once by its results; its interlude after them."""
    placed: list[Message] = []
    answered: set[str] = set()  # calls whose block is placed
    queue = deque(messages)
    while queue:
        message = queue.popleft()
        if message.role == "tool":
            if message.tool_call_id in answered:
                placed.append(message)  # a second result for a placed call stays where it is
            continue  # a result whose call is not in the conversation: dropped
        if message.role != "assistant" or not message.tool_calls:
            placed.append(message)
            continue
        call, results, interlude = _block(message, queue, shaping)
        order = [entry.call_id for entry in call.tool_calls]
        missing = [call_id for call_id in order if call_id not in results]
        match shaping.unanswered:
            case "pending":
                for call_id in missing:
                    results[call_id] = Message("tool", shaping.pending_output, tool_call_id=call_id)
            case "drop":
                kept = tuple(entry for entry in call.tool_calls if entry.call_id in results)
                call = replace(call, tool_calls=kept)
            case "keep":
                pass
        if call.content or call.tool_calls:
            placed.append(call)
        placed += [results[call_id] for call_id in order if call_id in results]
        answered.update(order)
        queue.extendleft(reversed(interlude))  # shaped in turn: it may hold calls of its own
    return placed


def _block(
    call: Message, queue: deque[Message], shaping: Shaping
) -> tuple[Message, dict[str, Message], list[Message]]:
    """Take ``call``'s results and interlude from the front of ``queue``, up to its last
    result (or the end, while a call is unanswered)."""
    pending = {entry.call_id for entry in call.tool_calls}
    results: dict[str, Message] = {}
    interlude: list[Message] = []
    heard_user = False
    while queue and pending:
        message = queue.popleft()
        if message.role == "tool" and message.tool_call_id in pending:
            results[message.tool_call_id] = message
            pending.discard(message.tool_call_id)
        elif message.role == "assistant" and not heard_user and message.tool_calls:
            pending.update(entry.call_id for entry in message.tool_calls)
            if shaping.interlude == "merge" or not message.content:
                call = _join(call, message)
            else:
                call = replace(call, tool_calls=call.tool_calls + message.tool_calls)
                interlude.append(replace(message, tool_calls=()))
        elif message.role == "assistant" and not heard_user and shaping.interlude == "merge":
            call = _join(call, message)
        else:
            heard_user = heard_user or message.role == "user"
            interlude.append(message)
    return call, results, interlude


def _entry(message: Message) -> Json:
    match message.role:
        case "assistant":
            entry: Json = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                entry["tool_calls"] = [_call_spec(call) for call in message.tool_calls]
            return entry
        case "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content,
            }
        case _:
            return {"role": message.role, "content": message.content}


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

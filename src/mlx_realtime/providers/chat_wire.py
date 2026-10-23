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

Sampling (M0 S1, G2): a request with tools attached is sent at ``tool_temperature`` /
``tool_top_p`` (0 and 1.0: Qwen3.6 makes 19 of 20 expected calls at temperature 0, 16–19 at
0.7); one without tools at ``temperature`` / ``top_p``, the server's own defaults (``--temp 0.7
--top-p 0.8 --top-k 20 --min-p 0.0``). top_k and min_p are left to the server's flags.

Warm-up (M0 S2): ``warmup_messages`` is the prefix of the next real request that a one-token
request can put in the server's prompt cache before that request is made (the real request's
own messages up to its last user message). The server saves cache checkpoints at the end of the
system message and at the end of the last user message, and only for a request that ends with a
user message; the system checkpoint also holds the next message's role token. A conversation
with no user message yet is warmed as ``[system, assistant "", cue]``, so that its system
checkpoint ends before the role token and serves a history that opens with the assistant (a
spoken greeting) as well as the cue.

Responses: one ``chat.completion.chunk`` per SSE event. ``delta.content`` is text,
``delta.reasoning`` is dropped (thinking is off), and ``delta.tool_calls`` holds complete calls:
the server parses each call (qwen3_coder) before sending it, so a call that does not check out
is a bad response, never a partial call. The usage chunk (``choices: []``) precedes
``data: [DONE]``. A reply made without tools may carry a JSON "call" in its text (M0 S1:
``{"tool_name": "repair_status", "tool_input": {…}}``); ``PseudoCallFilter`` takes it out before
it can reach the TTS.
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
    temperature: float  # requests without tools
    top_p: float
    tool_temperature: float = 0.0  # requests with tools attached
    tool_top_p: float = 1.0

    def of(self, body: Mapping[str, object]) -> tuple[float, float]:
        """``(temperature, top_p)`` for a request body: tool sampling when it carries tools."""
        if "tools" in body:
            return self.tool_temperature, self.tool_top_p
        return self.temperature, self.top_p


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
        # Off server-wide too (--chat-template-args): neither setting alone can turn it on.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # The server ignores tool_choice, so it is enforced here: "none" sends no tools at all.
    if tool_choice == "auto" and tools:
        body["tools"] = [_tool_spec(tool) for tool in tools]
    body["temperature"], body["top_p"] = sampling.of(body)
    if model is not None:
        body["model"] = model  # exactly the server's --model, or another model gets loaded
    return body


def warmup_messages(adapted: Sequence[Json], cue: str) -> list[Json]:
    """The warm-up prefix of a real request's adapted messages (the module docstring): up to
    its last user message, or ``[system, assistant "", cue]`` while the conversation has none
    (``adapted`` then holds the cue, which ``chat_messages`` added)."""
    system = [message for message in adapted[:1] if message["role"] == "system"]
    users = [index for index, message in enumerate(adapted) if message["role"] == "user"]
    if not users or (len(users) == 1 and adapted[users[0]]["content"] == cue):
        return [*system, {"role": "assistant", "content": ""}, {"role": "user", "content": cue}]
    return list(adapted[: users[-1] + 1])


def prefix_key(body: Mapping[str, object]) -> str:
    """What a warm-up shares with the requests that should wait for it: the system message and
    the tools, the part of the prompt the server renders first."""
    messages = cast(list[Json], body["messages"])
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    return json.dumps([system, body.get("tools")], ensure_ascii=False, sort_keys=True)


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

    @property
    def dropped_call(self) -> bool:
        """The stream ended with ``finish_reason: tool_calls`` and no call: the server's parser
        rejected the call the model wrote."""
        return self.finish_reason == "tool_calls" and self.calls == 0

    def finish(self) -> LLMDone:
        """The end of the stream, at ``[DONE]``."""
        if self.finish_reason is None:
            raise _bad("the stream ended without a finish_reason")
        if self.dropped_call:
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


# Pseudo-calls (M0 S1): with no tools sent, the model may still "call" one in its text.
_NAME_KEYS = frozenset({"name", "tool", "tool_name", "function", "function_name", "action"})
_ARGUMENT_KEYS = frozenset(
    {"arguments", "parameters", "params", "args", "input", "tool_input", "action_input"}
)
_WRAPPER_KEYS = frozenset({"tool_call", "tool_calls", "function_call", "function_calls"})
_FENCE = "```"
_MAX_HELD = 4000  # characters held back for one candidate; past this it is released as text


def is_pseudo_call(text: str) -> bool:
    """Whether ``text`` is a JSON object shaped like a function call, alone or in a code fence:
    a name (``name``, ``tool_name``, ``function``, …) with arguments (``arguments``,
    ``tool_input``, …), or such an object (or a list of them) under ``tool_call``,
    ``function_call`` or ``function``. Anything else, invalid JSON included, is not."""
    candidate = text.strip()
    if candidate.startswith(_FENCE):
        if len(candidate) < 2 * len(_FENCE) or not candidate.endswith(_FENCE):
            return False
        first, _, rest = candidate[len(_FENCE) : -len(_FENCE)].partition("\n")
        tag = first.strip()
        candidate = rest if not tag or tag.replace("_", "").isalpha() else first + rest
    try:
        value: object = json.loads(candidate)
    except ValueError:
        return False
    return isinstance(value, dict) and _call_shaped(cast(object, value))


def _call_shaped(value: object) -> bool:
    if isinstance(value, list):
        items = cast(list[object], value)
        return bool(items) and all(_call_shaped(item) for item in items)
    if not isinstance(value, dict):
        return False
    fields = cast(dict[str, object], value)
    wrapped = [fields[key] for key in _WRAPPER_KEYS if key in fields]
    if wrapped:
        return any(_call_shaped(inner) for inner in wrapped)
    function = fields.get("function")
    if isinstance(function, dict) and _call_shaped(cast(object, function)):
        return True  # an OpenAI tool_calls entry: {"type": "function", "function": {…}}
    named = any(isinstance(fields.get(key), str) and fields.get(key) for key in _NAME_KEYS)
    return named and any(key in fields for key in _ARGUMENT_KEYS)


class PseudoCallFilter:
    """Takes JSON pseudo-calls out of a reply's streamed text (applied when no tools were sent).

    Text flows through unchanged until a ``{`` whose next non-blank character is ``"`` (a JSON
    object) or a code fence starts. From there it is held back until the object (or fence)
    closes; it is then dropped if ``is_pseudo_call`` says so, and released as it was otherwise.
    Held text that never closes is released at ``flush`` (or once it passes ``_MAX_HELD``):
    nothing is dropped that is not a complete, valid pseudo-call.
    """

    def __init__(self) -> None:
        self.dropped: list[str] = []
        self._held = ""
        self._mode: Literal["object", "ticks", "fence"] | None = None
        self._keyed = False  # an object's first non-blank character was the quote of a key
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def push(self, text: str) -> str:
        """The text that can be released now."""
        if self._mode is None and "{" not in text and "`" not in text:
            return text
        return "".join([self._feed(char) for char in text])

    def flush(self) -> str:
        """At the end of the stream: whatever is still held, as text."""
        return self._release()

    def _feed(self, char: str) -> str:
        match self._mode:
            case None:
                if char == "{":
                    self._start("object", char)
                    self._depth = 1
                    return ""
                if char == "`":
                    self._start("ticks", char)
                    return ""
                return char
            case "ticks":
                if char == "`":
                    self._held += char
                    if self._held == _FENCE:
                        self._mode = "fence"
                    return ""
                return self._release() + self._feed(char)
            case "fence":
                self._held += char
                if len(self._held) > 2 * len(_FENCE) and self._held.endswith(_FENCE):
                    return self._decide()
                return self._overflow()
            case "object":
                return self._feed_object(char)

    def _feed_object(self, char: str) -> str:
        self._held += char
        if not self._keyed:
            if char.isspace():
                return self._overflow()
            if char != '"':
                return self._release()  # "{" then something other than a key: no JSON object
            self._keyed = True
        if self._in_string:
            if self._escaped:
                self._escaped = False
            elif char == "\\":
                self._escaped = True
            elif char == '"':
                self._in_string = False
        elif char == '"':
            self._in_string = True
        elif char == "{":
            self._depth += 1
        elif char == "}":
            self._depth -= 1
            if self._depth == 0:
                return self._decide()
        return self._overflow()

    def _start(self, mode: Literal["object", "ticks"], char: str) -> None:
        self._mode, self._held = mode, char
        self._keyed, self._depth, self._in_string, self._escaped = False, 0, False, False

    def _decide(self) -> str:
        if is_pseudo_call(self._held):
            self.dropped.append(self._held)
            self._mode, self._held = None, ""
            return ""
        return self._release()

    def _overflow(self) -> str:
        return self._release() if len(self._held) > _MAX_HELD else ""

    def _release(self) -> str:
        held, self._mode, self._held = self._held, None, ""
        return held

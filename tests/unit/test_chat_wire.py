"""The chat-completions wire format: message adaptation for the Qwen template, request bodies
and stream chunks (ARCHITECTURE §5.2)."""

from __future__ import annotations

import json

import pytest

from mlx_realtime.config import PENDING_TOOL_OUTPUT
from mlx_realtime.providers.base import Message, ProviderError, TextDelta, ToolCall, ToolDef, Usage
from mlx_realtime.providers.chat_wire import (
    ChunkReader,
    PseudoCallFilter,
    Sampling,
    Shaping,
    chat_messages,
    is_pseudo_call,
    prefix_key,
    request_body,
    warmup_messages,
)
from mlx_realtime.providers.sse import sse_data

CUE = "(The conversation has just started.)"
NOTE = "(System note: {text})"
SHAPING = Shaping(CUE, NOTE, "merge", "pending", PENDING_TOOL_OUTPUT)


def adapt(*messages: Message) -> list[dict[str, object]]:
    return chat_messages(messages, SHAPING)


def test_leading_system_messages_become_one() -> None:
    assert adapt(Message("system", "A."), Message("system", "B."), Message("user", "hi")) == [
        {"role": "system", "content": "A.\n\nB."},
        {"role": "user", "content": "hi"},
    ]


def test_a_later_system_message_becomes_a_marked_user_note() -> None:
    adapted = adapt(
        Message("system", "Be brief."),
        Message("user", "hi"),
        Message("assistant", "Hello."),
        Message("system", "the caller is waiting"),
    )
    assert adapted[-1] == {"role": "user", "content": "(System note: the caller is waiting)"}
    assert [entry["role"] for entry in adapted] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="empty"),
        pytest.param([Message("system", "Greet the caller.")], id="instructions-only"),
        pytest.param([Message("assistant", "Hello?")], id="assistant-only"),
    ],
)
def test_a_conversation_without_a_user_message_gets_the_cue(messages: list[Message]) -> None:
    adapted = adapt(*messages)
    first = 1 if adapted[0]["role"] == "system" else 0
    assert adapted[first] == {"role": "user", "content": CUE}  # it opens the conversation
    assert sum(entry["role"] == "user" for entry in adapted) == 1


def test_adjacent_assistant_messages_and_calls_merge() -> None:
    call = ToolCall("p1", "lookup", '{"id": 1}')
    adapted = adapt(
        Message("user", "查一下"),
        Message("assistant", "好的，"),
        Message("assistant", "", tool_calls=(call,)),
        Message("tool", '{"ok": true}', tool_call_id="p1"),
    )
    assert adapted[1:] == [
        {
            "role": "assistant",
            "content": "好的，",
            "tool_calls": [
                {
                    "id": "p1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "p1", "content": '{"ok": true}'},
    ]


def body(**changes: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "tool_choice": "auto",
        "max_output_tokens": None,
        "sampling": Sampling(512, 0.7, 0.8),
        "model": None,
        "shaping": SHAPING,
    }
    arguments.update(changes)
    return request_body(
        [Message("user", "hi")],
        [ToolDef("f", None, None)],
        **arguments,  # pyright: ignore[reportArgumentType]
    )


def test_request_body() -> None:
    plain = body()
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}
    assert plain["stream_options"] == {"include_usage": True}
    assert plain["tools"] == [{"type": "function", "function": {"name": "f"}}]
    assert "model" not in plain
    assert "tools" not in body(tool_choice="none")
    assert body(model="/m")["model"] == "/m"
    assert body(max_output_tokens=4096)["max_completion_tokens"] == 512
    assert body(max_output_tokens=7)["max_completion_tokens"] == 7
    assert "seed" not in plain  # a seed would turn off the server's batching


def chunk(delta: dict[str, object], finish: str | None = None) -> str:
    choice = {"index": 0, "finish_reason": finish, "delta": {"role": "assistant", **delta}}
    return json.dumps({"object": "chat.completion.chunk", "choices": [choice]})


def test_chunks_become_deltas() -> None:
    reader = ChunkReader()
    call = {
        "function": {"name": "f", "arguments": "{}"},
        "type": "function",
        "id": "c1",
        "index": 0,
    }
    assert reader.read(chunk({"content": "你好"})) == [TextDelta("你好")]
    assert reader.read(chunk({"reasoning": "hm"})) == []
    assert reader.read(chunk({"content": "，", "tool_calls": [call]})) == [
        ToolCall("c1", "f", "{}"),
        TextDelta("，"),
    ]
    assert reader.read(chunk({}, "tool_calls")) == []
    usage = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    assert reader.read(json.dumps({"choices": [], "usage": usage})) == []
    done = reader.finish()
    assert (done.finish_reason, done.usage) == ("tool_calls", Usage(9, 4, 0))
    assert reader.reasoning_chunks == 1


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        "[1]",
        chunk({"content": 5}),
        chunk({"tool_calls": "x"}),
        json.dumps({"choices": [], "usage": {"prompt_tokens": "9", "completion_tokens": 1}}),
    ],
)
def test_malformed_chunks(data: str) -> None:
    with pytest.raises(ProviderError) as failed:
        ChunkReader().read(data)
    assert failed.value.kind == "bad_response"


def test_finish_needs_a_reason() -> None:
    with pytest.raises(ProviderError):
        ChunkReader().finish()


async def test_sse_events_from_split_chunks() -> None:
    raw = [
        b": keepalive 512/2048\n\n",
        b'data: {"a"',
        b": 1}\r\n\r\ndata: [DONE]\n",
        b"\nevent: x\ndata: one\ndata: two\n\n",
        b"data: cut off\n",  # the stream ends inside this event: never dispatched
    ]

    async def chunks():
        for part in raw:
            yield part

    assert [data async for data in sse_data(chunks())] == ['{"a": 1}', "[DONE]", "one\ntwo"]


# LLM hardening (M0 S1/S2, G2)


def test_requests_with_tools_are_greedy() -> None:
    """G2: tools attached → tool_temperature / tool_top_p; otherwise the configured sampling."""
    with_tools = body()
    assert (with_tools["temperature"], with_tools["top_p"]) == (0.0, 1.0)
    without = body(tool_choice="none")
    assert (without["temperature"], without["top_p"]) == (0.7, 0.8)
    no_tools = request_body(
        [Message("user", "hi")],
        [],
        tool_choice="auto",
        max_output_tokens=None,
        sampling=Sampling(512, 0.7, 0.8),
        model=None,
        shaping=SHAPING,
    )
    assert (no_tools["temperature"], no_tools["top_p"]) == (0.7, 0.8)
    custom = body(sampling=Sampling(512, 0.7, 0.8, tool_temperature=0.2, tool_top_p=0.9))
    assert (custom["temperature"], custom["top_p"]) == (0.2, 0.9)


CALL = ToolCall("p1", "transfer_to_agent", '{"queue": "mobile_support"}')


def test_the_warmup_prefix_ends_at_the_last_user_message() -> None:
    """The next request (the call's result) will extend exactly this prefix; the placeholder
    result of the still unanswered call is not part of it."""
    adapted = adapt(
        Message("system", "Phase 2."),
        Message("assistant", "您好。"),
        Message("user", "转人工"),
        Message("assistant", "", tool_calls=(CALL,)),
    )
    assert adapted[-1]["content"] == PENDING_TOOL_OUTPUT
    assert warmup_messages(adapted, CUE) == [
        {"role": "system", "content": "Phase 2."},
        {"role": "assistant", "content": "您好。"},
        {"role": "user", "content": "转人工"},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([Message("system", "Greet.")], id="empty"),
        pytest.param([Message("system", "Greet."), Message("assistant", "您好。")], id="greeted"),
    ],
)
def test_the_warmup_prefix_before_any_user_message(messages: list[Message]) -> None:
    """Its system checkpoint must serve a history that opens with the assistant (the greeting)
    as well as the cue, so the message after the system is an empty assistant one."""
    assert warmup_messages(adapt(*messages), CUE) == [
        {"role": "system", "content": "Greet."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": CUE},
    ]


def test_prefix_key_is_the_system_message_and_the_tools() -> None:
    def key(messages: list[Message], tools: list[ToolDef]) -> str:
        return prefix_key(
            request_body(
                messages,
                tools,
                tool_choice="auto",
                max_output_tokens=None,
                sampling=Sampling(512, 0.7, 0.8),
                model=None,
                shaping=SHAPING,
            )
        )

    system, f = Message("system", "S."), ToolDef("f", None, None)
    first = key([system, Message("user", "hi")], [f])
    later = [system, Message("user", "hi"), Message("assistant", "嗯"), Message("user", "x")]
    assert key(later, [f]) == first
    assert key([system, Message("user", "hi")], [ToolDef("g", None, None)]) != first
    assert key([Message("system", "T."), Message("user", "hi")], [f]) != first
    assert key([Message("user", "hi")], [f]) != first


def test_a_dropped_call() -> None:
    reader = ChunkReader()
    reader.read(chunk({}, "tool_calls"))
    assert reader.dropped_call
    with pytest.raises(ProviderError, match="dropped a malformed call"):
        reader.finish()
    stop = ChunkReader()
    stop.read(chunk({"content": "好"}, "stop"))
    assert not stop.dropped_call


B1_PSEUDO_CALL = (  # M0 S1, the reply to an RMA lookup made without tools
    '好的，我查一下。\n{"tool_name": "repair_status", "tool_input": {"rma_number": "RMA2048"}}'
)


def filtered(*pieces: str) -> tuple[str, list[str]]:
    texts = PseudoCallFilter()
    out = "".join(texts.push(piece) for piece in pieces) + texts.flush()
    return out, texts.dropped


@pytest.mark.parametrize("cut", range(len(B1_PSEUDO_CALL) + 1))
def test_the_pseudo_call_m0_saw_is_removed_wherever_the_stream_splits_it(cut: int) -> None:
    out, dropped = filtered(B1_PSEUDO_CALL[:cut], B1_PSEUDO_CALL[cut:])
    assert out == "好的，我查一下。\n"
    assert dropped == ['{"tool_name": "repair_status", "tool_input": {"rma_number": "RMA2048"}}']


@pytest.mark.parametrize(
    "call",
    [
        '{"name": "hangup", "arguments": {}}',
        '{"name": "repair_status", "parameters": {"rma": "RMA1"}}',
        '{"function": "transfer_to_agent", "arguments": "{\\"queue\\": \\"x\\"}"}',
        '{"tool": "kb_search_slow", "input": {"query": "违约金 {多少}"}}',
        '{"action": "hangup", "action_input": {}}',
        '{"tool_call": {"name": "hangup", "arguments": {}}}',
        '{"tool_calls": [{"name": "hangup", "arguments": {}}]}',
        '{"type": "function", "function": {"name": "hangup", "arguments": "{}"}}',
        '```json\n{"name": "hangup", "arguments": {}}\n```',
        '```\n{"name": "hangup", "arguments": {}}\n```',
        '{\n  "name": "hangup",\n  "arguments": {"isFarewellSpoken": true}\n}',
    ],
)
def test_pseudo_call_shapes(call: str) -> None:
    assert is_pseudo_call(call)
    out, dropped = filtered("请稍等。", call, "好的。")
    assert (out, dropped) == ("请稍等。好的。", [call])


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("请按{1}键确认。", id="braces-in-speech"),
        pytest.param("价格是 {大约} 一百元。", id="braces-not-json"),
        pytest.param('返回 {"found": 1, "status": "维修中"}。', id="json-not-a-call"),
        pytest.param('{"name": "张三"}', id="a-name-alone"),
        pytest.param('{"arguments": {}}', id="arguments-alone"),
        pytest.param('{"name": 3, "arguments": {}}', id="a-name-that-is-not-text"),
        pytest.param("{}", id="empty-object"),
        pytest.param("一个 ` 反引号和 `` 两个。", id="backticks"),
        pytest.param("```\n普通文本\n```", id="a-fence-around-text"),
        pytest.param('{"name": "hangup", "arguments": {', id="never-closed"),
        pytest.param("左括号 { 没有右括号", id="an-open-brace"),
        pytest.param('{"name": "a\\"}", "x": "{"}', id="braces-and-quotes-in-strings"),
    ],
)
def test_everything_else_passes_unchanged(text: str) -> None:
    assert filtered(text) == (text, [])
    assert filtered(*text) == (text, [])  # one character per delta


def test_text_after_an_open_brace_waits_for_it_to_close() -> None:
    texts = PseudoCallFilter()
    assert texts.push('好的。{"name": "hangup", ') == "好的。"
    assert texts.push('"arguments": {}}') == ""
    assert texts.push("再见。") == "再见。"
    assert texts.flush() == ""


def test_a_held_object_is_released_past_the_limit() -> None:
    texts = PseudoCallFilter()
    long = '{"name": "x", "note": "' + "长" * 5000 + '"}'
    assert texts.push(long) + texts.flush() == long
    assert texts.dropped == []

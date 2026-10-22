"""The chat-completions wire format: message adaptation for the Qwen template, request bodies
and stream chunks (ARCHITECTURE §5.2)."""

from __future__ import annotations

import json

import pytest

from mlx_realtime.providers.base import Message, ProviderError, TextDelta, ToolCall, ToolDef, Usage
from mlx_realtime.providers.chat_wire import ChunkReader, Sampling, chat_messages, request_body
from mlx_realtime.providers.sse import sse_data

CUE = "(The conversation has just started.)"
NOTE = "(System note: {text})"


def adapt(*messages: Message) -> list[dict[str, object]]:
    return chat_messages(messages, cue=CUE, system_note=NOTE)


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
    assert adapted[-1] == {"role": "user", "content": CUE}
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
        "cue": CUE,
        "system_note": NOTE,
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

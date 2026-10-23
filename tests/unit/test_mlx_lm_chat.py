"""``MlxLmChatLLM`` against a fake mlx_lm.server: request shaping, the SSE stream, every deadline,
and cancellation at each await point."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing

import httpx2
import pytest

from mlx_realtime.config import MlxLmChatConfig
from mlx_realtime.providers.base import (
    Delta,
    LLMDone,
    Message,
    ProviderError,
    ProviderTimeout,
    TextDelta,
    ToolCall,
    ToolDef,
    Usage,
)
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from tests.fakes.mlx_lm import (
    Calls,
    Close,
    Done,
    Fail,
    FakeMlxLm,
    Finish,
    Gate,
    Hold,
    Keepalive,
    Reasoning,
    Sleep,
    Text,
    reply,
    tool_call,
)
from tests.fakes.mlx_lm import Usage as UsageChunk

ASK = [Message("system", "你是客服助手。"), Message("user", "你好")]
LOOKUP = ToolDef("lookup", "Look an account up.", {"type": "object", "properties": {}})


@pytest.fixture
def fake() -> FakeMlxLm:
    return FakeMlxLm()


@pytest.fixture
def base_url(fake: FakeMlxLm) -> Iterator[str]:
    with fake.serving() as url:
        yield url


TIMEOUTS = ("connect_ms", "first_item_ms", "idle_ms", "total_ms", "warmup_ms")


def provider(base_url: str, **overrides: object) -> MlxLmChatLLM:
    """A provider with short deadlines; deadline keys go to ``timeouts``, the rest to the top."""
    timeouts = {"connect_ms": 500, "first_item_ms": 2000, "idle_ms": 1000, "total_ms": 5000}
    timeouts |= {key: value for key, value in overrides.items() if key in TIMEOUTS}
    others = {key: value for key, value in overrides.items() if key not in TIMEOUTS}
    config = MlxLmChatConfig.model_validate(
        {"type": "mlx_lm_chat", "base_url": base_url, "timeouts": timeouts, **others}
    )
    return MlxLmChatLLM(config, warmup_instructions="You are a helpful assistant.")


async def collect(stream: AsyncGenerator[Delta]) -> list[Delta]:
    async with aclosing(stream):
        return [delta async for delta in stream]


async def test_the_request_is_shaped_for_mlx_lm(base_url: str, fake: FakeMlxLm) -> None:
    llm = provider(base_url)
    await collect(llm.stream(ASK, [LOOKUP], max_output_tokens=64))
    await collect(llm.stream(ASK, [LOOKUP], tool_choice="none"))
    await llm.aclose()
    auto, none = fake.requests
    assert auto["messages"] == [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": "你好"},
    ]
    assert auto["stream"] is True
    assert auto["stream_options"] == {"include_usage": True}
    assert auto["chat_template_kwargs"] == {"enable_thinking": False}
    assert (auto["temperature"], auto["top_p"]) == (0.0, 1.0)  # tools attached: greedy (G2)
    assert (none["temperature"], none["top_p"]) == (0.7, 0.8)  # the server's own sampling
    assert auto["max_completion_tokens"] == 64  # the response's limit, under the configured 512
    assert none["max_completion_tokens"] == 512
    assert "model" not in auto  # omitted: the server uses the model it was started with
    assert auto["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look an account up.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert "tools" not in none  # tool_choice "none": no tools at all
    assert set(auto) <= {
        "messages", "stream", "stream_options", "max_completion_tokens", "temperature",
        "top_p", "chat_template_kwargs", "tools",
    }  # fmt: skip


async def test_an_exact_model_name_is_sent_when_configured(base_url: str, fake: FakeMlxLm) -> None:
    llm = provider(base_url, model="/models/qwen3.6")
    await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert fake.requests[0]["model"] == "/models/qwen3.6"


async def test_text_keepalives_usage_and_done(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(
        Keepalive(512, 2048),
        Keepalive(2048, 2048),
        Text("你"),
        Text("好。"),
        Finish("stop"),
        UsageChunk(prompt=20, completion=2, cached=16),
        Done(),
    )
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert deltas == [
        TextDelta("你"),
        TextDelta("好。"),
        LLMDone("stop", Usage(input_tokens=20, output_tokens=2, cached_tokens=16)),
    ]


async def test_reasoning_deltas_are_dropped(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(Reasoning("Let me think."), Text("好的。"), Finish("stop"), UsageChunk(), Done())
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert [delta for delta in deltas if isinstance(delta, TextDelta)] == [TextDelta("好的。")]


async def test_tool_calls_arrive_complete(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(
        Text("我查一下。"),
        Calls([tool_call("lookup", {"account": "42"}, "call-a")]),
        Calls([tool_call("lookup", {"account": "43"}, "call-b", index=1)]),
        Finish("tool_calls"),
        UsageChunk(),
        Done(),
    )
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert deltas[:3] == [
        TextDelta("我查一下。"),
        ToolCall("call-a", "lookup", '{"account": "42"}'),
        ToolCall("call-b", "lookup", '{"account": "43"}'),
    ]
    assert isinstance(deltas[3], LLMDone)
    assert deltas[3].finish_reason == "tool_calls"


async def test_calls_as_the_real_server_streams_them(base_url: str, fake: FakeMlxLm) -> None:
    """A newline before the call rides in the call's chunk; two calls the parser returned from
    one span arrive in one chunk; the call ids are the server's uuids."""
    fake.script(
        Text("好的，"),
        Calls([tool_call("lookup", {"account": "42"}, "6f1c")], text="\n\n"),
        Calls(
            [
                tool_call("lookup", {"account": "43"}, "7a2d", index=1),
                tool_call("lookup", {"account": "44"}, "8b3e", index=2),
            ]
        ),
        Finish("tool_calls"),
        UsageChunk(),
        Done(),
    )
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert deltas[:5] == [
        TextDelta("好的，"),
        ToolCall("6f1c", "lookup", '{"account": "42"}'),
        TextDelta("\n\n"),
        ToolCall("7a2d", "lookup", '{"account": "43"}'),
        ToolCall("8b3e", "lookup", '{"account": "44"}'),
    ]
    assert isinstance(deltas[5], LLMDone)
    assert deltas[5].finish_reason == "tool_calls"


async def test_a_silently_dropped_second_call(base_url: str, fake: FakeMlxLm) -> None:
    """The server logs and skips a call its parser rejects. With another call delivered, the
    stream looks complete: the call that parsed is delivered, the response completes."""
    fake.script(
        Calls([tool_call("lookup", {"account": "42"})]), Finish("tool_calls"), UsageChunk(), Done()
    )
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert deltas[0] == ToolCall("call-1", "lookup", '{"account": "42"}')
    assert isinstance(deltas[1], LLMDone)
    assert deltas[1].finish_reason == "tool_calls"


async def test_history_is_shaped_for_the_template(base_url: str, fake: FakeMlxLm) -> None:
    """The wait line spoken between a call and its result: merged into the call (default), or
    after the result (``tool_interlude: after_output``); the result always follows its call."""
    call = ToolCall("6f1c", "lookup", '{"account": "42"}')
    history = [
        *ASK,
        Message("assistant", "", tool_calls=(call,)),
        Message("assistant", "请稍等。"),
        Message("user", "(The caller pressed 2 on their keypad.)"),
        Message("tool", '{"balance": "12.50"}', tool_call_id="6f1c"),
    ]
    for interlude in ("merge", "after_output"):
        llm = provider(base_url, tool_interlude=interlude)
        await collect(llm.stream(history, [LOOKUP]))
        await llm.aclose()
    merged, after = (request["messages"][2:] for request in fake.requests)
    spec = {
        "id": "6f1c",
        "type": "function",
        "function": {"name": "lookup", "arguments": call.arguments},
    }
    result = {"role": "tool", "tool_call_id": "6f1c", "content": '{"balance": "12.50"}'}
    note = {"role": "user", "content": "(The caller pressed 2 on their keypad.)"}
    assert merged == [
        {"role": "assistant", "content": "请稍等。", "tool_calls": [spec]},
        result,
        note,
    ]
    assert after == [
        {"role": "assistant", "content": "", "tool_calls": [spec]},
        result,
        {"role": "assistant", "content": "请稍等。"},
        note,
    ]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [
                Calls(
                    [{"function": {"name": "lookup", "arguments": "{oops"}, "id": "c", "index": 0}]
                )
            ],
            id="arguments-not-json",
        ),
        pytest.param(
            [Calls([{"function": {"arguments": "{}"}, "type": "function", "id": "c", "index": 0}])],
            id="no-name",
        ),
        pytest.param([Calls([tool_call("lookup", {}, call_id="")])], id="no-id"),
        pytest.param([Finish("weird")], id="unknown-finish-reason"),
    ],
)
async def test_malformed_streams_are_bad_responses(
    base_url: str, fake: FakeMlxLm, steps: list[object]
) -> None:
    script = [*steps]
    if not any(isinstance(step, Finish) for step in script):
        script.append(Finish("tool_calls"))
    fake.script(*script, UsageChunk(), Done())  # pyright: ignore[reportArgumentType]
    llm = provider(base_url)
    with pytest.raises(ProviderError) as failed:
        await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert failed.value.kind == "bad_response"


async def test_a_stream_cut_before_done_is_a_bad_response(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(Text("你好"), Close())
    llm = provider(base_url)
    received: list[Delta] = []

    async def consume() -> None:
        async with aclosing(llm.stream(ASK, [])) as stream:
            async for delta in stream:
                received.append(delta)  # noqa: PERF401 (the deltas before the failure count)

    with pytest.raises(ProviderError) as failed:
        await consume()
    await llm.aclose()
    assert received == [TextDelta("你好")]
    assert failed.value.kind == "bad_response"


async def test_length_is_reported(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(*reply("一二三", finish="length"))
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, [], max_output_tokens=3))
    await llm.aclose()
    assert isinstance(deltas[-1], LLMDone)
    assert deltas[-1].finish_reason == "length"


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [(404, "bad_response", False), (500, "internal", False), (503, "unavailable", True)],
)
async def test_error_statuses(
    base_url: str, fake: FakeMlxLm, status: int, kind: str, retryable: bool
) -> None:
    fake.script(Fail(status))
    llm = provider(base_url)
    with pytest.raises(ProviderError) as failed:
        await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert (failed.value.kind, failed.value.retryable) == (kind, retryable)
    assert str(status) in str(failed.value)


async def test_a_backend_that_is_down_is_unavailable() -> None:
    llm = provider(f"http://127.0.0.1:{free_port()}")
    with pytest.raises(ProviderError) as failed:
        await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert (failed.value.kind, failed.value.retryable) == ("unavailable", True)


async def test_the_connect_deadline() -> None:
    def refuse_slowly(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("timed out", request=request)

    config = MlxLmChatConfig.model_validate({"type": "mlx_lm_chat"})
    llm = MlxLmChatLLM(config, transport=httpx2.MockTransport(refuse_slowly))
    with pytest.raises(ProviderTimeout) as failed:
        await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert failed.value.stage == "connect"


@pytest.mark.parametrize(
    ("steps", "timeouts", "stage"),
    [
        pytest.param([Hold(Gate())], {"first_item_ms": 150}, "first_item", id="before-headers"),
        pytest.param(
            [Keepalive(), Keepalive(), Hold(Gate())],
            {"first_item_ms": 150},
            "first_item",
            id="keepalives-are-not-tokens",
        ),
        pytest.param(
            [Reasoning("hm"), Hold(Gate())],
            {"first_item_ms": 150},
            "first_item",
            id="reasoning-is-not-a-token",
        ),
        pytest.param([Text("你"), Hold(Gate())], {"idle_ms": 150}, "idle", id="between-tokens"),
        pytest.param(
            [Text("一"), Sleep(0.1), Text("二"), Sleep(0.1), Text("三"), Sleep(0.1), Text("四")],
            {"idle_ms": 1000, "total_ms": 220},
            "total",
            id="total",
        ),
    ],
)
async def test_deadlines(
    base_url: str, fake: FakeMlxLm, steps: list[object], timeouts: dict[str, int], stage: str
) -> None:
    fake.script(*steps)  # pyright: ignore[reportArgumentType]
    llm = provider(base_url, **timeouts)
    with pytest.raises(ProviderTimeout) as failed:
        await collect(llm.stream(ASK, []))
    assert failed.value.stage == stage
    assert failed.value.kind == "timeout"
    await fake.disconnected()  # the request was abandoned, not left running
    await llm.aclose()


async def cancel_when(task: asyncio.Task[object], reached: Gate) -> None:
    await reached.reached()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "point",
    ["awaiting-headers", "before-first-token", "between-tokens"],
)
async def test_cancellation_while_awaiting_the_backend(
    base_url: str, fake: FakeMlxLm, point: str
) -> None:
    gate = Gate()
    match point:
        case "awaiting-headers":
            fake.script(Hold(gate), *reply("x"))
        case "before-first-token":
            fake.script(Keepalive(), Hold(gate), *reply("x"))
        case _:
            fake.script(Text("你"), Hold(gate), *reply("x"))
    llm = provider(base_url)
    received: list[Delta] = []

    async def consume() -> None:
        async with aclosing(llm.stream(ASK, [])) as stream:
            async for delta in stream:
                received.append(delta)  # noqa: PERF401 (the deltas before the cancel count)

    task = asyncio.create_task(consume())
    if point == "between-tokens":
        async with asyncio.timeout(2):  # the first token reached the consumer, then the cancel
            while not received:  # noqa: ASYNC110 (a list the consumer task appends to)
                await asyncio.sleep(0.002)
    await cancel_when(task, gate)
    await fake.disconnected()
    gate.release()
    assert received == ([TextDelta("你")] if point == "between-tokens" else [])
    await llm.aclose()


async def test_cancellation_while_the_consumer_holds_a_delta(
    base_url: str, fake: FakeMlxLm
) -> None:
    gate = Gate()
    fake.script(Text("你"), Hold(gate), *reply("x"))
    llm = provider(base_url)
    holding = asyncio.Event()

    async def consume() -> None:
        async with aclosing(llm.stream(ASK, [])) as stream:
            assert await anext(stream) == TextDelta("你")
            holding.set()
            await asyncio.Event().wait()  # busy elsewhere, the stream suspended at its yield

    task = asyncio.create_task(consume())
    await holding.wait()
    await gate.reached()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await fake.disconnected()
    gate.release()
    await llm.aclose()


async def test_cancellation_while_waiting_for_a_slot(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.script(Text("你"), Hold(gate), *reply("x"))
    llm = provider(base_url, max_in_flight=1, queue_timeout_ms=5000)
    first = llm.stream(ASK, [])
    assert await anext(first) == TextDelta("你")
    waiting = asyncio.create_task(collect(llm.stream(ASK, [])))
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert len(fake.requests) == 1  # the queued request never reached the backend
    gate.release()
    async with aclosing(first):
        rest = [delta async for delta in first]
    assert isinstance(rest[-1], LLMDone)
    await collect(llm.stream(ASK, []))  # the slot was given back
    await llm.aclose()


async def test_too_many_requests_in_flight_is_overloaded(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.script(Text("你"), Hold(gate), *reply("x"))
    llm = provider(base_url, max_in_flight=1, queue_timeout_ms=50)
    first = llm.stream(ASK, [])
    await anext(first)
    with pytest.raises(ProviderError) as failed:
        await collect(llm.stream(ASK, []))
    assert (failed.value.kind, failed.value.retryable) == ("overloaded", True)
    gate.release()
    await collect(first)
    await llm.aclose()


async def test_readiness_follows_health(base_url: str, fake: FakeMlxLm) -> None:
    llm = provider(base_url)
    assert (await llm.ready()).ready
    fake.health = 503  # the generation thread died
    state = await llm.ready()
    assert not state.ready
    assert "503" in state.detail
    await llm.aclose()
    down = provider(f"http://127.0.0.1:{free_port()}")
    assert not (await down.ready()).ready
    await down.aclose()


async def test_warmup_is_a_one_token_completion_with_the_default_instructions(
    base_url: str, fake: FakeMlxLm
) -> None:
    llm = provider(base_url)
    await llm.warmup()
    await llm.aclose()
    assert fake.requests == []
    (request,) = fake.warmups
    assert request["max_completion_tokens"] == 1
    assert request["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "(The call has connected. Greet the caller as instructed.)"},
    ]
    assert "tools" not in request


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# LLM hardening (M0 S1/S2, G2): the dropped-call retry, pseudo-calls, prefix warming.

TRANSFER = tool_call("lookup", {"account": "42"}, "call-r")


async def test_a_dropped_call_is_retried_once_in_the_same_stream(
    base_url: str, fake: FakeMlxLm
) -> None:
    """The server dropped the call (tool_calls, no call): asked once more, transparently, at
    the sampling of requests without tools (greedy would repeat the same malformed call)."""
    fake.script(Text("\n\n"), Finish("tool_calls"), UsageChunk(), Done())
    fake.script(Calls([TRANSFER]), Finish("tool_calls"), UsageChunk(), Done())
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert deltas[0] == TextDelta("\n\n")  # whitespace: nothing the consumer speaks
    assert deltas[1] == ToolCall("call-r", "lookup", '{"account": "42"}')
    assert isinstance(deltas[2], LLMDone)
    assert deltas[2].finish_reason == "tool_calls"
    first, retry = fake.requests
    assert (first["temperature"], first["top_p"]) == (0.0, 1.0)
    assert (retry["temperature"], retry["top_p"]) == (0.7, 0.8)
    assert {**retry, "temperature": 0.0, "top_p": 1.0} == first


async def test_a_call_dropped_twice_fails_the_response(base_url: str, fake: FakeMlxLm) -> None:
    for _ in range(2):
        fake.script(Finish("tool_calls"), UsageChunk(), Done())
    llm = provider(base_url)
    with pytest.raises(ProviderError, match="dropped a malformed call") as failed:
        await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert failed.value.kind == "bad_response"
    assert len(fake.requests) == 2


async def test_a_call_dropped_after_text_is_not_retried(
    base_url: str, fake: FakeMlxLm, caplog: pytest.LogCaptureFixture
) -> None:
    fake.script(Text("好的，我帮您转接。"), Finish("tool_calls"), UsageChunk(), Done())
    llm = provider(base_url)
    received: list[Delta] = []

    async def consume() -> None:
        async with aclosing(llm.stream(ASK, [LOOKUP])) as stream:
            async for delta in stream:
                received.append(delta)  # noqa: PERF401 (the deltas before the failure count)

    with pytest.raises(ProviderError, match="dropped a malformed call"):
        await consume()
    await llm.aclose()
    assert received == [TextDelta("好的，我帮您转接。")]
    assert len(fake.requests) == 1
    assert "not retried" in caplog.text


async def test_the_retry_can_be_turned_off(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(Finish("tool_calls"), UsageChunk(), Done())
    llm = provider(base_url, retry_dropped_call=False)
    with pytest.raises(ProviderError, match="dropped a malformed call"):
        await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert len(fake.requests) == 1


async def test_the_retry_runs_under_the_first_requests_total_deadline(
    base_url: str, fake: FakeMlxLm
) -> None:
    fake.script(Sleep(0.15), Finish("tool_calls"), UsageChunk(), Done())
    fake.script(Text("一"), Sleep(0.4), *reply("二"))
    llm = provider(base_url, total_ms=300)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(ProviderTimeout) as failed:
        await collect(llm.stream(ASK, [LOOKUP]))
    assert failed.value.stage == "total"
    assert loop.time() - started < 0.45  # not 0.15 + a fresh 0.3
    await fake.disconnected()
    await llm.aclose()


async def test_cancelling_the_retry(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.script(Finish("tool_calls"), UsageChunk(), Done())
    fake.script(Hold(gate), Calls([TRANSFER]), Finish("tool_calls"), UsageChunk(), Done())
    llm = provider(base_url)
    await cancel_when(asyncio.create_task(collect(llm.stream(ASK, [LOOKUP]))), gate)
    await fake.disconnected()
    gate.release()
    await llm.aclose()


async def test_a_reply_made_without_tools_loses_its_pseudo_call(
    base_url: str, fake: FakeMlxLm
) -> None:
    """M0 S1: with no tools sent the model wrote a JSON "call" into its reply."""
    fake.script(
        Text("好的，我查一下。\n"),
        Text('{"tool_name": "repair_status", '),
        Text('"tool_input": {"rma_number": "RMA2048"}}'),
        *reply(),
    )
    fake.script(Text('{"tool_name": "x", "tool_input": {}}'), *reply())
    llm = provider(base_url)
    without_tools = await collect(llm.stream(ASK, [LOOKUP], tool_choice="none"))
    with_tools = await collect(llm.stream(ASK, [LOOKUP]))
    await llm.aclose()
    assert without_tools[:-1] == [TextDelta("好的，我查一下。\n")]
    assert isinstance(without_tools[-1], LLMDone)
    assert with_tools[0] == TextDelta('{"tool_name": "x", "tool_input": {}}')  # left alone


async def test_text_held_at_the_end_of_a_reply_is_released(base_url: str, fake: FakeMlxLm) -> None:
    fake.script(Text("请按 {"), Text('"键": 1'), Finish("length"), UsageChunk(), Done())
    llm = provider(base_url)
    deltas = await collect(llm.stream(ASK, []))
    await llm.aclose()
    assert deltas[:-1] == [TextDelta("请按 "), TextDelta('{"键": 1')]
    assert isinstance(deltas[-1], LLMDone)


HISTORY = [
    Message("system", "转接环节。"),
    Message("assistant", "您好。"),
    Message("user", "转人工"),
    Message("assistant", "", tool_calls=(ToolCall("p1", "lookup", "{}"),)),
]


async def test_warm_prefix_sends_the_next_requests_prefix(base_url: str, fake: FakeMlxLm) -> None:
    llm = provider(base_url)
    await llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto")
    await llm.warm_prefix([Message("system", "开场。")], [LOOKUP], tool_choice="auto")
    await llm.aclose()
    later, opening = fake.warmups
    assert later["max_completion_tokens"] == 1
    assert later["messages"] == [
        {"role": "system", "content": "转接环节。"},
        {"role": "assistant", "content": "您好。"},
        {"role": "user", "content": "转人工"},
    ]
    assert later["tools"][0]["function"]["name"] == "lookup"
    assert (later["temperature"], later["top_p"]) == (0.0, 1.0)  # as the real request
    assert opening["messages"] == [
        {"role": "system", "content": "开场。"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "(The call has connected. Greet the caller as instructed.)"},
    ]


async def test_a_request_waits_for_the_warmup_of_its_prefix(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    warming = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    answer = asyncio.create_task(
        collect(llm.stream([*HISTORY, Message("tool", "{}", tool_call_id="p1")], [LOOKUP]))
    )
    other = await collect(llm.stream([Message("system", "另一个。"), *ASK[1:]], [LOOKUP]))
    assert isinstance(other[-1], LLMDone)  # another prefix does not wait
    await asyncio.sleep(0.1)
    assert len(fake.requests) == 1  # the other prefix's; the answer waits for the warm-up
    gate.release()
    await warming
    deltas = await answer
    await llm.aclose()
    assert isinstance(deltas[-1], LLMDone)
    assert fake.requests[1]["messages"][-1]["role"] == "tool"


async def test_waiting_for_a_warmup_can_be_cancelled(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    warming = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    answer = asyncio.create_task(collect(llm.stream(HISTORY, [LOOKUP])))
    await asyncio.sleep(0.05)
    answer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await answer
    gate.release()
    await warming  # the warm-up itself carries on
    await llm.aclose()
    assert (len(fake.warmups), len(fake.requests)) == (1, 0)


async def test_a_warmup_that_fails_is_only_logged(
    base_url: str, fake: FakeMlxLm, caplog: pytest.LogCaptureFixture
) -> None:
    fake.warmup(Fail(500))
    llm = provider(base_url)
    await llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto")
    deltas = await collect(llm.stream(HISTORY, [LOOKUP]))
    await llm.aclose()
    assert isinstance(deltas[-1], LLMDone)
    assert "prefix warm-up failed" in caplog.text


async def test_warming_can_be_turned_off(base_url: str, fake: FakeMlxLm) -> None:
    llm = provider(base_url, warm_on_update=False)
    await llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto")
    await llm.aclose()
    assert fake.warmups == []


OTHER_CALL = [Message("system", "转接环节。"), Message("user", "我要留言")]  # HISTORY's prefix


async def test_a_second_warmup_of_a_prefix_in_flight_is_skipped(
    base_url: str, fake: FakeMlxLm
) -> None:
    """A burst of new calls with the same instructions warms them once: the server cannot drop
    a request it is prefilling, so abandoned duplicates would only delay the real ones."""
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    first = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    await llm.warm_prefix(OTHER_CALL, [LOOKUP], tool_choice="auto")  # returns at once
    gate.release()
    await first
    await llm.aclose()
    assert len(fake.warmups) == 1


async def test_warmups_go_one_at_a_time(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    first = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    other = [Message("system", "另一个流程。"), Message("user", "你好")]
    second = asyncio.create_task(llm.warm_prefix(other, [LOOKUP], tool_choice="auto"))
    await asyncio.sleep(0.1)
    assert len(fake.warmups) == 1  # the second queues behind the first
    gate.release()
    await asyncio.gather(first, second)
    await llm.aclose()
    assert [w["messages"][0]["content"] for w in fake.warmups] == ["转接环节。", "另一个流程。"]


async def test_a_queued_warmup_is_dropped_once_its_request_went_out(
    base_url: str, fake: FakeMlxLm
) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    first = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    other = [Message("system", "另一个流程。"), Message("user", "你好")]
    queued = asyncio.create_task(llm.warm_prefix(other, [LOOKUP], tool_choice="auto"))
    await asyncio.sleep(0.05)
    await collect(llm.stream(other, [LOOKUP]))  # the real request goes out: no warm-up to wait for
    gate.release()
    await asyncio.gather(first, queued)
    await llm.aclose()
    assert len(fake.warmups) == 1
    assert len(fake.requests) == 1


async def test_a_queued_warmup_cancelled_never_reaches_the_server(
    base_url: str, fake: FakeMlxLm
) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    first = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    other = [Message("system", "另一个流程。"), Message("user", "你好")]
    queued = asyncio.create_task(llm.warm_prefix(other, [LOOKUP], tool_choice="auto"))
    await asyncio.sleep(0.05)
    queued.cancel()  # its session ended, or its instructions changed again
    with pytest.raises(asyncio.CancelledError):
        await queued
    gate.release()
    await first
    await llm.aclose()
    assert len(fake.warmups) == 1


async def test_a_sent_warmup_outlives_its_caller(base_url: str, fake: FakeMlxLm) -> None:
    """The server finishes a prefill it has started whatever the client does, so a warm-up whose
    session went away still counts: the next session with the same instructions waits for it
    rather than sending the same prefill again."""
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    caller = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    caller.cancel()  # its session closed
    with pytest.raises(asyncio.CancelledError):
        await caller
    await llm.warm_prefix(OTHER_CALL, [LOOKUP], tool_choice="auto")  # skipped: still in flight
    answer = asyncio.create_task(collect(llm.stream(OTHER_CALL, [LOOKUP])))
    await asyncio.sleep(0.1)
    assert (len(fake.warmups), len(fake.requests), fake.disconnects) == (1, 0, 0)
    gate.release()
    deltas = await answer
    await llm.aclose()
    assert isinstance(deltas[-1], LLMDone)
    assert len(fake.requests) == 1


async def test_closing_the_provider_stops_a_warmup(base_url: str, fake: FakeMlxLm) -> None:
    gate = Gate()
    fake.warmup(Hold(gate), *reply(finish="length"))
    llm = provider(base_url)
    caller = asyncio.create_task(llm.warm_prefix(HISTORY, [LOOKUP], tool_choice="auto"))
    await gate.reached()
    await llm.aclose()
    await fake.disconnected()
    with pytest.raises(asyncio.CancelledError):
        await caller
    gate.release()

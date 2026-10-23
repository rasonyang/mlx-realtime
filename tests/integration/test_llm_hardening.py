"""LLM hardening on the wire: the real LLM provider against a fake ``mlx_lm.server`` (which
streams as the real one does), behind a gateway with mock ASR and TTS.

- a dropped call (``finish_reason: tool_calls``, no call) is asked again within the response:
  the client sees one completed response with its call, and no error;
- a JSON "call" the model writes into a reply made without tools is neither spoken nor in the
  transcript;
- a ``session.update`` with new instructions sends the warm-up request first, and a response
  created while it is in flight waits for it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mlx_realtime.config import MlxLmChatConfig, validate_config
from mlx_realtime.providers.mlx_lm_chat import MlxLmChatLLM
from mlx_realtime.providers.mock import ScriptedASR, SineTTS
from mlx_realtime.providers.registry import Providers
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedSpeech
from tests.fakes.mlx_lm import Calls, Done, FakeMlxLm, Finish, Gate, Hold, Text, reply, tool_call
from tests.fakes.mlx_lm import Usage as UsageChunk
from tests.support import Client, Event, running, with_echo_off

LOOKUP = {
    "type": "function",
    "name": "lookup_balance",
    "parameters": {"type": "object", "properties": {"accountId": {"type": "string"}}},
}
LINE = "Say exactly this, word for word, and add nothing else:\n"
SESSION = {
    "audio": {"input": {"turn_detection": None}},
    "instructions": "You answer the phone for NovaNet.",
    "tools": [LOOKUP],
}


@asynccontextmanager
async def stack() -> AsyncIterator[tuple[Client, FakeMlxLm]]:
    fake = FakeMlxLm()
    with fake.serving() as url:
        llm = MlxLmChatLLM(MlxLmChatConfig.model_validate({"type": "mlx_lm_chat", "base_url": url}))
        providers = Providers(llm=llm, asr=ScriptedASR(["你好。"]), tts=SineTTS(ms_per_char=20))
        gateway = Gateway(
            validate_config(with_echo_off(None)),
            providers,
            speech_detector=lambda: ScriptedSpeech(()),
        )
        async with running(gateway) as address, Client.connect(address) as client:
            await client.start(SESSION)
            yield client, fake
        await llm.aclose()


def transcript(events: list[Event]) -> str:
    return "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    )


async def test_a_dropped_call_is_asked_again_within_the_response() -> None:
    async with stack() as (client, fake):
        await client.say("我的余额是多少？")
        fake.script(Text("\n"), Finish("tool_calls"), UsageChunk(), Done())  # dropped by the parser
        call = tool_call("lookup_balance", {"accountId": "42"})
        fake.script(Calls([call]), *reply(finish="tool_calls"))
        await client.send({"type": "response.create"})
        events = await client.until("response.done")
    assert [event["type"] for event in events if event["type"] == "error"] == []
    done = events[-1]["response"]
    assert done["status"] == "completed"
    assert [item["type"] for item in done["output"]] == ["function_call"]
    assert done["output"][0]["arguments"] == '{"accountId": "42"}'
    first, retry = fake.requests
    assert (first["temperature"], retry["temperature"]) == (0.0, 0.7)


async def test_a_pseudo_call_is_not_spoken() -> None:
    async with stack() as (client, fake):
        await client.say("帮我查一下维修单RMA2048。")
        fake.script(
            Text("好的，我查一下。\n"),
            Text('{"tool_name": "repair_status", '),
            Text('"tool_input": {"rma_number": "RMA2048"}}'),
            *reply(),
        )
        await client.send({"type": "response.create", "response": {"tool_choice": "none"}})
        events = await client.until("response.done")
    assert "tools" not in fake.requests[0]
    assert transcript(events) == "好的，我查一下。"
    assert events[-1]["response"]["status"] == "completed"


async def warmed(fake: FakeMlxLm, count: int) -> None:
    """Until ``count`` warm-ups (one-token requests) have been answered."""
    async with asyncio.timeout(5):
        while len(fake.warmups) < count or fake.finished < count:  # noqa: ASYNC110 (counters)
            await asyncio.sleep(0.005)


async def test_a_response_waits_for_the_warmup_of_new_instructions() -> None:
    async with stack() as (client, fake):
        await warmed(fake, 2)  # the gateway's startup warmup, the session's first update's
        await client.say("你好")
        gate = Gate()
        fake.warmup(Hold(gate), *reply(finish="length"))
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", "instructions": "Phase 2."}}
        )
        await client.until("session.updated")
        await gate.reached()
        await client.send({"type": "response.create"})
        await client.until("response.created")
        await asyncio.sleep(0.2)
        assert fake.requests == []  # the response waits for the warm-up in flight
        gate.release()
        events = await client.until("response.done")
    assert events[-1]["response"]["status"] == "completed"
    warmup = fake.warmups[-1]
    assert warmup["messages"] == [
        {"role": "system", "content": "Phase 2."},
        {"role": "user", "content": "你好"},
    ]
    assert warmup["tools"] == fake.requests[0]["tools"]
    assert fake.requests[0]["messages"][0] == {"role": "system", "content": "Phase 2."}


async def test_a_burst_of_calls_with_one_flow_warms_it_once() -> None:
    """Twenty calls start with the same instructions and hang up after their greeting line (the
    first warm-up still running): the LLM gets one prefill, not twenty it cannot drop."""
    fake = FakeMlxLm()
    with fake.serving() as url:
        llm = MlxLmChatLLM(MlxLmChatConfig.model_validate({"type": "mlx_lm_chat", "base_url": url}))
        providers = Providers(llm=llm, asr=ScriptedASR(["你好。"]), tts=SineTTS(ms_per_char=20))
        gateway = Gateway(
            validate_config(with_echo_off(None)),
            providers,
            speech_detector=lambda: ScriptedSpeech(()),
        )
        async with running(gateway) as address:
            await warmed(fake, 1)  # the gateway's own startup warmup
            gate = Gate()
            fake.warmup(Hold(gate), *reply(finish="length"))
            for _ in range(20):
                async with Client.connect(address) as client:
                    await client.start(SESSION)
                    line = {"instructions": f"{LINE}您好。"}
                    await client.send({"type": "response.create", "response": line})
                    await client.until("response.done")
            gate.release()
        await llm.aclose()
    assert len(fake.warmups) == 2
    assert fake.requests == []

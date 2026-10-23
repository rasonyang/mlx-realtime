"""Prefix warming (M0 S2): a ``session.update`` that changes the instructions or the tools makes
the session warm its LLM's prompt cache for them, once this session's LLM is idle.

What a warm-up sends, and how a response waits for one, is the provider's
(``tests/unit/test_mlx_lm_chat.py``); here the session decides when to warm and with what.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.providers.base import Message, ToolChoice, ToolDef
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR, SineTTS
from mlx_realtime.providers.registry import Providers
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.scripted import ScriptedSpeech
from tests.support import Client, running, with_echo_off

MANUAL = {"audio": {"input": {"turn_detection": None}}}
LOOKUP = {
    "type": "function",
    "name": "lookup_balance",
    "parameters": {"type": "object", "properties": {"accountId": {"type": "string"}}},
}


class WarmingLLM(EchoLLM):
    """The echo LLM, plus a ``warm_prefix`` that records each call and waits for ``gate``."""

    def __init__(self, *, token_delay_s: float = 0.0) -> None:
        super().__init__(token_delay_s=token_delay_s)
        self.warmups: list[tuple[tuple[Message, ...], tuple[ToolDef, ...], ToolChoice]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.cancelled = 0
        self.fail = False

    async def warm_prefix(
        self, messages: Sequence[Message], tools: Sequence[ToolDef], *, tool_choice: ToolChoice
    ) -> None:
        self.warmups.append((tuple(messages), tuple(tools), tool_choice))
        if self.fail:
            raise RuntimeError("the warm-up broke")
        try:
            await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def gateway_with(llm: WarmingLLM) -> Gateway:
    return Gateway(
        validate_config(with_echo_off(None)),
        Providers(llm=llm, asr=ScriptedASR(["你好。"]), tts=SineTTS(ms_per_char=20, chunk_ms=40)),
        speech_detector=lambda: ScriptedSpeech(()),
        turn_end_detector=lambda: None,
    )


async def update(client: Client, **session: object) -> None:
    await client.send({"type": "session.update", "session": {"type": "realtime", **session}})
    await client.until("session.updated")


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_new_instructions_or_tools_warm_the_prefix() -> None:
    llm = WarmingLLM()
    async with running(gateway_with(llm)) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "instructions": "Phase 1.", "tools": [LOOKUP]})
        await client.say("你好")
        await update(client, instructions="Phase 2.")
        await settle()
        (first, second) = llm.warmups
    messages, tools, tool_choice = second
    assert first[0] == (Message("system", "Phase 1."),)  # the first update, before any turn
    assert messages == (Message("system", "Phase 2."), Message("user", "你好"))
    assert [tool.name for tool in tools] == ["lookup_balance"]
    assert tool_choice == "auto"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"instructions": "Phase 1."}, id="same-instructions"),
        pytest.param({"audio": {"output": {"voice": "serena"}}}, id="voice-only"),
    ],
)
async def test_an_update_that_leaves_the_prefix_alone_does_not_warm(
    change: dict[str, object],
) -> None:
    llm = WarmingLLM()
    async with running(gateway_with(llm)) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "instructions": "Phase 1."})
        await update(client, **change)
        await settle()
    assert len(llm.warmups) == 1


async def test_a_newer_update_cancels_the_older_warmup() -> None:
    llm = WarmingLLM()
    llm.gate.clear()
    async with running(gateway_with(llm)) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "instructions": "Phase 1."})
        await update(client, instructions="Phase 2.")
        await settle()
        llm.gate.set()
        await settle()
    assert [warmup[0][0].content for warmup in llm.warmups] == ["Phase 1.", "Phase 2."]
    assert llm.cancelled == 1


async def test_while_the_llm_generates_the_warmup_waits_for_it() -> None:
    """AICC re-pins the instructions when a call arrives mid-response: the warm-up starts once
    this session's LLM has finished, from the conversation as it then is."""
    llm = WarmingLLM(token_delay_s=0.05)
    async with running(gateway_with(llm)) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "instructions": "Phase 1."})
        await client.say("一二三四五六")
        await client.send({"type": "response.create"})
        await client.until("response.output_item.added")
        await update(client, instructions="Phase 2.")
        await settle()
        assert len(llm.warmups) == 1  # only the first update's: the LLM is still answering
        await client.until("response.done")
        await settle()
    messages = llm.warmups[1][0]
    assert messages[0] == Message("system", "Phase 2.")
    assert Message("user", "一二三四五六") in messages


async def test_a_failing_warmup_leaves_the_session_working() -> None:
    llm = WarmingLLM()
    llm.fail = True
    async with running(gateway_with(llm)) as address, Client.connect(address) as client:
        await client.start({**MANUAL, "instructions": "Phase 1."})
        await client.say("你好")
        await client.send({"type": "response.create"})
        done = (await client.until("response.done"))[-1]
    assert done["response"]["status"] == "completed"

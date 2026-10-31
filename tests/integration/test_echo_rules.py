"""The barge-in rules and echo-turn suppression (D17's second line of defence) through the
session, on a virtual clock: the client streams 20 ms of audio per 20 ms of ``ManualClock``
time, so the gateway models its playout exactly as for a real-time client, and the scripted
detector puts speech at given input positions.

- An onset inside the first ``echo_guard_ms`` of a response's playback never interrupts; past
  the guard, 400 ms of speech does (with ``speech_started`` before ``response.done``).
- A turn whose transcript matches what the client was playing is noise: its response completes
  with no output and no LLM call, and the LLM never sees it; another turn is answered.
"""

from __future__ import annotations

import asyncio
import base64

from tests.support import Client, ManualClock, build_gateway, running

STEP_MS = 20
SILENCE = base64.b64encode(bytes(2 * 24 * STEP_MS)).decode()
LONG = "请讲一个很长很长的故事，越长越好，一直讲下去不要停。讲完一个再讲下一个。" * 2  # 7.2 s


async def stream(client: Client, clock: ManualClock, ms: int) -> None:
    """``ms`` of input in 20 ms appends, each when the virtual clock reaches it."""
    for _ in range(ms // STEP_MS):
        clock.advance(STEP_MS / 1000)
        await client.send({"type": "input_audio_buffer.append", "audio": SILENCE})
        await asyncio.sleep(0.002)  # let the gateway take it before time moves on


def types(client: Client) -> list[str]:
    return [event["type"] for event in client.events]


async def drain(client: Client) -> None:
    while True:
        try:
            await client.recv(wait_s=0.05)
        except TimeoutError:
            return


async def test_an_onset_in_the_echo_guard_never_interrupts_and_a_later_one_does() -> None:
    clock = ManualClock()
    config = {
        "output": {"max_lead_ms": 1000},  # the answer is still being sent while it plays
        "barge_in": {"echo_guard_ms": 2000, "min_speech_ms_while_playing": 400},
    }
    # Speech at input 1000-1800 ms (inside the guard), then 3500-4300 ms (after it).
    gateway, _ = build_gateway(
        config, speech=[(1000, 1800), (3500, 4300)], ms_per_char=100, chunk_ms=100, clock=clock
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await stream(client, clock, 200)
        await client.say(LONG)
        await client.send({"type": "response.create"})  # its playback starts at about 200 ms
        await client.until("response.output_audio.delta")
        await stream(client, clock, 3000)  # to 3.2 s: the guarded onset came and went
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        assert "response.done" not in types(client)
        await stream(client, clock, 1400)  # the second onset, 400 ms of it, and some
        await drain(client)
    kinds = types(client)
    started = kinds.index("input_audio_buffer.speech_started")
    assert kinds.index("response.done") > started
    speech = client.events[started]
    assert speech["audio_start_ms"] == 3488 - 300  # the onset frame, less the prefix padding
    done = next(event for event in client.events if event["type"] == "response.done")
    assert done["response"]["status_details"] == {"type": "cancelled", "reason": "turn_detected"}


async def test_without_the_rules_the_guarded_onset_interrupts() -> None:
    clock = ManualClock()
    config = {"output": {"max_lead_ms": 1000}}  # build_gateway turns the rules off
    gateway, _ = build_gateway(
        config, speech=[(1000, 1800)], ms_per_char=100, chunk_ms=100, clock=clock
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await stream(client, clock, 200)
        await client.say(LONG)
        await client.send({"type": "response.create"})
        await client.until("response.output_audio.delta")
        await stream(client, clock, 1400)
        await drain(client)
    kinds = types(client)
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index("response.done")


async def test_a_turn_that_is_our_own_speech_is_noise_and_another_is_answered() -> None:
    clock = ManualClock()
    config = {"barge_in": {"echo_turn_suppression": True}}  # the other rules stay off
    answer = "但鸟类还是有很多地方跟恐龙相似。"  # 1.6 s at 100 ms/char
    gateway, mocks = build_gateway(
        config,
        speech=[(700, 1500), (6000, 6800)],
        transcripts=["鸟类还是有很多地方跟恐龙", "我想问一下价格。"],
        ms_per_char=100,
        chunk_ms=100,
        clock=clock,
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await stream(client, clock, 200)
        await client.say(answer)
        await client.send({"type": "response.create"})  # our answer plays from about 200 ms
        await client.until("response.done")
        llm_calls = len(mocks.llm.requests)
        await stream(client, clock, 2000)  # the client heard its own speech back: one turn
        echo = (await client.until("response.done"))[-1]["response"]
        assert (echo["status"], echo["output"]) == ("completed", [])  # noise (D7)
        assert len(mocks.llm.requests) == llm_calls  # no LLM call
        await stream(client, clock, 5000)  # a real turn, long after our answer
        reply = (await client.until("response.done"))[-1]["response"]
        assert reply["output"][0]["content"][0]["transcript"] == "我想问一下价格。"
    history = [(message.role, message.content) for message in mocks.llm.requests[-1].messages]
    assert ("user", "鸟类还是有很多地方跟恐龙") not in history  # the echo turn never reaches it
    assert history[-1] == ("user", "我想问一下价格。")


ANSWER = "但鸟类还是有很多地方跟恐龙相似。"  # 1.6 s at 100 ms/char


async def merged_turn(heard: str) -> tuple[Client, list[str]]:
    """One turn over our own answer (the client played it back), transcribed as ``heard``."""
    clock = ManualClock()
    config = {"barge_in": {"echo_turn_suppression": True}}
    gateway, mocks = build_gateway(
        config,
        speech=[(700, 1500)],
        transcripts=[heard],
        ms_per_char=100,
        chunk_ms=100,
        clock=clock,
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start()
        await stream(client, clock, 200)
        await client.say(ANSWER)
        await client.send({"type": "response.create"})
        await client.until("response.done")
        await stream(client, clock, 2000)
        await client.until("response.done")
    return client, [message.content for message in mocks.llm.requests[-1].messages]


async def test_a_caller_who_speaks_into_the_echo_is_answered_with_what_they_said() -> None:
    """Issue #8: the echo prefix and the caller's words are one VAD turn (14 of 19 units are
    ours, contiguity 0.74). The remainder is the turn's transcript for the LLM, the transcription
    event and the item."""
    client, history = await merged_turn("鸟类还是有很多地方跟恐龙相似我想问价格")
    assert history[-1] == "我想问价格"
    kinds = types(client)
    at = kinds.index("conversation.item.input_audio_transcription.completed")
    assert client.events[at]["transcript"] == "我想问价格"
    assert kinds[at + 1] == "conversation.item.done"
    assert client.events[at + 1]["item"]["content"][0]["transcript"] == "我想问价格"


async def test_a_remainder_below_the_minimum_units_leaves_the_turn_an_echo_turn() -> None:
    client, history = await merged_turn("鸟类还是有很多地方跟恐龙相似我想问")  # 3 units after
    assert "我想问" not in history[-1]
    reply = [e for e in client.events if e["type"] == "response.done"][-1]["response"]
    assert reply["output"] == []

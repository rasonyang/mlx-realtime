"""Holding ``speech_started`` while the client plays our audio (``barge_in.confirm_while_playing``,
issue #5) through the session, on a virtual clock as in ``test_echo_rules.py``: the client streams
20 ms of audio per 20 ms of ``ManualClock`` time, the scripted detector puts speech at given input
positions, and an ASR whose answer is set by the test (the audio is silence: it can only tell the
requests apart by their length and order) plays the part of the ASR hearing the turn.

A turn that starts while the client is modelled as playing is held. Partial ASR checks decide:
the turn is released once two of its units are not our own words; one that ends as our echo, or
as nothing, is dropped with no event at all; a fault of the ASR releases it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from prometheus_client.parser import text_string_to_metric_families

from mlx_realtime.audio.pcm import CANONICAL_RATE, bytes_to_ms
from mlx_realtime.config import validate_config
from mlx_realtime.providers.base import ProviderError, Transcript, TTSBackend
from mlx_realtime.providers.mock import EchoLLM, SineTTS, _MockBackend
from mlx_realtime.providers.registry import Providers
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.vad.base import TurnEndDetector
from mlx_realtime.vad.scripted import ScriptedSpeech, ScriptedTurnEnd
from tests.integration.test_echo_rules import LONG, drain, stream, types
from tests.support import Client, Event, ManualClock, running, with_echo_off

ECHO = "很长很长的故事越长越好"  # the answer's own words
SHORT_ECHO = "很长很"  # three of them: below echo_match_min_units
CALLER = "我想问一下价格吧"
ONSET_START = 3488 - 300  # the onset frame of a span at 3500 ms, less the prefix padding
RULES = {
    "echo_guard_ms": 2000,
    "min_speech_ms_while_playing": 400,
    "echo_turn_suppression": True,
    "confirm_while_playing": True,
    "confirm_timeout_ms": 10_000,
}


class HeardASR(_MockBackend):
    """An ASR that answers ``text`` (or fails with ``error``) and records the length of each
    request. ``arm()`` makes the next request wait for ``open()`` before it answers."""

    def __init__(
        self,
        text: str = ECHO,
        *,
        by_length: dict[int, str] | None = None,
        by_index: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.text = text
        self.by_index = by_index or {}  # the answer to the request that is the n-th (from 0)
        self.by_length = by_length or {}  # the answer to a request of exactly this many ms
        self.error: ProviderError | None = None
        self.fail_first = False
        self.requests: list[int] = []
        self._armed = False
        self._gate = asyncio.Event()
        self.waiting = asyncio.Event()

    def arm(self) -> None:
        self._armed = True
        self._gate.clear()
        self.waiting.clear()

    def open(self) -> None:
        self._gate.set()

    async def transcribe(
        self, pcm16_16k: bytes, *, language: str | None = None, context: str | None = None
    ) -> Transcript:
        audio_ms = bytes_to_ms(len(pcm16_16k), CANONICAL_RATE)
        first = not self.requests
        index = len(self.requests)
        self.requests.append(audio_ms)
        if self._armed:
            self._armed = False
            self.waiting.set()
            await self._gate.wait()
        if self.error is not None or (self.fail_first and first):
            raise self.error or ProviderError("unavailable", "down")
        text = self.by_index.get(index, self.by_length.get(audio_ms, self.text))
        return Transcript(text=text, language=language, audio_ms=audio_ms)


def build(
    asr: HeardASR,
    speech: list[tuple[int, int]],
    clock: ManualClock,
    rules: Mapping[str, object] | None = None,
    *,
    lead_ms: int = 1000,
    turn_end: TurnEndDetector | None = None,
    tts: TTSBackend | None = None,
    aec: Mapping[str, object] | None = None,
) -> Gateway:
    config = {
        "output": {"max_lead_ms": lead_ms},
        "barge_in": {**RULES, **(rules or {})},
        "aec": dict(aec or {}),
    }
    tts = tts or SineTTS(ms_per_char=100, chunk_ms=100)
    providers = Providers(llm=EchoLLM(), asr=asr, tts=tts)
    return Gateway(
        validate_config(with_echo_off(config)),
        providers,
        speech_detector=lambda: ScriptedSpeech(speech),
        turn_end_detector=lambda: turn_end,
        clock=clock,
    )


def count(gateway: Gateway, name: str, **labels: str) -> float:
    for family in text_string_to_metric_families(gateway.metrics.exposition().decode()):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0.0


async def start_answer(client: Client, clock: ManualClock) -> None:
    """The long answer, playing from about 200 ms of input."""
    await client.start()
    await stream(client, clock, 200)
    await client.say(LONG)
    await client.send({"type": "response.create"})
    await client.until("response.output_audio.delta")


def of_type(client: Client, kind: str) -> list[Event]:
    return [event for event in client.events if event["type"] == kind]


async def test_an_echo_burst_over_a_playing_answer_is_dropped_unannounced() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)
    gateway = build(asr, [(3500, 5000)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4000)  # to 4.2 s: the burst began, the turn ends at ~5.5 s
        await drain(client)
        assert asr.requests, "the held turn was checked at once"
        await stream(client, clock, 3000)  # to 7.2 s
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        assert "input_audio_buffer.speech_stopped" not in types(client)
        assert "input_audio_buffer.committed" not in types(client)
        done = of_type(client, "response.done")
        assert [d["response"]["status"] for d in done] == ["completed"]  # never cut
        assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
        assert count(gateway, "mlxrt_held_turns_released_total") == 0
        assert count(gateway, "mlxrt_barge_ins_total") == 0
    assert "response.created" in types(client)
    assert types(client).count("response.created") == 1  # nothing answered the echo


async def test_an_echo_burst_over_the_queued_tail_after_response_done_is_dropped() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)
    gateway = build(asr, [(3500, 5000)], clock, lead_ms=10_000)  # all of it sent at once
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await client.until("response.done")  # the answer is sent; the client still plays it
        await stream(client, clock, 5400)  # to 5.6 s, past the turn's end
        await drain(client)
        kinds = types(client)
        assert "input_audio_buffer.speech_started" not in kinds
        assert "input_audio_buffer.speech_stopped" not in kinds
        assert "input_audio_buffer.committed" not in kinds
        assert kinds.count("response.created") == 1
        assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1


async def test_a_caller_over_the_answer_is_released_with_the_original_start() -> None:
    clock, asr = ManualClock(), HeardASR(CALLER)
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3000)
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)  # held
        await stream(client, clock, 1400)
        await drain(client)
        await stream(client, clock, 2000)
        await drain(client)
        assert count(gateway, "mlxrt_held_turns_released_total") == 1
        assert count(gateway, "mlxrt_barge_ins_total") == 1
    kinds = types(client)
    (started,) = of_type(client, "input_audio_buffer.speech_started")
    assert started["audio_start_ms"] == ONSET_START
    first_done = next(e for e in client.events if e["type"] == "response.done")
    assert first_done["response"]["status_details"] == {
        "type": "cancelled",
        "reason": "turn_detected",
    }
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index("response.done")
    assert kinds.index("input_audio_buffer.speech_stopped") < kinds.index(
        "input_audio_buffer.committed"
    )
    replies = of_type(client, "response.done")
    assert [r["response"]["status"] for r in replies] == ["cancelled", "completed"]


async def test_a_short_utterance_of_our_own_words_is_released_at_the_turns_end() -> None:
    clock, asr = ManualClock(), HeardASR(SHORT_ECHO)  # 3 units: consistent, and not an echo turn
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3800)  # to 4.0 s: held, and still held
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 1600)  # past the turn's end
        await drain(client)
        at_stop = len(asr.requests)
        assert "input_audio_buffer.speech_stopped" in types(client)
        reply = None
        while reply is None:
            clock.advance(0.02)
            await asyncio.sleep(0.002)
            reply = next(
                (
                    e
                    for e in client.events
                    if e["type"] == "response.done" and e["response"]["status"] == "completed"
                ),
                None,
            )
        # the final check's transcript was the commit's: no second request for the same audio
        assert len(asr.requests) == at_stop
        assert count(gateway, "mlxrt_held_turns_released_total") == 1
    kinds = types(client)
    (started,) = of_type(client, "input_audio_buffer.speech_started")
    assert started["audio_start_ms"] == ONSET_START
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index(
        "input_audio_buffer.speech_stopped"
    )
    cancelled = [
        r for r in of_type(client, "response.done") if r["response"]["status"] == "cancelled"
    ]
    assert len(cancelled) == 1


async def test_noise_over_the_answer_is_dropped_with_no_barge_in() -> None:
    clock, asr = ManualClock(), HeardASR("")
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 5000)
        await drain(client)
        kinds = types(client)
        assert "input_audio_buffer.speech_started" not in kinds
        assert "input_audio_buffer.speech_stopped" not in kinds
        assert "input_audio_buffer.committed" not in kinds
        assert count(gateway, "mlxrt_held_turns_dropped_total", reason="noise") == 1
        assert count(gateway, "mlxrt_barge_ins_total") == 0
        await stream(client, clock, 4000)
        done = (await client.until("response.done"))[-1]["response"]
        assert done["status"] == "completed"  # the answer was never cut


async def test_an_asr_failure_during_a_check_releases_the_turn() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)
    asr.fail_first = True
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        (started,) = of_type(client, "input_audio_buffer.speech_started")
        assert started["audio_start_ms"] == ONSET_START
        done = next(e for e in client.events if e["type"] == "response.done")
        assert done["response"]["status_details"] == {
            "type": "cancelled",
            "reason": "turn_detected",
        }
        assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_a_check_over_its_time_releases_the_turn() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)
    gateway = build(asr, [(3500, 4300)], clock, {"confirm_timeout_ms": 100})
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3500)  # to 3.7 s: before the hold begins
        asr.arm()  # the first check never answers
        await stream(client, clock, 1000)
        async with asyncio.timeout(3.0):
            await asr.waiting.wait()
        await asyncio.sleep(0.3)  # real time: the check's 100 ms
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")
        asr.open()


async def test_pure_echo_held_across_the_end_of_the_playback_is_dropped() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)  # consistent with our words all along
    gateway = build(asr, [(6500, 8500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 6500)  # to 6.7 s: held
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 1800)  # the answer (7.4 s) and its echo tail end: still held
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 2500)  # the turn ends, and is our echo
        await drain(client)
    kinds = types(client)
    assert "input_audio_buffer.speech_started" not in kinds
    assert "input_audio_buffer.speech_stopped" not in kinds
    assert "input_audio_buffer.committed" not in kinds
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    assert count(gateway, "mlxrt_held_turns_released_total") == 0


async def test_a_word_that_is_not_ours_releases_the_turn_when_the_playback_ends() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO + "价")  # one unit not ours: not enough while playing
    gateway = build(asr, [(6500, 9500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 6500)
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 1800)  # the playback ends
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")
        assert "input_audio_buffer.speech_stopped" not in types(client)  # the caller still talks
        assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_a_one_word_utterance_is_released_at_the_pause_not_the_turns_end() -> None:
    clock, asr = ManualClock(), HeardASR("喂")
    gateway = build(asr, [(3500, 4300)], clock, turn_end=ScriptedTurnEnd(default=0.0))
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3950)  # to 4.15 s: held; one word is not enough yet
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 1500)  # the gate ends the speech: the pause check
        await drain(client)
        (started,) = of_type(client, "input_audio_buffer.speech_started")
        assert started["audio_start_ms"] == ONSET_START
        # Smart Turn still waits for the rest of the turn (its 3 s fallback)
        assert "input_audio_buffer.speech_stopped" not in types(client)
        await stream(client, clock, 3500)
        await drain(client)
        assert "input_audio_buffer.speech_stopped" in types(client)


async def test_no_check_runs_through_the_silence_and_the_utterance_check_is_the_final_one() -> None:
    clock, asr = ManualClock(), HeardASR(SHORT_ECHO)  # all ours, too short to be an echo turn
    gateway = build(asr, [(3500, 4300)], clock, turn_end=ScriptedTurnEnd(default=0.0))
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 5200)  # to 5.4 s: the speech ended at 4.3 s, the pause checked
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        asked = len(asr.requests)
        await stream(client, clock, 1200)  # silence, and the turn is not over
        await drain(client)
        assert len(asr.requests) == asked
        await stream(client, clock, 2500)  # Smart Turn's fallback ends the turn
        await drain(client)
        assert "input_audio_buffer.speech_stopped" in types(client)
        sent = len(asr.requests)
        await stream(client, clock, 1000)
        await drain(client)
        assert len(asr.requests) == sent  # the pause check was the commit's transcript too
    kinds = types(client)
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index(
        "input_audio_buffer.speech_stopped"
    )
    assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_speech_that_resumes_after_a_pause_goes_back_to_partial_checks() -> None:
    clock, asr = ManualClock(), HeardASR(SHORT_ECHO)
    gateway = build(asr, [(3500, 4300), (5000, 6200)], clock, turn_end=ScriptedTurnEnd(default=0.0))
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4900)  # to 5.1 s: the pause is checked, the speech resumes
        await drain(client)
        paused = len(asr.requests)
        longest = max(asr.requests)
        await stream(client, clock, 1400)  # the resumed speech: partial checks again
        await drain(client)
        assert len(asr.requests) > paused
        assert max(asr.requests) > longest
        assert "input_audio_buffer.speech_started" not in types(client)


async def test_an_echo_turn_with_two_misheard_units_is_dropped_at_the_final_decision() -> None:
    """The partial checks hear our words; the whole turn has 2 units that are not (so a partial
    check would release it), but 9 of 11 are, and the echo-turn rule decides."""
    clock = ManualClock()
    misheard = "很长很长的故事越长价格"  # 11 units, 2 not ours
    probe = HeardASR(misheard)
    gateway = build(probe, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
    final = probe.requests[-1]  # the check over the whole turn is the last one
    clock, asr = ManualClock(), HeardASR(ECHO, by_length={final: misheard})
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        assert asr.requests[-1] == final
        assert asr.requests.count(final) == 1  # no partial check was the whole turn
    assert "input_audio_buffer.speech_started" not in types(client)
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1


async def test_an_echo_garbled_in_the_middle_is_still_dropped() -> None:
    """One misheard unit splits the echo in two long runs: 10 of its 11 units are in one."""
    clock, asr = ManualClock(), HeardASR("很长很长的故障越长越好")
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
    assert "input_audio_buffer.speech_started" not in types(client)
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1


async def test_a_caller_who_reuses_our_words_in_short_fragments_is_released() -> None:
    """Every unit is ours and in order, so the plain overlap would call it an echo, but no
    three of them follow each other as we said them: a caller picking words from the answer."""
    quoted = "很很的事越好"  # 6 units, no unit that is not ours, contiguity 0
    clock, asr = ManualClock(), HeardASR(quoted)
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 0
        assert count(gateway, "mlxrt_held_turns_released_total") == 1
    assert of_type(client, "input_audio_buffer.speech_started")


PREFIX = "请讲一个很长很长的故事越长越好一直讲下去不要停讲完一个再讲下一个"  # 31 units of LONG
QUESTION = "我想问一下价格吧"


async def merged_turn(final_text: str) -> tuple[Client, Gateway]:
    """A held turn whose partial checks hear our words and whose final check hears
    ``final_text``."""
    clock = ManualClock()
    probe = HeardASR(PREFIX)
    gateway = build(probe, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
    final = probe.requests[-1]
    clock, asr = ManualClock(), HeardASR(PREFIX, by_length={final: final_text})
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        await stream(client, clock, 2000)
        await drain(client)
    return client, gateway


async def test_a_caller_who_spoke_into_a_held_echo_is_released_with_what_they_said() -> None:
    """Issue #8: every partial check hears our words; the final one hears them and the caller's
    question after them (31 of 39 units are ours: an echo turn as a whole). The turn is released
    at the final decision, and the question is its transcript."""
    client, gateway = await merged_turn(PREFIX + QUESTION)
    assert count(gateway, "mlxrt_held_turns_released_total") == 1
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 0
    assert of_type(client, "input_audio_buffer.speech_started")
    (done,) = of_type(client, "conversation.item.input_audio_transcription.completed")
    assert done["transcript"] == QUESTION
    reply = of_type(client, "response.done")[-1]["response"]
    assert reply["output"][0]["content"][0]["transcript"] == QUESTION


async def test_a_held_echo_with_a_garbled_tail_is_still_dropped() -> None:
    """The ASR garbles the last three units of the echo into words that are not ours: they come
    after the prefix, but they are too few to be a caller (the same rule as two misheard units)."""
    client, gateway = await merged_turn(PREFIX[:-3] + "货不出")
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    assert "input_audio_buffer.speech_started" not in types(client)


async def test_a_turn_held_only_for_order_is_released_at_the_front_once_nothing_plays() -> None:
    clock, asr = ManualClock(), HeardASR(SHORT_ECHO)
    gateway = build(asr, [(3500, 4300), (8200, 9000)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3000)
        await drain(client)
        asr.arm()  # the first turn's next check waits
        await stream(client, clock, 6000)  # to 9.2 s: the second turn began after the playback
        async with asyncio.timeout(3.0):
            await asr.waiting.wait()
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        asr.text = CALLER
        asr.open()
        await stream(client, clock, 1500)
        await drain(client)
    kinds = [
        e["type"].rsplit("_", 1)[1]
        for e in client.events
        if e["type"] in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped")
    ]
    assert kinds == ["started", "stopped", "started", "stopped"]
    assert count(gateway, "mlxrt_held_turns_released_total") == 2


@pytest.mark.parametrize("how", ["clear", "update"])
async def test_a_reset_cancels_a_held_turn_and_its_check(how: str) -> None:
    clock, asr = ManualClock(), HeardASR(CALLER)  # would release the turn, if it counted
    gateway = build(asr, [(3500, 4300)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3000)
        asr.arm()
        await stream(client, clock, 900)  # the turn is held; a check waits for the ASR
        async with asyncio.timeout(3.0):
            await asr.waiting.wait()
        if how == "clear":
            await client.send({"type": "input_audio_buffer.clear"})
        else:
            update = {
                "audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 0.6}}}
            }
            await client.send(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "audio": update["audio"]},
                }
            )
        await drain(client)
        asr.open()
        await stream(client, clock, 1000)
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        assert count(gateway, "mlxrt_held_turns_released_total") == 0


async def test_without_confirm_while_playing_the_onset_interrupts_at_once() -> None:
    clock, asr = ManualClock(), HeardASR(ECHO)
    gateway = build(asr, [(3500, 4300)], clock, {"confirm_while_playing": False})
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        (started,) = of_type(client, "input_audio_buffer.speech_started")
        assert started["audio_start_ms"] == ONSET_START
        assert len(asr.requests) == 1  # the commit's only: no check
        done = next(e for e in client.events if e["type"] == "response.done")
        assert done["response"]["status_details"] == {
            "type": "cancelled",
            "reason": "turn_detected",
        }


async def test_a_turn_that_starts_behind_a_held_one_keeps_the_events_in_order() -> None:
    clock, asr = ManualClock(), HeardASR(SHORT_ECHO)  # holds on every check
    gateway = build(asr, [(3500, 4300), (5600, 6400)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 3000)
        await drain(client)
        asr.arm()  # the next check, then, waits
        await stream(client, clock, 3100)  # to 6.3 s: the first turn ended, the second began
        async with asyncio.timeout(3.0):
            await asr.waiting.wait()
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        assert "input_audio_buffer.speech_stopped" not in types(client)
        asr.text = CALLER
        asr.open()
        await stream(client, clock, 1500)
        await drain(client)
    order = [
        (e["type"], e["item_id"])
        for e in client.events
        if e["type"] in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped")
    ]
    kinds = [kind.rsplit("_", 1)[1] for kind, _ in order]
    assert kinds == ["started", "stopped", "started", "stopped"]
    assert order[0][1] == order[1][1] != order[2][1] == order[3][1]
    starts = of_type(client, "input_audio_buffer.speech_started")
    assert starts[0]["audio_start_ms"] == ONSET_START < starts[1]["audio_start_ms"]


MISHEARD_EDGE = ECHO[:-2] + "价格"  # 11 units, 2 not ours, one of them the last unit
EDGE_CHECK = 1  # the second partial check: the first heard only our words


async def probe_checks(
    spans: list[tuple[int, int]], run_ms: int, *, text: str = ECHO, tail_ms: int = 0
) -> list[int]:
    """The lengths of the ASR requests of a hold whose checks all hear ``text``."""
    clock, asr = ManualClock(), HeardASR(text)
    gateway = build(asr, spans, clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, run_ms)
        await drain(client)
        if tail_ms:
            await stream(client, clock, tail_ms)
            await drain(client)
    return asr.requests


async def test_a_mishearing_at_a_checks_cut_edge_does_not_release_a_held_echo() -> None:
    """Issue #10: the checks hear our words, then one hears two units that are not ours, one of
    them its last unit (a syllable the cut truncated). The hold waits for the next check, which
    hears our words again, and the turn ends as an echo."""
    lengths = await probe_checks([(3500, 5500)], 6000)
    assert len(lengths) > EDGE_CHECK + 2
    clock, asr = ManualClock(), HeardASR(ECHO, by_length={lengths[EDGE_CHECK]: MISHEARD_EDGE})
    gateway = build(asr, [(3500, 5500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 4400)
        await drain(client)
        assert asr.requests.count(lengths[EDGE_CHECK]) == 1  # the mishearing was heard
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 4000)
        await drain(client)
    assert "input_audio_buffer.speech_started" not in types(client)
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    assert count(gateway, "mlxrt_held_turns_released_total") == 0


async def test_units_that_are_not_ours_in_two_checks_release_a_held_echo() -> None:
    """A caller who joins the echo: the units that are not ours come in the next check too, and
    the hold is released then, one check interval later than a fresh hold would be."""
    lengths = await probe_checks([(3500, 5500)], 6000)
    clock, asr = (
        ManualClock(),
        HeardASR(
            ECHO,
            by_length={
                lengths[EDGE_CHECK]: MISHEARD_EDGE,
                lengths[EDGE_CHECK + 1]: MISHEARD_EDGE,
            },
        ),
    )
    gateway = build(asr, [(3500, 5500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        # the check before the mishearing is answered; the mishearing is held
        while len(asr.requests) <= EDGE_CHECK:
            await stream(client, clock, 20)
            await drain(client)
        await stream(client, clock, 60)
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        while len(asr.requests) <= EDGE_CHECK + 1:
            await stream(client, clock, 20)
            await drain(client)
        await stream(client, clock, 60)
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")
        assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_a_caller_over_a_matched_hold_is_released_when_his_words_pass_the_cut_edge() -> None:
    """Two units not ours before the last one need no second check, matched hold or not."""
    lengths = await probe_checks([(3500, 5500)], 6000)
    said = ECHO[:-3] + "价格吧"  # 3 not ours, 2 before the last
    clock, asr = ManualClock(), HeardASR(ECHO, by_length={lengths[EDGE_CHECK]: said})
    gateway = build(asr, [(3500, 5500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        while len(asr.requests) <= EDGE_CHECK:
            await stream(client, clock, 20)
            await drain(client)
        await stream(client, clock, 60)
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")


async def test_a_fresh_hold_is_released_at_its_first_check_as_before() -> None:
    clock, asr = ManualClock(), HeardASR(CALLER)  # no check of the hold matched our words
    gateway = build(asr, [(3500, 5500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        while not asr.requests:
            await stream(client, clock, 20)
            await drain(client)
        await stream(client, clock, 60)
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")
    assert len(asr.requests) == 1


async def test_a_matched_hold_with_one_word_not_ours_is_released_after_the_playback_ends() -> None:
    """The early release agrees with the rule: a single unit that is not ours does not release a
    matched hold while the answer plays, and when it persists once nothing plays (one unit is
    enough then) the hold is released, at the check that sees it twice."""
    lengths = await probe_checks([(6500, 9500)], 6500, tail_ms=1800)
    word = ECHO + "价"
    clock = ManualClock()
    asr = HeardASR(ECHO, by_length=dict.fromkeys(lengths[1:], word))
    gateway = build(asr, [(6500, 9500)], clock)
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream(client, clock, 6500)
        await drain(client)
        assert "input_audio_buffer.speech_started" not in types(client)
        await stream(client, clock, 1800)  # the playback ends
        await drain(client)
        assert of_type(client, "input_audio_buffer.speech_started")
        assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_a_mishearing_at_a_slid_windows_first_unit_does_not_release_a_matched_hold() -> None:
    """Once the 4 s cap slides the window's start past the hold's, the first unit of a check is
    cut as the last one is: two units not ours at the start count as one. The same two units in a
    window that starts with the hold are released."""
    lengths = await probe_checks([(3500, 9500)], 8800)
    slid = lengths.index(4000)  # the first check whose window slid
    said = "价格" + ECHO[2:]  # 11 units, 2 not ours, at the start

    async def run(index: int) -> tuple[Client, Gateway]:
        clock, asr = ManualClock(), HeardASR(ECHO, by_index={index: said})
        gateway = build(asr, [(3500, 9500)], clock)
        async with running(gateway) as address, Client.connect(address) as client:
            await start_answer(client, clock)
            while len(asr.requests) <= index:
                await stream(client, clock, 20)
                await drain(client)
            await stream(client, clock, 60)
            await drain(client)
            held = not of_type(client, "input_audio_buffer.speech_started")
            await stream(client, clock, 4500)
            await drain(client)
        assert held == (index == slid)
        return client, gateway

    await run(2)  # a window that starts with the hold: released
    client, gateway = await run(slid)
    assert "input_audio_buffer.speech_started" not in types(client)
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1

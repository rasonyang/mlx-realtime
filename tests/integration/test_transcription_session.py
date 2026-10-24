"""Transcription sessions (``session.type: "transcription"``, issue #2): AICC's tap of a bridged
phone leg. The client appends audio all the time, silence included, commits after each
utterance and reads the transcripts. No model runs behind it: no LLM, no TTS, no speech
detection, no echo cancellation.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from mlx_realtime.server.gateway import Gateway
from mlx_realtime.session.actor import SessionActor
from tests.integration import test_echo_rules as echo_rules
from tests.support import Client, Event, ManualClock, build_gateway, running, tone

DONE = "conversation.item.input_audio_transcription.completed"
COMMIT: Event = {"type": "input_audio_buffer.commit"}

# What AICC's internal/transcribe/openairt.go sends after connecting.
AICC_UPDATE: Event = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {
                    "model": "gpt-4o-transcribe",
                    "language": "zh",
                    "prompt": "NovaNet",
                },
                "turn_detection": None,
            }
        },
    },
}


def silence(ms: int) -> bytes:
    return bytes(48 * ms)


async def open_session(client: Client, update: Event | None = AICC_UPDATE) -> Event:
    """Consume the connect events; apply ``update``; return ``session.created``'s session."""
    created = (await client.until("conversation.created"))[0]["session"]
    if update is not None:
        await client.send(update)
        await client.until("session.updated")
    return created


async def settled(active: int, gateway: Gateway) -> None:
    """Until ``active`` sessions are open."""
    async with asyncio.timeout(2):
        while gateway.admission.active != active:  # noqa: ASYNC110 (a counter, not an event)
            await asyncio.sleep(0.005)


async def test_the_intent_query_makes_a_transcription_session_from_the_start() -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        created = await open_session(c, None)
        assert created["type"] == "transcription"
        assert created["object"] == "realtime.transcription_session"
        assert created["audio"]["input"]["turn_detection"] is None
        assert created["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        # Only what applies: no model, instructions, tools or output.
        assert set(created) == {"type", "object", "id", "include", "audio"}
        assert set(created["audio"]) == {"input"}
        await c.send(AICC_UPDATE)
        updated = (await c.until("session.updated"))[-1]["session"]
    assert updated["type"] == "transcription"
    assert updated["audio"]["input"]["transcription"] == {
        "model": "gpt-4o-transcribe",
        "language": "zh",
        "prompt": "NovaNet",
    }
    assert updated["audio"]["input"]["turn_detection"] is None


async def test_a_commit_yields_the_item_and_its_transcript_and_asks_no_model() -> None:
    gateway, mocks = build_gateway(transcripts=("你好，NovaNet。",))
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(800))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
    assert [e["type"] for e in events] == [
        "input_audio_buffer.committed",
        "conversation.item.added",
        DONE,
        "conversation.item.done",
    ]
    committed, added, completed, done = events
    item_id = committed["item_id"]
    assert committed["previous_item_id"] is None
    assert added["item"]["id"] == done["item"]["id"] == completed["item_id"] == item_id
    assert completed["transcript"] == "你好，NovaNet。"
    assert completed["language"] == "zh"  # the mock reports the language it was given
    assert (mocks.asr.languages, mocks.asr.contexts) == (["zh"], ["NovaNet"])
    assert (mocks.llm.requests, mocks.tts.requests) == ([], [])
    assert not any(
        e["type"].startswith(("response.", "input_audio_buffer.speech")) for e in c.events
    )


async def test_items_chain_and_nothing_piles_up_in_the_session() -> None:
    gateway, _ = build_gateway(transcripts=("a", "b", "c"))
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        ids: list[str] = []
        previous: list[str | None] = []
        for _ in range(3):
            await c.append(tone(300))
            await c.send(COMMIT)
            committed = (await c.until("input_audio_buffer.committed"))[-1]
            ids.append(committed["item_id"])
            previous.append(committed["previous_item_id"])
            await c.until(DONE)
        (connection,) = gateway.admission._active  # pyright: ignore[reportPrivateUsage]
        actor: SessionActor = connection._actor  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        retained = list(actor._conversation)  # pyright: ignore[reportPrivateUsage]
    assert previous == [None, ids[0], ids[1]]
    assert retained == []


@pytest.mark.parametrize("audio_ms", [0, 600])
async def test_an_empty_or_silent_commit_is_an_empty_transcript_not_an_error(audio_ms: int) -> None:
    gateway, mocks = build_gateway(transcripts=("never asked",))
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        if audio_ms:
            await c.append(silence(audio_ms))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
        await c.quiet()
    completed = next(e for e in events if e["type"] == DONE)
    assert completed["transcript"] == ""
    assert [e["type"] for e in events][:2] == [
        "input_audio_buffer.committed",
        "conversation.item.added",
    ]
    assert mocks.asr.requests == []
    assert not any(e["type"] == "error" for e in c.events)


async def test_two_commits_in_a_row_the_second_is_empty() -> None:
    gateway, mocks = build_gateway(transcripts=("你好。",))
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(500))
        await c.send(COMMIT)
        await c.send(COMMIT)
        # The empty one is answered at once, so it may overtake the first one's transcript.
        async with asyncio.timeout(5):
            while len([e for e in c.events if e["type"] == "conversation.item.done"]) < 2:
                await c.recv()
        await c.quiet()
    committed = [e["item_id"] for e in c.events if e["type"] == "input_audio_buffer.committed"]
    transcripts = {e["item_id"]: e["transcript"] for e in c.events if e["type"] == DONE}
    assert transcripts == {committed[0]: "你好。", committed[1]: ""}
    assert len(mocks.asr.requests) == 1
    assert not any(e["type"] == "error" for e in c.events)


async def test_realtime_only_events_are_rejected_by_name_and_the_session_survives() -> None:
    gateway, mocks = build_gateway()
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    sent: list[Event] = [
        {"type": "response.create"},
        {"type": "response.cancel"},
        {"type": "output_audio_buffer.clear"},
        {"type": "conversation.item.create", "item": item},
        {"type": "conversation.item.delete", "item_id": "item_x"},
        {
            "type": "conversation.item.truncate",
            "item_id": "i",
            "content_index": 0,
            "audio_end_ms": 0,
        },
    ]
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        errors = []
        for event in sent:
            await c.send({**event, "event_id": "c1"})
            errors.append((await c.until("error"))[-1]["error"])
        await c.append(tone(300))
        await c.send(COMMIT)
        await c.until(DONE)
    for event, error in zip(sent, errors, strict=True):
        assert (error["type"], error["code"]) == ("invalid_request_error", "invalid_event")
        assert event["type"] in error["message"]
    assert [e["param"] for e in errors] == ["type"] * 6
    assert errors[0]["event_id"] == "c1"
    assert (mocks.llm.requests, mocks.tts.requests) == ([], [])


async def test_turn_detection_and_other_rates_are_rejected() -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c, None)
        for audio, param in (
            ({"turn_detection": {"type": "server_vad"}}, "session.audio.input.turn_detection"),
            ({"format": {"type": "audio/pcm", "rate": 16000}}, "session.audio.input.format.rate"),
        ):
            await c.send(
                {
                    "type": "session.update",
                    "session": {"type": "transcription", "audio": {"input": audio}},
                }
            )
            error = (await c.until("error"))[-1]["error"]
            assert (error["code"], error["param"]) == ("invalid_value", param)
        # An unknown model is accepted and echoed, as in realtime sessions; G.711 is accepted.
        await c.send(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcmu"},
                            "transcription": {"model": "nope"},
                        }
                    },
                },
            }
        )
        updated = (await c.until("session.updated"))[-1]["session"]
    assert updated["audio"]["input"]["transcription"]["model"] == "nope"
    assert updated["audio"]["input"]["format"] == {"type": "audio/pcmu"}


async def test_the_type_can_change_until_the_session_has_had_audio() -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address) as c:  # no intent
        created = await open_session(c, None)
        assert created["type"] == "realtime"
        await c.send(AICC_UPDATE)
        updated = (await c.until("session.updated"))[-1]["session"]
        assert updated["type"] == "transcription"
        assert updated["audio"]["input"]["turn_detection"] is None
        # Back to realtime while nothing has happened: the realtime defaults again.
        await c.send({"type": "session.update", "session": {"type": "realtime"}})
        back = (await c.until("session.updated"))[-1]["session"]
        assert back["type"] == "realtime"
        assert back["audio"]["input"]["turn_detection"]["type"] == "server_vad"
        await c.send(AICC_UPDATE)
        await c.until("session.updated")
        await c.append(tone(300))
        await c.send(
            {"type": "session.update", "session": {"type": "realtime", "instructions": "x"}}
        )
        error = (await c.until("error"))[-1]["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.type")
        # Still a working transcription session.
        await c.send(COMMIT)
        await c.until(DONE)


async def test_a_realtime_session_cannot_become_a_transcription_session_after_audio() -> None:
    gateway, _ = build_gateway()
    async with running(gateway) as address, Client.connect(address) as c:
        await open_session(c, None)
        await c.append(tone(100))
        await c.send(AICC_UPDATE)
        error = (await c.until("error"))[-1]["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "session.type")
        await c.send({"type": "session.update", "session": {"type": "realtime"}})
        assert (await c.until("session.updated"))[-1]["session"]["type"] == "realtime"


async def test_malformed_base64_is_an_error_and_the_session_survives() -> None:
    gateway, _ = build_gateway(transcripts=("ok",))
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.send({"type": "input_audio_buffer.append", "audio": "!!not base64!!"})
        error = (await c.until("error"))[-1]["error"]
        assert (error["code"], error["param"]) == ("invalid_value", "audio")
        await c.append(tone(300))
        await c.send(COMMIT)
        assert (await c.until(DONE))[-1]["transcript"] == "ok"


async def test_a_client_that_leaves_mid_utterance_leaks_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway, mocks = build_gateway()
    mocks.asr.result.hold()
    with caplog.at_level("WARNING"):
        async with running(gateway) as address:
            async with Client.connect(address, intent="transcription") as c:
                await open_session(c)
                await c.append(tone(500))
                await c.send(COMMIT)
                await mocks.asr.result.held()  # transcribing when the client goes
                await c.append(tone(200))  # and a second utterance in progress
            await settled(0, gateway)
            mocks.asr.result.release()
    assert [r for r in caplog.records if r.levelname in ("ERROR", "CRITICAL")] == []


async def test_the_session_limit_counts_both_kinds_and_the_fifth_is_refused() -> None:
    gateway, _ = build_gateway({"gateway": {"max_sessions": 4}})
    async with running(gateway) as address:
        async with (
            Client.connect(address, intent="transcription") as a,
            Client.connect(address, intent="transcription") as b,
            Client.connect(address) as c,
            Client.connect(address) as d,
        ):
            for client in (a, b, c, d):
                await client.until("conversation.created")
            async with Client.connect(address, intent="transcription") as fifth:
                error = (await fifth.until("error"))[-1]["error"]
                await fifth.ws.wait_closed()
            assert (error["type"], error["code"]) == ("server_error", "session_limit_reached")
            assert fifth.ws.close_code == 1013
        await settled(0, gateway)
        async with Client.connect(address, intent="transcription") as again:
            assert (await again.recv())["type"] == "session.created"


async def test_a_silence_far_beyond_the_buffer_maximum_does_not_end_the_session() -> None:
    gateway, mocks = build_gateway(
        {"limits": {"input_audio_buffer_max_ms": 2000, "inbox_max_ms": 120_000}},
        transcripts=("after the hold",),
    )
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        for _ in range(900):  # 90 s of silence in 100 ms appends, with pings in between
            await c.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(silence(100)).decode(),
                }
            )
            if _ % 100 == 0:
                await c.ws.ping()
        await c.append(tone(600))
        await c.send(COMMIT)
        completed = (await c.until(DONE))[-1]
    assert completed["transcript"] == "after the hold"
    # The ASR heard the utterance with at most the second of silence that was kept before it.
    assert mocks.asr.requests[0] <= 1000 + 600 + 60
    assert not any(e["type"] == "error" for e in c.events)


async def test_speech_over_the_buffer_maximum_loses_its_oldest_audio_not_the_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway, mocks = build_gateway(
        {"limits": {"input_audio_buffer_max_ms": 1000, "inbox_max_ms": 10_000}}
    )
    with caplog.at_level("WARNING", logger="mlx_realtime.session.input_buffer"):
        async with (
            running(gateway) as address,
            Client.connect(address, intent="transcription") as c,
        ):
            await open_session(c)
            await c.append(tone(3000))
            await c.send(COMMIT)
            await c.until(DONE)
            await c.append(tone(300))  # and it goes on
            await c.send(COMMIT)
            await c.until(DONE)
    assert mocks.asr.requests[0] <= 1000
    assert any("oldest audio" in r.getMessage() for r in caplog.records)
    assert not any(e["type"] == "error" for e in c.events)


async def test_transcription_turns_show_in_the_asr_stage_metrics() -> None:
    gateway, _ = build_gateway({"observability": {"debug_endpoints": True}})
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(300))
        await c.send(COMMIT)
        await c.until(DONE)
        async with asyncio.timeout(2):
            while "asr_done" not in gateway.metrics.windowed():  # noqa: ASYNC110
                await asyncio.sleep(0.005)
        records = gateway.metrics.debug
        turns = [records.session(session) for session in records.sessions()]
    assert turns[0] is not None
    assert turns[0][0]["values"] == {"session_type": "transcription"}


# The audio path is made again when a session still without audio changes type. The feeder once
# kept the echo reference of the type the session started as.


def _feeder_reference(gateway: Gateway) -> object:
    (connection,) = gateway.admission._active  # pyright: ignore[reportPrivateUsage]
    actor: SessionActor = connection._actor  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    assert actor._feeder._reference is actor._echo  # pyright: ignore[reportPrivateUsage]
    return actor._echo  # pyright: ignore[reportPrivateUsage]


async def test_a_realtime_session_that_becomes_a_transcription_session_transcribes() -> None:
    config = {"barge_in": {"echo_guard_ms": 2000, "min_speech_ms_while_playing": 400}}
    # A scripted speech turn: a transcription session must not run speech detection on it.
    gateway, mocks = build_gateway(config, transcripts=("你好，NovaNet。",), speech=[(0, 600)])
    async with running(gateway) as address, Client.connect(address) as c:  # no intent
        created = await open_session(c, None)
        assert created["type"] == "realtime"
        assert _feeder_reference(gateway) is not None  # the realtime path: an echo reference
        await c.send(AICC_UPDATE)
        await c.until("session.updated")
        assert _feeder_reference(gateway) is None  # no echo path, and the feeder agrees
        await c.append(tone(800))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
        await c.quiet()
    assert next(e for e in events if e["type"] == DONE)["transcript"] == "你好，NovaNet。"
    assert (mocks.llm.requests, mocks.tts.requests) == ([], [])
    assert not any(
        e["type"].startswith(("response.", "input_audio_buffer.speech")) for e in c.events
    )


async def test_a_transcription_session_that_becomes_a_realtime_session_gets_its_echo_path() -> None:
    clock = ManualClock()
    config = {
        "output": {"max_lead_ms": 1000},
        "barge_in": {"echo_guard_ms": 2000, "min_speech_ms_while_playing": 400},
    }
    # Speech at 1000-1800 ms (inside the echo guard of the answer playing from about 200 ms),
    # then at 3500-4300 ms (after it): as in test_echo_rules, now on a session that began as a
    # transcription session.
    gateway, _ = build_gateway(
        config, speech=[(1000, 1800), (3500, 4300)], ms_per_char=100, chunk_ms=100, clock=clock
    )
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        created = await open_session(c, None)
        assert created["type"] == "transcription"
        assert _feeder_reference(gateway) is None
        await c.send({"type": "session.update", "session": {"type": "realtime"}})
        await c.until("session.updated")
        assert _feeder_reference(gateway) is not None  # attached again
        await echo_rules.stream(c, clock, 200)
        await c.say(echo_rules.LONG)
        await c.send({"type": "response.create"})
        await c.until("response.output_audio.delta")
        await echo_rules.stream(c, clock, 3000)  # the guarded onset came and went
        await echo_rules.drain(c)
        assert "input_audio_buffer.speech_started" not in echo_rules.types(c)
        await echo_rules.stream(c, clock, 1400)  # the second onset interrupts
        await echo_rules.drain(c)
    kinds = echo_rules.types(c)
    assert kinds.index("response.done") > kinds.index("input_audio_buffer.speech_started")


# The speech check (transcription.min_speech_ms): AICC's pump commits hold music and DTMF, which
# Qwen3-ASR answers with its prompt or by running to its token limit.

CHECK = {"transcription": {"min_speech_ms": 224, "speech_threshold": 0.5}}


async def test_a_commit_with_no_speech_is_answered_empty_without_asking_the_asr() -> None:
    # A loud tone (voiced, so the buffer holds it) that the detector finds no speech in.
    gateway, mocks = build_gateway(CHECK, transcripts=("NovaNet, eSIM。",), speech=[])
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(2000))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
        await c.append(tone(300))  # the session goes on
        await c.send(COMMIT)
        await c.until("conversation.item.done")
        metrics = gateway.metrics.exposition().decode()
    completed = next(e for e in events if e["type"] == DONE)
    assert completed["transcript"] == ""
    assert "usage" in completed
    assert (mocks.asr.languages, mocks.asr.contexts) == ([], [])  # no ASR request at all
    assert "mlxrt_transcription_skipped_total 2.0" in metrics


@pytest.mark.parametrize(("speech_ms", "asked"), [(192, False), (256, True)])
async def test_the_commit_needs_min_speech_ms_of_speech_to_reach_the_asr(
    speech_ms: int, asked: bool
) -> None:
    gateway, mocks = build_gateway(CHECK, transcripts=("你好。",), speech=[(0, speech_ms)])
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(1000))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
    completed = next(e for e in events if e["type"] == DONE)
    assert completed["transcript"] == ("你好。" if asked else "")
    assert bool(mocks.asr.contexts) is asked


async def test_the_speech_check_is_off_at_zero_and_never_applies_to_realtime_sessions() -> None:
    off = {"transcription": {"min_speech_ms": 0}}
    gateway, mocks = build_gateway(off, transcripts=("你好。",), speech=[])
    async with running(gateway) as address, Client.connect(address, intent="transcription") as c:
        await open_session(c)
        await c.append(tone(500))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
    assert next(e for e in events if e["type"] == DONE)["transcript"] == "你好。"
    assert len(mocks.asr.contexts) == 1
    # A realtime session commits a buffer with no detected speech: still transcribed.
    gateway, mocks = build_gateway(CHECK, transcripts=("你好。",), speech=[])
    async with running(gateway) as address, Client.connect(address) as c:
        await c.start()
        await c.append(tone(500))
        await c.send(COMMIT)
        events = await c.until("conversation.item.done")
    assert next(e for e in events if e["type"] == DONE)["transcript"] == "你好。"

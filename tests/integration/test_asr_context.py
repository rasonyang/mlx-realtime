"""``transcription.prompt`` reaches the ASR as the turn's context.

The prompt in force when a turn is committed is the one its transcription gets; a later
``session.update`` applies from the next turn and never re-transcribes a committed one.
"""

from __future__ import annotations

from tests.support import Client, build_gateway, running, tone

MANUAL = {"audio": {"input": {"turn_detection": None}}}
DONE = "conversation.item.input_audio_transcription.completed"


def prompt(text: str | None) -> dict[str, object]:
    return {"audio": {"input": {"transcription": {"prompt": text}}}}


async def commit(client: Client) -> None:
    await client.append(tone(500))
    await client.send({"type": "input_audio_buffer.commit"})
    await client.until(DONE)


async def test_a_turn_carries_the_prompt_of_the_session_when_it_was_committed() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start(MANUAL)
        await commit(client)  # no prompt
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", **prompt("NovaNet")}}
        )
        await client.until("session.updated")
        await commit(client)
        await client.send({"type": "session.update", "session": {"type": "realtime", **prompt("")}})
        await client.until("session.updated")
        await commit(client)  # an empty prompt is no prompt
    assert mocks.asr.contexts == [None, "NovaNet", ""]


async def test_a_prompt_changed_while_a_turn_is_in_flight_only_affects_later_turns() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({**MANUAL, **prompt("NovaNet")})
        mocks.asr.result.hold()
        await client.append(tone(500))
        await client.send({"type": "input_audio_buffer.commit"})
        await mocks.asr.result.held()
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", **prompt("eSIM")}}
        )
        await client.until("session.updated")
        mocks.asr.result.release()
        await client.until(DONE)
        mocks.asr.result.release()
        await commit(client)
    assert mocks.asr.contexts == ["NovaNet", "eSIM"]


async def test_a_prompt_over_the_limit_is_rejected_and_the_session_keeps_its_own() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({**MANUAL, **prompt("NovaNet")})
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", **prompt("a" * 1025)}}
        )
        error = (await client.until("error"))[-1]["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "session.audio.input.transcription.prompt"
        assert "1024" in error["message"]
        await client.quiet()
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", **prompt("b" * 1024)}}
        )
        updated = (await client.until("session.updated"))[-1]["session"]
        assert updated["audio"]["input"]["transcription"]["prompt"] == "b" * 1024
        await commit(client)
    assert mocks.asr.contexts == ["b" * 1024]


async def test_a_rejected_update_leaves_the_session_unchanged() -> None:
    gateway, mocks = build_gateway()
    async with running(gateway) as address, Client.connect(address) as client:
        await client.start({**MANUAL, **prompt("NovaNet")})
        await client.send(
            {"type": "session.update", "session": {"type": "realtime", **prompt("a" * 2000)}}
        )
        await client.until("error")
        await commit(client)
    assert mocks.asr.contexts == ["NovaNet"]

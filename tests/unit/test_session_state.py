"""Truncation arithmetic, session merge rules, the delivery ledger, the inbox and the outbox."""

from __future__ import annotations

import asyncio

import pytest

from mlx_realtime.audio.formats import PCM, PCMA, PCMU
from mlx_realtime.providers.base import ToolDef
from mlx_realtime.server.outbox import Outbound, Outbox
from mlx_realtime.session.commands import AppendAudio, CommitAudio, UpdateSession
from mlx_realtime.session.conversation import AssistantMessage, SpokenSegment
from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.inbox import Inbox
from mlx_realtime.session.output import Delivery, OutputLedger
from mlx_realtime.session.settings import (
    ServerVad,
    ServerVadPatch,
    SessionPatch,
    SessionSettings,
    Transcription,
    TranscriptionPatch,
)
from mlx_realtime.session.signals import SpeechFinished
from mlx_realtime.session.truncation import clamp_audio_end, heard_text
from mlx_realtime.text.normalize import normalize_with_map


def segments(*parts: tuple[str, int]) -> list[SpokenSegment]:
    """Consecutive segments of ``(text, audio_ms)`` at 24 kHz."""
    result: list[SpokenSegment] = []
    start = 0
    for text, audio_ms in parts:
        result.append(SpokenSegment(text, text, start, audio_ms * 48))
        start += audio_ms
    return result


SPOKEN = segments(("你好。", 600), ("我是小王，", 1000), ("Thanks for calling today.", 2000))


@pytest.mark.parametrize(
    ("audio_end_ms", "heard"),
    [
        (0, ""),
        (300, "你"),  # half of a three-character segment
        (600, "你好。"),
        (1100, "你好。我是"),
        (1600, "你好。我是小王，"),
        (2000, "你好。我是小王，"),  # 20% of 25 characters ends inside "Thanks": snapped back
        (2120, "你好。我是小王，Thanks"),  # 26%: 6 characters
        (2480, "你好。我是小王，Thanks for "),  # 44%: 11 characters
        (3600, "你好。我是小王，Thanks for calling today."),
        (9999, "你好。我是小王，Thanks for calling today."),
    ],
)
def test_heard_text_is_whole_segments_plus_a_proportional_prefix(
    audio_end_ms: int, heard: str
) -> None:
    assert heard_text(SPOKEN, audio_end_ms) == heard


def test_segments_without_audio_count_only_before_the_cut() -> None:
    parts = segments(("好。", 500), ("\n", 0), ("再见。", 500))
    assert heard_text(parts, 500) == "好。"
    assert heard_text(parts, 501) == "好。\n"


def normalized_segment(text: str, audio_ms: int) -> SpokenSegment:
    spoken, source = normalize_with_map(text)
    return SpokenSegment(text, spoken, 0, audio_ms * 48, source)


def test_the_cut_is_measured_in_what_was_spoken_and_mapped_back() -> None:
    """The audio speaks the normalized text: "138 0013 8000" read digit by digit takes most of
    the segment. 80 % into the audio the reader is inside the last group, so that group is not
    heard; measured on the written text, the cut would claim the whole number and more."""
    segment = normalized_segment("Call 138 0013 8000 now, please.", 3000)
    assert segment.spoken.startswith("Call one three eight, zero zero one three, eight zero")
    assert heard_text([segment], 2400) == "Call 138 0013 "
    assert heard_text([segment], 1200) == "Call 138 "
    assert heard_text([segment], 3000) == segment.text
    written = SpokenSegment(segment.text, segment.text, 0, 3000 * 48)  # no map: as before
    assert heard_text([written], 2400) == "Call 138 0013 8000 now, "


def test_without_normalization_the_map_changes_nothing() -> None:
    parts = [normalized_segment(text, ms) for text, ms in (("你好。", 600),)]
    assert heard_text(parts, 300) == "你"


def test_truncate_overshoot_is_clamped_to_the_audio_sent() -> None:
    assert clamp_audio_end(1500, sent_ms=1200) == 1200
    assert clamp_audio_end(800, sent_ms=1200) == 800


DEFAULTS = SessionSettings(
    transcription=Transcription(model="qwen3-asr"),
    turn_detection=ServerVad(silence_duration_ms=200),
)


def test_merge_changes_only_present_fields() -> None:
    patched = DEFAULTS.apply(SessionPatch(instructions="New."), DEFAULTS)
    assert patched == SessionSettings(
        instructions="New.",
        transcription=DEFAULTS.transcription,
        turn_detection=DEFAULTS.turn_detection,
    )


def test_nested_objects_merge_field_by_field() -> None:
    patched = DEFAULTS.apply(
        SessionPatch(turn_detection=ServerVadPatch(silence_duration_ms=500)), DEFAULTS
    )
    assert patched.turn_detection == ServerVad(silence_duration_ms=500)
    language = SessionPatch(transcription=TranscriptionPatch(language="zh"))
    assert DEFAULTS.apply(language, DEFAULTS).transcription == Transcription("qwen3-asr", "zh")


def test_null_clears_and_a_fresh_object_starts_from_the_defaults() -> None:
    cleared = DEFAULTS.apply(SessionPatch(turn_detection=None, transcription=None), DEFAULTS)
    assert (cleared.turn_detection, cleared.transcription) == (None, None)
    restored = cleared.apply(
        SessionPatch(
            turn_detection=ServerVadPatch(threshold=0.7), transcription=TranscriptionPatch()
        ),
        DEFAULTS,
    )
    assert restored.turn_detection == ServerVad(threshold=0.7, silence_duration_ms=200)
    assert restored.transcription == Transcription(model="qwen3-asr")


def test_tools_and_tokens_replace() -> None:
    tools = (ToolDef("f", None, None),)
    patched = DEFAULTS.apply(SessionPatch(tools=tools, max_output_tokens=10), DEFAULTS)
    assert (patched.tools, patched.max_output_tokens) == (tools, 10)
    assert patched.apply(SessionPatch(max_output_tokens=None), DEFAULTS).max_output_tokens is None


def test_ledger_counts_what_was_handed_to_the_socket() -> None:
    ledger = OutputLedger()
    audio = Delivery(1, "item_1", 0, "audio", 4800)
    ledger.record(Delivery(1, "item_1", 0, "transcript"))
    ledger.record(audio)
    ledger.record(audio)
    assert ledger.sent_ms("item_1") == 200
    assert ledger.transcript_sent("item_1", 0)
    assert not ledger.transcript_sent("item_1", 1)
    assert not ledger.is_stale(audio)
    ledger.discard(1)
    assert ledger.is_stale(audio)
    assert not ledger.is_stale(Delivery(2, "item_2", 0, "audio", 10))


async def test_inbox_bounds_client_audio_and_commands() -> None:
    inbox = Inbox(max_audio_ms=100, max_commands=2)
    inbox.put_command(AppendAudio(b"\x00" * 4800))  # 100 ms at 24 kHz
    with pytest.raises(SessionFatal) as raised:
        inbox.put_command(AppendAudio(b"\x00\x00"))
    assert raised.value.code == "input_queue_overflow"
    inbox.put_command(CommitAudio())
    inbox.put_command(CommitAudio())
    with pytest.raises(SessionFatal):
        inbox.put_command(CommitAudio())
    inbox.put_signal(SpeechFinished(1))  # signals are not charged
    assert isinstance(await inbox.get(), AppendAudio)
    inbox.put_command(AppendAudio(b"\x00" * 4800))  # room again once consumed
    assert await inbox.get() == CommitAudio()
    inbox.put_command(CommitAudio())


async def test_inbox_preserves_arrival_order() -> None:
    inbox = Inbox(max_audio_ms=1000, max_commands=10)
    sequence = [AppendAudio(b"\x00\x00"), CommitAudio("a"), SpeechFinished(3), CommitAudio("b")]
    for message in sequence:
        if isinstance(message, SpeechFinished):
            inbox.put_signal(message)
        else:
            inbox.put_command(message)
    assert [await inbox.get() for _ in sequence] == sequence


async def test_outbox_backpressure_and_overflow() -> None:
    outbox = Outbox(limit=4)
    outbox.put(Outbound("a", None))
    outbox.put(Outbound("b", None))
    waiter = asyncio.create_task(outbox.wait_for_room())
    await asyncio.sleep(0)
    assert not waiter.done()  # at the low-water mark: producers wait
    assert (await outbox.get()).payload == "a"
    await asyncio.wait_for(waiter, 1)
    outbox.put(Outbound("c", None))
    outbox.put(Outbound("d", None))
    outbox.put(Outbound("e", None))
    with pytest.raises(SessionFatal) as raised:
        outbox.put(Outbound("f", None))
    assert (raised.value.code, raised.value.close_code) == ("slow_client", 1008)
    assert [entry.payload for entry in outbox.drain()] == ["b", "c", "d", "e"]


def test_g711_formats_merge_and_input_and_output_differ() -> None:
    patched = DEFAULTS.apply(SessionPatch(input_format=PCMU, output_format=PCMA), DEFAULTS)
    assert (patched.input_format, patched.output_format) == (PCMU, PCMA)
    kept = patched.apply(SessionPatch(instructions="x"), DEFAULTS)
    assert (kept.input_format, kept.output_format) == (PCMU, PCMA)
    back = kept.apply(SessionPatch(output_format=PCM), DEFAULTS)
    assert (back.input_format, back.output_format) == (PCMU, PCM)


def test_ledger_counts_g711_at_8_khz_one_byte_a_sample() -> None:
    ledger = OutputLedger()
    ledger.record(Delivery(1, "item_1", 0, "audio", 800, audio_format=PCMU))
    ledger.record(Delivery(1, "item_1", 0, "audio", 812, audio_format=PCMU))
    ledger.record(Delivery(2, "item_2", 0, "audio", 4800))  # 24 kHz PCM
    assert ledger.sent_ms("item_1") == 201  # 1612 bytes: 201.5 ms, rounded down
    assert ledger.sent_ms("item_2") == 100


def test_g711_segments_count_their_own_milliseconds() -> None:
    spoken = [
        SpokenSegment("你好。", "你好。", 0, 600 * 8, audio_format=PCMA),
        SpokenSegment("我是小王，", "我是小王，", 600, 1000 * 8, audio_format=PCMA),
    ]
    assert [segment.audio_ms for segment in spoken] == [600, 1000]
    assert heard_text(spoken, 300) == heard_text(SPOKEN[:2], 300) == "你"
    assert heard_text(spoken, 1100) == "你好。我是"
    message = AssistantMessage("item_1", "audio", segments=spoken, audio_format=PCMA)
    assert message.audio_ms == 1600


async def test_inbox_charges_g711_audio_by_its_duration() -> None:
    inbox = Inbox(max_audio_ms=100, max_commands=8, input_format=PCMU)
    inbox.put_command(AppendAudio(bytes(800)))  # 100 ms at 8 kHz
    with pytest.raises(SessionFatal):
        inbox.put_command(AppendAudio(bytes(1)))
    assert isinstance(await inbox.get(), AppendAudio)
    # A queued update to 24 kHz PCM: appends after it are charged at 48 bytes a millisecond.
    inbox.put_command(UpdateSession(SessionPatch(input_format=PCM)))
    inbox.put_command(AppendAudio(bytes(4800)))
    with pytest.raises(SessionFatal):
        inbox.put_command(AppendAudio(bytes(2)))
    assert isinstance(await inbox.get(), UpdateSession)
    assert isinstance(await inbox.get(), AppendAudio)
    # The actor rejected that update (the voice lock): back to the session's format.
    inbox.input_format_settled(PCMU)
    inbox.put_command(AppendAudio(bytes(800)))
    with pytest.raises(SessionFatal):
        inbox.put_command(AppendAudio(bytes(1)))


async def test_inbox_keeps_the_latest_queued_format_until_every_update_is_handled() -> None:
    inbox = Inbox(max_audio_ms=100, max_commands=8)
    inbox.put_command(UpdateSession(SessionPatch(input_format=PCMU)))
    inbox.put_command(UpdateSession(SessionPatch(input_format=PCMA)))
    inbox.input_format_settled(PCMU)  # the first is handled; the second is still queued
    inbox.put_command(AppendAudio(bytes(800)))  # charged as G.711: 100 ms
    with pytest.raises(SessionFatal):
        inbox.put_command(AppendAudio(bytes(1)))

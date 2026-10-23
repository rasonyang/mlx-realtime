from __future__ import annotations

import pytest

from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.input_buffer import InputBuffer
from tests.support import tone

CANONICAL_BYTES_PER_MS = 32


def test_append_reports_where_canonical_audio_starts() -> None:
    buffer = InputBuffer(max_ms=60_000)
    total = 0
    for _ in range(10):
        start, canonical = buffer.append(tone(20))
        assert start == total
        total += len(canonical) // 2
    assert buffer.end_ms == total // 16
    assert 170 <= buffer.end_ms <= 200  # the resampler holds back up to ~20 ms


def test_commit_flushes_the_resampler_and_covers_everything() -> None:
    buffer = InputBuffer(max_ms=60_000)
    for _ in range(25):
        buffer.append(tone(20))
    audio = buffer.commit()
    assert audio is not None
    assert (audio.start_ms, audio.end_ms, audio.audio_ms) == (0, 500, 500)
    assert len(audio.pcm) == 500 * CANONICAL_BYTES_PER_MS
    assert buffer.commit() is None  # empty now
    buffer.append(tone(100))
    second = buffer.commit()
    assert second is not None
    assert (second.start_ms, second.end_ms) == (500, 600)


def test_commit_span_keeps_later_audio() -> None:
    buffer = InputBuffer(max_ms=60_000)
    buffer.append(tone(1000))
    audio = buffer.commit_span(200, 700)
    assert (audio.start_ms, audio.end_ms, len(audio.pcm)) == (
        200,
        700,
        500 * CANONICAL_BYTES_PER_MS,
    )
    assert buffer.start_ms == 700
    clamped = buffer.commit_span(0, 10_000)
    assert (clamped.start_ms, clamped.end_ms) == (700, buffer.end_ms)


def test_clear_drops_audio_but_keeps_the_timeline() -> None:
    buffer = InputBuffer(max_ms=60_000)
    buffer.append(tone(300))
    buffer.clear()
    assert buffer.commit() is None
    start, _ = buffer.append(tone(300))
    assert start == 300 * 16  # the dropped audio, resampler tail included, still counts
    audio = buffer.commit()
    assert audio is not None
    assert (audio.start_ms, audio.end_ms) == (300, 600)


def test_idle_trim_and_slice() -> None:
    buffer = InputBuffer(max_ms=60_000)
    buffer.append(tone(1000))
    buffer.trim_before(600)
    assert buffer.start_ms == 600
    assert len(buffer.slice(500, 700)) == 100 * CANONICAL_BYTES_PER_MS  # clamped to what is held
    buffer.trim_before(100)  # never moves backwards
    assert buffer.start_ms == 600


def test_size_cap_is_fatal() -> None:
    buffer = InputBuffer(max_ms=1000)
    buffer.append(tone(900))
    with pytest.raises(SessionFatal) as raised:
        buffer.append(tone(200))
    assert raised.value.code == "input_audio_buffer_overflow"


def test_the_echo_cancellers_track_replaces_the_input_where_it_has_run() -> None:
    buffer = InputBuffer(max_ms=60_000, processed=True, asr_processed=True)
    start, canonical = buffer.append(tone(1000))
    assert start == 0
    processed = bytes(len(canonical) // 2)  # the canceller's output for the first half
    buffer.apply_processed(0, processed)
    halfway = len(processed) // 2  # samples
    assert buffer.slice(0, halfway // 16) == processed[: halfway // 16 * 32]  # Smart Turn's view
    assert buffer.slice(0, 1000)[len(processed) :] == canonical[len(processed) :]
    span = buffer.commit_span(0, 400)
    assert span.pcm == processed[: 400 * CANONICAL_BYTES_PER_MS]  # the ASR gets the output
    buffer.apply_processed(0, b"\x01\x00" * 6400)  # positions already committed: ignored
    assert buffer.slice(400, 401) == processed[400 * 32 : 401 * 32]


def test_asr_input_raw_commits_the_input_as_received() -> None:
    buffer = InputBuffer(max_ms=60_000, processed=True, asr_processed=False)
    _, canonical = buffer.append(tone(500))
    buffer.apply_processed(0, bytes(len(canonical)))
    assert buffer.slice(0, 100) == bytes(100 * CANONICAL_BYTES_PER_MS)  # detectors: processed
    audio = buffer.commit()
    assert audio is not None
    assert audio.pcm.startswith(canonical)  # the ASR: raw (plus the resampler's tail)


def test_without_the_canceller_there_is_one_track() -> None:
    buffer = InputBuffer(max_ms=60_000)
    _, canonical = buffer.append(tone(200))
    buffer.apply_processed(0, bytes(len(canonical)))  # no processed track: nothing changes
    assert buffer.slice(0, 100) == canonical[: 100 * CANONICAL_BYTES_PER_MS]

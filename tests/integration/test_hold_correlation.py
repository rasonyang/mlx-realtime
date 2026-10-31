"""The correlation veto of a held turn (``barge_in.confirm_echo_psr``): while AEC3 is in its
no-echo bypass, the raw input over the held speech is correlated (GCC-PHAT) with the audio we
played. A leak of our own words has a sharp peak, whatever the ASR made of it, so a release waits
for the turn's end and a strong peak drops the turn there; a real caller has none.

Through the real canceller on the virtual clock of ``test_hold_while_playing.py``. The answer is
seeded noise (a broadband stand-in for speech), and the input is built by the test: silence, with
a delayed copy of that noise (a leak), independent noise (a caller), or a steady echo of the
answer all along (a line on which AEC3 is not bypassed).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mlx_realtime.providers.base import Lang, PCMChunk, SpeechTiming
from mlx_realtime.providers.mock import _MockBackend
from mlx_realtime.server.gateway import Gateway
from tests.integration.test_echo_rules import drain, types
from tests.integration.test_hold_while_playing import (
    CALLER,
    ECHO,
    HeardASR,
    build,
    count,
    of_type,
    start_answer,
)
from tests.support import Client, ManualClock, running

RATE = 24_000
STEP_MS = 20
SEED = 7
NOISE = np.random.default_rng(SEED).normal(0.0, 3000.0, 30 * RATE)
OTHER = np.random.default_rng(SEED + 1).normal(0.0, 3000.0, 30 * RATE)
SPEECH = (5500, 6300)  # the held turn; the answer plays until about 7.4 s


class NoiseTTS(_MockBackend):
    """The answer: 100 ms of seeded noise per character, one continuous stream."""

    def __init__(self) -> None:
        super().__init__()
        self._cursor = 0

    async def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        total = sum(1 for char in text if not char.isspace()) * 100 * RATE // 1000
        chunk = 100 * RATE // 1000
        for start in range(0, total, chunk):
            await asyncio.sleep(0)
            first = self._cursor + start
            wave = NOISE[first : first + min(chunk, total - start)]
            yield PCMChunk(wave.astype("<i2").tobytes(), RATE)
        self._cursor += total


def pcm(wave: np.ndarray) -> str:
    return base64.b64encode(np.clip(wave, -32768, 32767).astype("<i2").tobytes()).decode()


async def stream_input(
    client: Client, clock: ManualClock, start_ms: int, end_ms: int, source: np.ndarray | None
) -> None:
    """Input from ``start_ms`` to ``end_ms``: ``source`` is the wave per 24 kHz sample of the
    input position (None: silence)."""
    for at in range(start_ms, end_ms, STEP_MS):
        clock.advance(STEP_MS / 1000)
        first = at * RATE // 1000
        wave = np.zeros(STEP_MS * RATE // 1000) if source is None else source[first : first + 480]
        await client.send({"type": "input_audio_buffer.append", "audio": pcm(wave)})
        await asyncio.sleep(0.002)


def delayed(delay_ms: int, gain: float, lo_ms: int, hi_ms: int) -> np.ndarray:
    """Our answer, ``delay_ms`` late and ``gain`` loud, only between ``lo_ms`` and ``hi_ms``."""
    out = np.zeros(len(NOISE))
    shift = (delay_ms + 200) * RATE // 1000  # the answer plays from about 200 ms of input
    lo, hi = lo_ms * RATE // 1000, hi_ms * RATE // 1000
    out[lo:hi] = gain * NOISE[lo - shift : hi - shift]
    return out


def independent(gain: float, lo_ms: int, hi_ms: int) -> np.ndarray:
    out = np.zeros(len(OTHER))
    lo, hi = lo_ms * RATE // 1000, hi_ms * RATE // 1000
    out[lo:hi] = gain * OTHER[lo:hi]
    return out


async def call(
    tmp_path: Path,
    source: np.ndarray,
    rules: Mapping[str, object] | None = None,
    *,
    text: str = CALLER,
) -> tuple[Client, Gateway, list[dict[str, Any]]]:
    clock, asr = ManualClock(), HeardASR(text)
    gateway = build(
        asr,
        [SPEECH],
        clock,
        rules,
        tts=NoiseTTS(),
        aec={"enabled": True, "capture_dir": str(tmp_path)},
    )
    async with running(gateway) as address, Client.connect(address) as client:
        await start_answer(client, clock)
        await stream_input(client, clock, 200, 9500, source)
        await drain(client)
    await asyncio.sleep(1.0)  # the capture's writer thread
    (directory,) = list(tmp_path.glob("*"))  # noqa: ASYNC240
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    return client, gateway, events


def checks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["kind"] == "hold_check"]


async def test_a_leak_the_asr_calls_a_caller_stays_held_and_is_dropped_at_the_end(
    tmp_path: Path,
) -> None:
    leak = delayed(400, 0.3, SPEECH[0] - 100, SPEECH[1] + 200)
    client, gateway, events = await call(tmp_path, leak)
    assert types(client).count("input_audio_buffer.speech_started") == 0
    assert "input_audio_buffer.speech_stopped" not in types(client)
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    first = checks(events)[0]
    assert first["psr"] is not None
    assert first["psr"] >= 15
    assert first["verdict"] == "echo_signal"  # the ASR said caller; the signal says it is ours
    (drop,) = [event for event in events if event["kind"] == "hold_drop"]
    assert drop["reason"] == "echo_signal"


async def test_a_correlation_between_the_thresholds_is_released_at_the_turns_end(
    tmp_path: Path,
) -> None:
    leak = delayed(400, 0.3, SPEECH[0] - 100, SPEECH[1] + 200)
    rules = {"confirm_echo_drop_psr": 1e6}  # never on the correlation alone
    client, _, events = await call(tmp_path, leak, rules)
    assert checks(events)[0]["verdict"] == "echo_signal"
    (started,) = of_type(client, "input_audio_buffer.speech_started")
    assert started["audio_start_ms"] > 0
    kinds = types(client)
    assert kinds.index("input_audio_buffer.speech_started") < kinds.index(
        "input_audio_buffer.speech_stopped"
    )
    release = [event for event in events if event["kind"] == "hold_release"]
    assert [event["reason"] for event in release] == ["turn_end"]


async def test_a_real_caller_has_no_correlation_and_is_released_at_once(tmp_path: Path) -> None:
    client, gateway, events = await call(tmp_path, independent(0.3, SPEECH[0] - 100, SPEECH[1]))
    (started,) = of_type(client, "input_audio_buffer.speech_started")
    assert started["audio_start_ms"] > 0
    first = checks(events)[0]
    assert first["psr"] is not None
    assert first["psr"] < 10
    assert first["verdict"] == "caller"
    assert (
        "input_audio_buffer.speech_stopped"
        not in types(client)[: types(client).index("input_audio_buffer.speech_started")]
    )
    assert count(gateway, "mlxrt_held_turns_released_total") == 1


async def test_no_veto_when_the_canceller_is_not_bypassed(tmp_path: Path) -> None:
    echo = delayed(400, 0.3, 700, 9000)  # a steady echo all along: AEC3 finds it
    caller = independent(0.3, SPEECH[0] - 100, SPEECH[1])
    client, gateway, events = await call(tmp_path, echo + caller)
    first = checks(events)[0]
    assert first["psr"] is None
    assert first["psr_why"] == "no_bypass"
    assert of_type(client, "input_audio_buffer.speech_started")
    assert count(gateway, "mlxrt_held_turns_released_total") == 1


@pytest.mark.parametrize("why", ["disabled"])
async def test_no_veto_when_it_is_switched_off(tmp_path: Path, why: str) -> None:
    leak = delayed(400, 0.3, SPEECH[0] - 100, SPEECH[1] + 200)
    client, _, events = await call(tmp_path, leak, {"confirm_echo_psr": None})
    first = checks(events)[0]
    assert first["psr"] is None
    assert first["psr_why"] == why
    assert of_type(client, "input_audio_buffer.speech_started")  # the ASR's word, as before


async def test_our_words_over_an_independent_input_are_dropped_by_the_transcript(
    tmp_path: Path,
) -> None:
    """On a live line our echo correlated as weakly as a caller (issue #9): a low correlation
    does not release a turn whose transcript is ours in long runs."""
    caller = independent(0.3, SPEECH[0] - 100, SPEECH[1])
    client, gateway, events = await call(tmp_path, caller, text=ECHO)
    final = checks(events)[-1]
    assert final["check"] == "final"
    assert final["psr"] is not None
    assert final["psr"] < 15
    assert final["verdict"] == "echo"
    assert not of_type(client, "input_audio_buffer.speech_started")
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    assert types(client).count("response.created") == 1


async def test_a_paraphrase_of_our_words_is_a_caller_whatever_the_correlation(
    tmp_path: Path,
) -> None:
    """The case a low correlation was once trusted for: the caller reuses our words, but in
    short fragments (every unit ours, in order, none three in a row)."""
    caller = independent(0.3, SPEECH[0] - 100, SPEECH[1])
    client, gateway, events = await call(tmp_path, caller, text="很很的事越好")
    assert checks(events)[-1]["verdict"] == "caller"
    assert of_type(client, "input_audio_buffer.speech_started")
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 0
    assert types(client).count("response.created") == 2  # the turn was answered


async def test_our_words_over_a_delayed_copy_of_our_audio_are_still_dropped(
    tmp_path: Path,
) -> None:
    leak = delayed(400, 0.3, SPEECH[0] - 100, SPEECH[1] + 200)
    client, gateway, events = await call(tmp_path, leak, text=ECHO)
    assert checks(events)[-1]["psr"] >= 15
    assert not of_type(client, "input_audio_buffer.speech_started")
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1
    assert types(client).count("response.created") == 1


async def test_our_words_are_dropped_by_the_transcript_rule_when_the_canceller_is_not_bypassed(
    tmp_path: Path,
) -> None:
    echo = delayed(400, 0.3, 700, 9000)  # a steady echo all along: AEC3 finds it
    caller = independent(0.3, SPEECH[0] - 100, SPEECH[1])
    client, gateway, events = await call(tmp_path, echo + caller, text=ECHO)
    assert checks(events)[-1]["psr_why"] == "no_bypass"
    assert not of_type(client, "input_audio_buffer.speech_started")
    assert count(gateway, "mlxrt_held_turns_dropped_total", reason="echo") == 1

"""Echo end to end (M4, D17): the gateway's own speech comes back on a simulated AICC call
(``tests/phone.py``: AICC's downsampler and playout, G.711, a synthetic echo path, AICC's
upsampler, 20 ms appends in real time), through the real canceller and Silero VAD.

- With the M4 defaults (the canceller, the onset guard, 400 ms while playing), the echo of a long
  answer never interrupts it, and a real caller talking over it later still does. The answer is
  paced two seconds ahead of real time, so the server is still sending it when the caller
  speaks, and cancels it.
- With the canceller and the rules off, the same echo interrupts the answer: the simulated
  echo is strong enough to matter.
- M7: the same call in G.711 passthrough (AICC with ``AcceptsG711: true``): μ-law or A-law both
  ways, 160-byte frames; the canceller's reference is our G.711 output decoded and resampled to
  16 kHz, as the caller's audio is.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager

import pytest

from mlx_realtime.config import validate_config
from mlx_realtime.providers.base import Lang, PCMChunk, Readiness, SpeechTiming
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR
from mlx_realtime.providers.registry import Providers
from mlx_realtime.server.gateway import Gateway
from tests.phone import Codec, SimulatedCall
from tests.speech import EN, SILERO, ZH, clip, needs_models, to_wire
from tests.support import Client, running

pytestmark = needs_models

SAY = "Say exactly this, word for word, and add nothing else:\n"
LINE = "但鸟类还是有很多地方跟恐龙相似。它们的骨骼也很像。科学家们一直在研究这个问题。"
CALLER = "Many people don't think about them as dinosaurs."
AICC_SESSION = {  # what AICC's gateway profile sends (spikes/aicc/traces)
    "audio": {
        "input": {
            "format": {"rate": 24000, "type": "audio/pcm"},
            "turn_detection": {"silence_duration_ms": 500, "type": "server_vad"},
        },
        "output": {"format": {"rate": 24000, "type": "audio/pcm"}},
    },
    "instructions": "You answer the phone for NovaNet.",
}


class ClipTTS:
    """A TTS that speaks: every segment is the Chinese fixture, so our echo is real speech."""

    def __init__(self, chunk_ms: int = 80) -> None:
        self._pcm = to_wire(clip(ZH))
        self._chunk = 24 * chunk_ms * 2
        self.requests: list[str] = []

    async def stream(
        self,
        text: str,
        voice: str,
        lang: Lang,
        *,
        speed: float = 1.0,
        timing: SpeechTiming | None = None,
    ) -> AsyncGenerator[PCMChunk]:
        self.requests.append(text)
        for start in range(0, len(self._pcm), self._chunk):
            await asyncio.sleep(0)
            yield PCMChunk(self._pcm[start : start + self._chunk], 24_000)

    async def warmup(self) -> None:
        return None

    async def ready(self) -> Readiness:
        return Readiness(ready=True, detail="clip")

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def echo_session(
    config: Mapping[str, object], session: Mapping[str, object] = AICC_SESSION
) -> AsyncIterator[Client]:
    """A gateway with the real Silero VAD (no Smart Turn), mock LLM and ASR, the clip TTS."""
    settings = validate_config(
        {
            "turn": {"vad": {"type": "silero", "model": str(SILERO)}},
            "limits": {"inbox_max_ms": 10_000},
            # Two seconds ahead of real time: the answer is still being sent while it plays.
            "output": {"max_lead_ms": 2000},
            **config,
        }
    )
    providers = Providers(llm=EchoLLM(), asr=ScriptedASR([CALLER]), tts=ClipTTS())
    gateway = Gateway(settings, providers)
    async with running(gateway) as address, Client.connect(address) as client:
        await client.until("conversation.created")
        await client.send({"type": "session.update", "session": {"type": "realtime", **session}})
        await client.until("session.updated")
        yield client


def g711_session(codec: Codec) -> dict[str, object]:
    audio_format = {"type": f"audio/{codec}"}
    return {
        **AICC_SESSION,
        "audio": {
            "input": {
                "format": audio_format,
                "turn_detection": {"silence_duration_ms": 500, "type": "server_vad"},
            },
            "output": {"format": audio_format},
        },
    }


async def until_playing(call: SimulatedCall, timeout_s: float = 5.0) -> float:
    async with asyncio.timeout(timeout_s):
        while not call.log.playback_starts:  # noqa: ASYNC110 (the call's ticker records it)
            await asyncio.sleep(0.01)
    return call.log.playback_starts[0]


@pytest.mark.parametrize("codec", ["pcm", "pcmu", "pcma"])
async def test_the_echo_never_interrupts_and_the_caller_still_does(
    codec: Codec, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="mlx_realtime.session.echo")
    session = AICC_SESSION if codec == "pcm" else g711_session(codec)
    async with (
        echo_session({}, session) as client,
        SimulatedCall(client, echo_db=10.0, codec=codec) as call,
    ):
        await asyncio.sleep(0.5)  # the call is up: room noise flows before anything is said
        await client.send({"type": "response.create", "response": {"instructions": SAY + LINE}})
        started = await until_playing(call)
        await asyncio.sleep(6.0)  # six seconds of our own speech coming back
        assert call.log.speech_started == [], "the echo interrupted the answer"
        assert "response.done" not in call.types()
        speaking = asyncio.get_running_loop().time()
        call.say(clip(EN))  # the caller talks over the answer
        async with asyncio.timeout(3.0):
            await call.done.wait()
        done = next(event for _, event in call.events if event["type"] == "response.done")
        assert done["response"]["status_details"] == {
            "type": "cancelled",
            "reason": "turn_detected",
        }
        (heard,) = call.log.speech_started
        assert speaking < heard < speaking + 1.5  # the caller's onset, not the echo's
        assert heard - started > 6.0
        types = call.types()
        assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
        call.done.clear()
        async with asyncio.timeout(10.0):  # the caller's turn is answered after it ends
            await call.done.wait()
    replies = [event for _, event in call.events if event["type"] == "response.done"]
    assert replies[-1]["response"]["status"] == "completed"
    expected = (
        {"type": "audio/pcm", "rate": 24000} if codec == "pcm" else {"type": f"audio/{codec}"}
    )
    assert replies[-1]["response"]["audio"]["output"]["format"] == expected
    (summary,) = [record for record in caplog.records if record.getMessage() == "echo summary"]
    erle = summary.__dict__["erle_db"]
    assert erle is not None
    assert erle >= 10, erle  # the canceller, not the barge-in rules alone, kept the echo out
    assert summary.__dict__["barge_ins"] == 1
    # G7 dual path: the ASR path's Speex reference was aligned from the guard's estimates a few
    # seconds into the answer (its ERLE, a median that the caller's double talk pulls down, is
    # measured offline: spikes/g7dual), and its filter never diverged.
    keys = ("speex", "anchor", "realign", "shift", "tick")
    fields = {key: value for key, value in summary.__dict__.items() if key.startswith(keys)}
    print(f"{codec}: AEC3 ERLE {erle} dB; {fields}")
    aligned = summary.__dict__["speex_aligned_at_ms"]
    assert aligned is not None
    assert aligned <= 5000, fields  # input ms: 0.5 s of room noise, then the answer plays
    assert summary.__dict__["speex_erle_db"] is not None
    assert summary.__dict__["speex_resets"] == 0


async def test_without_the_defences_the_echo_interrupts() -> None:
    off = {
        "aec": {"enabled": False},
        "barge_in": {
            "echo_guard_ms": 0,
            "min_speech_ms_while_playing": None,
            "echo_turn_suppression": False,
        },
    }
    async with echo_session(off) as client, SimulatedCall(client, echo_db=10.0) as call:
        await asyncio.sleep(0.5)
        await client.send({"type": "response.create", "response": {"instructions": SAY + LINE}})
        await until_playing(call)
        async with asyncio.timeout(6.0):
            await call.done.wait()
    done = next(event for _, event in call.events if event["type"] == "response.done")
    assert done["response"]["status_details"] == {"type": "cancelled", "reason": "turn_detected"}

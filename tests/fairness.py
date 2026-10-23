"""Multi-session TTS fairness (M6, DECISIONS 2026-10-23): several sessions share the one serialized
TTS stream, and AICC's client rules are measured on the wire.

The TTS is the real provider (``mlx_audio_speech``, one stream in flight) against the fake
``mlx_audio.server``, at M0 S3's pace: ``capped`` (first chunk 110 ms, then one 160 ms chunk per
90 ms: RTF 0.56, Low Power Mode, the pace the M3 tests use) or ``uncapped`` (76 ms, RTF 0.33,
High Power). It speaks ``MS_PER_CHAR`` of audio per character. The LLM is the echoing mock
(first token 150 ms, then 40 tokens/s), so each reply is the caller's text, in segments. Each
session plays ``turns`` turns: a text item, ``response.create``, the whole reply, then
``think_s`` of the caller talking. Session ``i`` starts at ``i × stagger_s``.

Per scenario: the largest gap between a response's audio deltas and how many reach 2 s (AICC's
watchdog), the first audio delta after ``response.created`` (AICC: within 3 s), and the
responses that failed (the TTS queue timeout: ``backend_busy``) or ended without audio.

The before/after table (real time, no GPU; about 6 minutes with the defaults):

    uv run --locked python -m tests.fairness [--sessions 2 3 4 8] [--turns 2] [--pace capped]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from mlx_realtime.config import MlxAudioSpeechConfig, validate_config
from mlx_realtime.providers.mock import EchoLLM, ScriptedASR
from mlx_realtime.providers.registry import Providers, build_speech
from mlx_realtime.providers.voices import VoiceBook
from mlx_realtime.server.gateway import Gateway
from tests.fakes.mlx_audio import FakeMlxAudio, serve_asgi
from tests.support import Client, Event, running, with_echo_off

MS_PER_CHAR = 220  # Mandarin at about 4.5 characters per second
PACES = {  # (first chunk, each later chunk of 160 ms), M0 S3 at streaming_interval 0.16
    "capped": (0.11, 0.09),
    "uncapped": (0.076, 0.053),
}
REPLY = "您好，这里是客服中心。您的订单已经发货了，明天送到。"  # 2 segments, ~5.3 s of audio
GAP_RULE_S, FIRST_AUDIO_RULE_S = 2.0, 3.0


@dataclass(frozen=True)
class Policy:
    name: str
    scheduling: str  # edf | fifo
    reserve: bool
    grace_ms: int = 0
    queue_timeout_ms: int = 2000


BEFORE = Policy("before: fifo", "fifo", reserve=False)
RESERVE_ONLY = Policy("fifo + reserve", "fifo", reserve=True)
EDF_ONLY = Policy("edf", "edf", reserve=False)
AFTER = Policy("after: edf + reserve", "edf", reserve=True)
POLICIES = {"before": BEFORE, "reserve": RESERVE_ONLY, "edf": EDF_ONLY, "after": AFTER}


@dataclass
class Result:
    policy: str
    sessions: int
    responses: int = 0
    failed: int = 0
    without_audio: int = 0
    max_gap_s: float = 0.0
    gaps_over_rule: int = 0
    first_audio_max_s: float = 0.0
    first_audio_p50_s: float = 0.0
    first_audio_over_rule: int = 0
    errors: dict[str, int] = field(default_factory=dict[str, int])


async def run_scenario(
    sessions: int,
    policy: Policy,
    *,
    turns: int = 2,
    think_s: float = 3.0,
    stagger_s: float = 0.7,
    reply: str = REPLY,
    pace: str = "capped",
) -> Result:
    speech = FakeMlxAudio()
    async with serve_asgi(speech) as tts_url:
        gateway = _gateway(tts_url, policy, sessions)
        async with running(gateway) as address:
            speech.first_chunk_s, speech.chunk_s = PACES[pace]  # after the warmup
            speech.ms_per_char = MS_PER_CHAR
            timelines = await asyncio.gather(
                *(
                    _session(address, index * stagger_s, turns, think_s, reply)
                    for index in range(sessions)
                )
            )
    return measure(policy.name, sessions, timelines)


def _gateway(tts_url: str, policy: Policy, sessions: int) -> Gateway:
    tts = {
        "type": "mlx_audio_speech",
        "base_url": tts_url,
        "scheduling": policy.scheduling,
        "first_segment_grace_ms": policy.grace_ms,
        "queue_timeout_ms": policy.queue_timeout_ms,
        "fallback": {"enabled": False, "model": "unused"},
    }
    config = validate_config(
        with_echo_off(
            {
                "gateway": {"max_sessions": max(sessions, 1)},
                "voices": {"default": "vivian", "map": {"vivian": {"speaker": "vivian"}}},
                "output": {"reserve": {"enabled": policy.reserve}},
                "providers": {"tts": tts},
            }
        )
    )
    tts_config = config.providers.tts
    assert isinstance(tts_config, MlxAudioSpeechConfig)
    providers = Providers(
        llm=EchoLLM(first_token_delay_s=0.15, token_delay_s=0.025),
        asr=ScriptedASR(["你好。"]),
        tts=build_speech(tts_config, VoiceBook(config.voices)),
    )
    return Gateway(config, providers)


type Timeline = Sequence[tuple[float, str, Event]]


async def _session(
    address: str, delay_s: float, turns: int, think_s: float, reply: str
) -> Timeline:
    await asyncio.sleep(delay_s)
    async with Client.connect(address) as client:
        await client.start({"audio": {"input": {"turn_detection": None}}})
        for turn in range(turns):
            if turn:
                await asyncio.sleep(think_s)
            await client.say(reply)
            await client.send({"type": "response.create"})
            await client.until("response.done", wait_s=120)
        return client.timeline


def measure(policy: str, sessions: int, timelines: Sequence[Timeline]) -> Result:
    """AICC's rules over every session's recorded timeline."""
    result = Result(policy, sessions)
    gaps: list[float] = []
    firsts: list[float] = []
    for timeline in timelines:
        created: dict[str, float] = {}
        last: dict[str, float] = {}
        for at, direction, event in timeline:
            if direction != "server":
                continue
            kind = event["type"]
            if kind == "response.created":
                created[event["response"]["id"]] = at
            elif kind == "response.output_audio.delta":
                response_id = event["response_id"]
                if response_id in last:
                    gaps.append(at - last[response_id])
                else:
                    firsts.append(at - created[response_id])
                last[response_id] = at
            elif kind == "error":
                code = event["error"]["code"]
                result.errors[code] = result.errors.get(code, 0) + 1
            elif kind == "response.done":
                result.responses += 1
                result.failed += event["response"]["status"] != "completed"
                result.without_audio += event["response"]["id"] not in last
    result.max_gap_s = round(max(gaps, default=0.0), 3)
    result.gaps_over_rule = sum(1 for gap in gaps if gap >= GAP_RULE_S)
    result.first_audio_max_s = round(max(firsts, default=0.0), 3)
    result.first_audio_p50_s = round(statistics.median(firsts), 3) if firsts else 0.0
    result.first_audio_over_rule = sum(1 for first in firsts if first > FIRST_AUDIO_RULE_S)
    return result


def markdown(results: Sequence[Result]) -> str:
    lines = [
        "| sessions | policy | responses: failed / no audio | max delta gap | gaps ≥ 2 s "
        "| first audio p50 / max | first audio > 3 s |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.sessions} | {r.policy} | {r.responses}: {r.failed} / {r.without_audio} "
        f"| {r.max_gap_s:.2f} s | {r.gaps_over_rule} "
        f"| {r.first_audio_p50_s:.2f} / {r.first_audio_max_s:.2f} s | {r.first_audio_over_rule} |"
        for r in results
    ]
    return "\n".join(lines)


async def _main(arguments: argparse.Namespace) -> None:
    results: list[Result] = []
    for count in arguments.sessions:
        for name in arguments.policies:
            result = await run_scenario(
                count,
                POLICIES[name],
                turns=arguments.turns,
                stagger_s=arguments.stagger,
                pace=arguments.pace,
            )
            print(json.dumps(asdict(result), ensure_ascii=False), flush=True)
            results.append(result)
    print(markdown(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-session TTS fairness, before and after.")
    parser.add_argument("--sessions", type=int, nargs="+", default=[2, 3, 4, 8])
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--stagger", type=float, default=0.7)
    parser.add_argument("--pace", choices=sorted(PACES), default="capped")
    parser.add_argument(
        "--policies", nargs="+", choices=list(POLICIES), default=["before", "reserve", "after"]
    )
    asyncio.run(_main(parser.parse_args()))

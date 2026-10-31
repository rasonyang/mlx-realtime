"""Fresh ASR checks on a live capture, for the holds whose release the v7 rule defers to a check the
recording build never made (issue #10).

For each hold the script re-slices the capture's `asr.wav` (the ASR path's track, which is what
`InputBuffer.asr_slice` gives a check) the way the gateway does (the last `confirm_window_ms`
ending at the check's end, not before the hold's start), transcribes

- the recorded last partial check again (does re-slicing reproduce the recorded text?),
- `--next` successive partial checks, each `confirm_step_ms` further on,
- the whole turn (the final check, over the span of the capture's `transcript` event),

and walks the recorded checks followed by the fresh ones through `hold_replay_v7.walk7`. A fresh
check takes the recorded check's played text (the assistant text around the turn) and the
verdict `caller` when it has `confirm_min_new_units` units that are not ours (`hold` else); the
replay cannot tell whether the client played at that moment.

Starts the ASR backend only (`hold_backends`, stopped on exit) and prints markdown:

    uv run --locked python spikes/live_echo/hold_next_check.py \\
        spikes/live_echo/captures/live/20261030T040025725Z_sess_bnblfvhsgir6mt70 \\
        --items 56h0,ggyf [--next 3]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

import hold_replay_v7 as v7
from hold_backends import REPO, backends_running
from hold_lib import read_wav
from hold_replay_v5 import CONFIG, load, turns

from mlx_realtime.audio.pcm import CANONICAL_RATE
from mlx_realtime.config import load_config
from mlx_realtime.providers.registry import build_asr
from mlx_realtime.text.similarity import unmatched, units, unmatched_inside


def slice_ms(pcm: bytes, start_ms: int, end_ms: int) -> bytes:
    return pcm[start_ms * CANONICAL_RATE // 1000 * 2 : end_ms * CANONICAL_RATE // 1000 * 2]


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(REPO / "config.yaml", os.environ)
    asr = build_asr(config.providers.asr)
    pcm = read_wav(args.capture / "asr.wav")
    events = load(args.capture)
    spans = {e["item"]: (e["audio_start_ms"], e["audio_end_ms"]) for e in events if e["kind"] == "transcript"}
    step, window = CONFIG.confirm_step_ms, CONFIG.confirm_window_ms
    wanted = args.items.split(",")
    await asr.warmup() if hasattr(asr, "warmup") else None

    async def hear(start: int, end: int) -> str:
        transcript = await asr.transcribe(slice_ms(pcm, start, end))
        return transcript.text

    for turn in turns(args.capture):
        if turn["item"][-4:] not in wanted:
            continue
        checks = list(turn["checks"])
        last = checks[-1]
        played = last["played_text"] or ""
        start_ms = last["audio_start_ms"]
        interval = (checks[-1]["t"] - checks[0]["t"]) / max(len(checks) - 1, 1)
        print(f"\n## {turn['item'][-4:]} (recorded: {turn['recorded']}, {len(checks)} checks)\n")
        print("| check | input ms | recorded | fresh | unmatched | before the last unit |")
        print("|---|---|---|---|---|---|")
        again = await hear(last["audio_start_ms"], last["audio_end_ms"])
        print(
            f"| last recorded, again | {last['audio_start_ms']}-{last['audio_end_ms']} | "
            f"{last['text']} | {again} | {unmatched(again, played)} | "
            f"{unmatched_inside(again, played)} |"
        )
        end_ms, fresh = last["audio_end_ms"], []
        for n in range(1, args.next + 1):
            end_ms += step
            begin = max(start_ms, end_ms - window)
            text = await hear(begin, end_ms)
            new = unmatched(text, played)
            fresh.append(
                {
                    "check": "partial",
                    "verdict": "caller" if new >= CONFIG.confirm_min_new_units else "hold",
                    "text": text,
                    "played_text": played,
                    "psr": None,
                    "t": last["t"] + n * interval,
                    "units": len(units(text)),
                    "unmatched": new,
                    "audio_start_ms": begin,
                    "audio_end_ms": end_ms,
                }
            )
            print(
                f"| next {n} | {begin}-{end_ms} | - | {text} | {new} | "
                f"{unmatched_inside(text, played)} |"
            )
        if turn["item"] in spans:
            begin, end = spans[turn["item"]]
            text = await hear(begin, end)
            fresh.append(
                {
                    "check": "final",
                    "verdict": "hold",
                    "text": text,
                    "played_text": played,
                    "psr": None,
                    "t": last["t"] + (args.next + 1) * interval,
                    "units": len(units(text)),
                    "unmatched": unmatched(text, played),
                    "audio_start_ms": begin,
                    "audio_end_ms": end,
                }
            )
            print(f"| final | {begin}-{end} | - | {text} | {unmatched(text, played)} | |")
        walk: list[dict[str, Any]] = checks + fresh
        outcome, index, status = v7.walk7(walk)
        delay = "-" if index is None else f"{walk[index]['t'] - last['t']:.0f}"
        print(
            f"\nv7 walk: {outcome} at check {index} of {len(walk)} ({status}); "
            f"time after the recorded release check: {delay} ms "
            f"(recorded check interval {interval:.0f} ms)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--items", required=True, help="last 4 characters of the item ids")
    parser.add_argument("--next", type=int, default=3)
    args = parser.parse_args()
    with backends_running(Path(__file__).parent / "captures" / "hold" / "logs", only={"asr"}):
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

"""Self-test of ``--aicc-recording`` on a synthetic AICC stereo recording with known content.

    uv run --locked python selftest_recording.py <simulated capture dir> [--out results/recording_selftest.md]

The recording is built from a simulated capture: right = the bot as the switch sends it (the
capture's reference audio, 2.5 s into the recording), left = the caller as the switch receives it
(the bot through a 40 ms, ERL 12 dB path plus line noise). It then injects what the analysis must
flag: a 250 ms digital silence in the bot channel while an answer plays, and a 40 ms stall (the
audio after the first barge-in arrives 40 ms later). Checks: the switch-side delay (40 ms) and ERL
(12 dB) are recovered, the remainder is the end-to-end delay less 40 ms, both injected faults are
flagged with their times, and a clean recording is not flagged.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

import analyze

RATE = 8000
OFFSET_S = 2.5
SWITCH_DELAY_MS = 40.0
SWITCH_ERL_DB = 12.0
GAP_MS = 250
STALL_MS = 40
FAKE_FLUSH_S = 20.0  # a barge-in between the first and second answer (the capture's own is at its end)


def build(directory: Path, *, faults: bool) -> tuple[np.ndarray, dict, float]:
    ref, rate = analyze.read_wav(directory / "ref_aec3.wav")
    bot = ref[:, 0][:: rate // RATE] * 1.0  # a decimation is enough for a test signal
    events = analyze.load_events(directory)
    first_flush = FAKE_FLUSH_S + OFFSET_S
    right = np.concatenate((np.zeros(round(OFFSET_S * RATE)), bot))
    truth: dict = {}
    if faults:
        active = analyze.active_frames(analyze.to_analysis(ref[:, 0], rate))
        loud = next(f for f in range(24 * 50, 36 * 50) if active[f : f + 25].all())  # 0.5 s of speech
        gap_at = loud * 0.02 + 0.1
        gap = round((gap_at + OFFSET_S) * RATE)
        right[gap : gap + GAP_MS * RATE // 1000] = 0
        cut = round((first_flush + 1.0) * RATE)  # after the barge-in, before the next answer
        right = np.concatenate((right[:cut], np.zeros(STALL_MS * RATE // 1000), right[cut:]))
        truth = {"gap_recording_s": gap_at + OFFSET_S, "stall_after_s": first_flush}
    rng = np.random.default_rng(3)
    path = rng.standard_normal(40) * np.exp(-np.arange(40) / 8)
    path *= 10 ** (-SWITCH_ERL_DB / 20) / np.sqrt(np.sum(path**2))
    echo = np.convolve(right, path)[: len(right)]
    echo = np.concatenate((np.zeros(round(SWITCH_DELAY_MS * RATE / 1000)), echo))[: len(right)]
    left = echo + rng.standard_normal(len(right)) * 3.0
    stereo = np.stack((left, right), axis=1)
    # The path's gain on this speech (its ripples meet the speech spectrum), 300-3400 Hz.
    block = 4096
    psd = np.zeros(block // 2 + 1)
    for start in range(0, len(right) - block, block // 2):
        part = right[start : start + block]
        if np.mean(part**2) > 1e3:
            psd += np.abs(np.fft.rfft(part * np.hanning(block))) ** 2
    freqs = np.fft.rfftfreq(block, 1 / RATE)
    response = np.abs(np.exp(-2j * np.pi * np.outer(freqs / RATE, np.arange(len(path)))) @ path) ** 2
    band = (freqs >= 300) & (freqs <= 3400)
    expected_erl = float(-10 * np.log10(np.sum(psd[band] * response[band]) / np.sum(psd[band])))
    return np.clip(np.rint(stereo), -32768, 32767).astype("<i2"), truth, expected_erl


def write(path: Path, stereo: np.ndarray) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(stereo.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "recording_selftest.md")
    args = parser.parse_args()
    out = ["# AICC recording self-test", "", f"Synthetic stereo recording from `{args.capture.name}`: switch-side path {SWITCH_DELAY_MS:g} ms, ERL {SWITCH_ERL_DB:g} dB; faults injected: {GAP_MS} ms of silence in the bot channel while an answer plays (about 27 s into the capture), and a {STALL_MS} ms stall one second after a barge-in (a flush event added at {FAKE_FLUSH_S:g} s of the capture).", ""]
    for faults in (False, True):
        stereo, truth, expected_erl = build(args.capture, faults=faults)
        wav = args.capture / ("recording_faults.wav" if faults else "recording_clean.wav")
        write(wav, stereo)
        report = analyze.analyse_capture(args.capture, wav)
        rec = report["aicc_recording"]
        data, rate = analyze.read_wav(wav)
        reference = analyze.to_analysis(analyze.read_wav(args.capture / "ref_aec3.wav")[0][:, 0], 16_000)
        events = [*analyze.load_events(args.capture), {"t": FAKE_FLUSH_S * 1000, "kind": "flush"}]
        bot = analyze.bot_channel_flags(data[:, 1], rate, analyze.to_analysis(data[:, 1], rate), reference, events)
        out += [
            f"## {'With faults' if faults else 'Clean'}",
            "",
            f"- switch-side delay: {rec['switch_delay_ms']:.1f} ms (expected {SWITCH_DELAY_MS:g}); ERL {rec['erl_db']:.1f} dB ({expected_erl:.1f} on this speech; configured {SWITCH_ERL_DB:g}); dispersion {rec['dispersion_ms']:.1f} ms",
            f"- end-to-end after the model {rec['end_to_end_vs_model_ms']:.1f} ms = switch side + remainder {rec['remainder_ms']:.1f} ms",
            f"- bot channel aligned with our reference at {bot['alignment_lag_ms']:.1f} ms (built with {OFFSET_S * 1000:g})",
            f"- flags: {json.dumps([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()} for f in bot['flags']])}",
            f"- injected: {truth or 'none'}",
            "",
        ]
        print(out[-4])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out))


if __name__ == "__main__":
    main()

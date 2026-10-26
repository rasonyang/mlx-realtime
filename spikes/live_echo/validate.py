"""Compare analyze.py's report on simulated captures with their ground truth (simulate.py).

    uv run --locked python validate.py <capture dir>... [--out results/simulated.md]

Ground truth (``ground_truth.json`` beside each capture):

- ERL: the echo path's gain, as configured (G.168: raised to the model's minimum).
- Delay: the echo of audio the client plays at input time t appears at
  ``t + dominant_delay`` (bulk delay + the path's dominant tap + the downsampler's group delay).
  The playout model places the same audio at its own modelled start, so the analysis's delay
  "after the model" (lag + shift - lead) of answer i must equal
  ``dominant_delay + (true start of run i - modelled start of run i)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze

DELAY_TOLERANCE_MS = 2.0
ERL_TOLERANCE_DB = 1.5


def effective_erl_db(directory: Path, path: list[float]) -> float:
    """The ERL the echo path really has on this reference: the reference's 300-3400 Hz energy over
    the same band after AICC's 24 -> 8 kHz downsampler (a 33-tap Hamming-windowed sinc,
    tests/phone.py) and the echo path, weighted by the reference's own spectrum. The configured
    ERL is the path's energy gain over white noise; a speech spectrum meets a random path's
    ripples and weighs its gain differently by up to a dB or so."""
    data, rate = analyze.read_wav(directory / "ref_aec3.wav")
    ref = data[:, 0]
    taps = 33
    index = np.arange(taps) - (taps - 1) / 2
    cutoff = 0.5 / 3
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(index == 0, 2 * cutoff, np.sin(2 * np.pi * cutoff * index) / (np.pi * index))
    h *= 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(taps) / (taps - 1))
    h /= h.sum()
    block = 4096
    window = np.hanning(block)
    psd = np.zeros(block // 2 + 1)
    for start in range(0, len(ref) - block, block // 2):
        part = ref[start : start + block]
        if np.mean(part**2) > 10 ** (-50 / 10) * 32768.0**2:
            psd += np.abs(np.fft.rfft(part * window)) ** 2
    freqs = np.fft.rfftfreq(block, 1 / rate)
    down = np.abs(np.exp(-2j * np.pi * np.outer(freqs / 24_000, np.arange(taps))) @ h) ** 2
    line = np.abs(np.exp(-2j * np.pi * np.outer(freqs / 8_000, np.arange(len(path)))) @ np.asarray(path)) ** 2
    band = (freqs >= 300) & (freqs <= 3400)
    return float(-10 * np.log10(np.sum(psd[band] * down[band] * line[band]) / np.sum(psd[band])))


def validate(directory: Path) -> dict:
    truth = json.loads((directory / "ground_truth.json").read_text())
    report = analyze.analyse_capture(directory, None)
    (directory / "report.json").write_text(json.dumps(analyze.clean(report), indent=2) + "\n")
    (directory / "report.md").write_text(analyze.markdown(report) + "\n")
    events = analyze.load_events(directory)
    models = [e["t"] for e in events if e["kind"] == "play_start"]
    row: dict = {
        "scenario": truth["scenario"],
        "g168": truth.get("g168"),
        "class": report["class"],
        "truth": {k: v for k, v in truth.items() if k != "path"},
    }
    if truth["erl_db"] is None:
        bypass = report["bypass"]
        row.update(
            expected_class="none",
            ok=report["class"] == "none" and bypass["engaged"] >= 1 and not bypass["flapping"],
            bypass_engaged=bypass["engaged"],
            bypass_first_s=bypass["first_on_after_first_answer_s"],
            bypass_flapping=bypass["flapping"],
            self_interruptions=len(report["caller"]["self_interruptions"]),
            barge_ins=report["caller"]["barge_ins"],
        )
        return row
    answers = []
    report_lead = json.loads((directory / "meta.json").read_text())["aec"]["reference_lead_ms"]
    for index, entry in enumerate(report["delay"]["per_response"]):
        median = entry.get("median_vs_model_ms")
        if median is None or index >= len(truth["client_play_start_ms"]) or index >= len(models):
            continue
        offset = truth["client_play_start_ms"][index] - models[index]
        expected = truth["dominant_delay_ms"] + offset
        shift = analyze.shift_at(events, entry["start_ms"] + 1000)
        peak = report["segment_peaks_ms"].get(index)
        answers.append({
            "answer": index + 1,
            "filter_peak_vs_model_ms": None if peak is None else peak + shift - float(report_lead),
            "model_error_ms": offset,
            "expected_ms": expected,
            "measured_ms": median,
            "start_ms": entry.get("start_vs_model_ms"),
            "end_ms": entry.get("end_vs_model_ms"),
            "error_ms": median - expected,
        })
    # The model learns AICC's tick phase during the first answer and moves by up to a tick, so
    # that answer's modelled start is not where the rest of it sits: it is reported, not judged.
    errors = [abs(a["error_ms"]) for a in answers[1:]]
    expected_erl = effective_erl_db(directory, truth["path"])
    loss = expected_erl - truth["erl_db"]
    erl_error = report["erl_db"] - expected_erl if report.get("erl_db") is not None else None
    row.update(
        expected_class="echo",
        answers=answers,
        max_delay_error_ms=max(errors, default=None),
        erl_measured=report.get("erl_db"),
        erl_expected=expected_erl,
        effective_minus_nominal_db=loss,
        erl_error_db=erl_error,
        dispersion_ms=report.get("dispersion_ms"),
        aec3_erle=report["erle"].get("aec3"),
        speex_erle=report["erle"].get("asr"),
        speex_first_s=report["speex"]["first_alignment_after_first_answer_s"],
        gateway_vs_independent_ms=report["gateway_estimator"]["median_abs_diff_ms"],
        self_interruptions=len(report["caller"]["self_interruptions"]),
        barge_ins=report["caller"]["barge_ins"],
        drift_flag=report["delay"]["drift_flag"],
        ok=bool(
            report["class"] == "echo"
            and errors
            and max(errors) <= DELAY_TOLERANCE_MS
            and erl_error is not None
            and abs(erl_error) <= ERL_TOLERANCE_DB
        ),
    )
    return row


INJECTED = [  # (start s, duration s, lag ms, gain re the reference dB)
    (15.0, 2.0, 350.0, -12.0),
    (27.0, 1.5, 500.0, -30.0),
]


def injected_bursts(directory: Path) -> dict:
    """The no-echo capture (canceller bypassed) with leaks added to its input: delayed copies of
    the reference at known times, lags and levels, as a phone's AEC lets through. The burst
    detector must find them, with their lags, and nothing else."""
    events = analyze.load_events(directory)
    tracks = {}
    for name in ("input", "ref_aec3"):
        data, rate = analyze.read_wav(directory / f"{name}.wav")
        tracks[name] = data[:, 0]
    inp = tracks["input"].copy()
    ref16 = tracks["ref_aec3"]
    for start, duration, lag, gain in INJECTED:
        a, b = round(start * 16000), round((start + duration) * 16000)
        shift = round(lag * 16)
        inp[a:b] += ref16[a - shift : b - shift] * 10 ** (gain / 20)
    ref8 = analyze.to_analysis(ref16, 16000)
    in8 = analyze.to_analysis(inp, 16000)
    base = analyze.to_analysis(tracks["input"], 16000)
    floor = analyze.noise_floor(base, ~analyze.active_frames(ref8))
    responses = analyze.responses_of(events)
    found = analyze.detect_bursts(ref8, in8, events, floor, responses)
    return {"injected": INJECTED, "found": found["bursts"], "bypassed_s": found["bypassed_s"]}


def f(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(rows: list[dict]) -> str:
    out = [
        "# Validation on simulated echo",
        "",
        "analyze.py on captures of the real gateway (aec.capture_dir) in `tests/phone.py`'s "
        "simulated AICC call: 24 kHz PCM both ways, AICC's 24 -> 8 kHz downsampler, G.711 mu-law "
        "round trip, a synthetic or ITU-T G.168 echo path after a bulk delay, AICC's 8 -> 24 kHz "
        "uplink, three answers heard in silence and a fourth cut by a caller. Regenerate with "
        "`simulate.py` and `validate.py` (README).",
        "",
        f"Pass: delay within {DELAY_TOLERANCE_MS} ms and ERL within {ERL_TOLERANCE_DB} dB (echo lines), "
        "class `none` with the bypass engaged and not flapping (no-echo line). The first answer is reported but not judged: the playout model learns AICC's tick phase during it and moves by up to a tick, so its modelled start is not where the rest of the answer sits.",
        "",
        "## Summary",
        "",
        "| scenario | expected | measured | result |",
        "|---|---|---|---|",
    ]
    for row in rows:
        truth = row["truth"]
        if row["expected_class"] == "none":
            out.append(
                f"| {row['scenario']} | no echo, bypass engaged | class {row['class']}, bypass {row['bypass_engaged']}x "
                f"(first {f(row['bypass_first_s'], 2)} s after the first answer starts), flapping {row['bypass_flapping']} "
                f"| {'PASS' if row['ok'] else 'FAIL'} |"
            )
            continue
        worst = max(row["answers"], key=lambda a: abs(a["error_ms"]), default=None)
        out.append(
            f"| {row['scenario']} | ERL {f(truth['erl_db'])} dB (effective on this speech: {f(row['erl_expected'])} dB), delay {f(truth['bulk_delay_ms'], 0)} ms "
            f"(dominant tap {f(truth['dominant_delay_ms'], 2)} ms) | ERL {f(row['erl_measured'])} dB "
            f"(error {f(row['erl_error_db'])}), delay error up to {f(row['max_delay_error_ms'], 2)} ms over "
            f"answers 2-{len(row['answers'])}{'' if worst is None else ''} | {'PASS' if row['ok'] else 'FAIL'} |"
        )
    for row in rows:
        out += ["", f"## {row['scenario']}", ""]
        truth = row["truth"]
        if truth.get("g168"):
            out += [f"G.168 model {truth['g168']} (coefficients not committed), delay {truth['bulk_delay_ms']:g} ms before the path.", ""]
        if row["expected_class"] == "none":
            out += [
                f"- line class: {row['class']}; bypass engaged {row['bypass_engaged']}x, first {f(row['bypass_first_s'], 2)} s "
                f"after the first answer starts (expected about 1.5 s plus the first estimator window); flapping: {row['bypass_flapping']}",
                f"- self-interruptions: {row['self_interruptions']}",
                f"- caller barge-ins: {[(round(b['latency_ms']), round(b['excess_ms'])) for b in row['barge_ins']]} (latency, excess over the guard, ms)",
            ]
            continue
        out += ["| answer | model error (true - modelled start, ms) | expected delay after model (ms) | measured (ms) | error (ms) | start / end of answer (ms) | filter's dominant tap (ms) |",
                "|---|---|---|---|---|---|---|"]
        for a in row["answers"]:
            out.append(f"| {a['answer']} | {f(a['model_error_ms'])} | {f(a['expected_ms'], 2)} | {f(a['measured_ms'], 2)} | {f(a['error_ms'], 2)} | {f(a['start_ms'], 2)} / {f(a['end_ms'], 2)} | {f(a['filter_peak_vs_model_ms'], 2)} |")
        aec3, speex = row["aec3_erle"], row["speex_erle"]
        out += [
            "",
            f"- ERL: configured {f(truth['erl_db'])} dB on the downsampled audio, which is {f(row['erl_expected'])} dB when the path's ripples meet this speech's spectrum (and AICC's downsampler); measured {f(row['erl_measured'])} dB; dispersion (90 % of the filter's energy) {f(row['dispersion_ms'])} ms",
            f"- AEC3 ERLE: converges to 10 dB {f(aec3 and aec3['convergence_s'], 2)} s after the first answer starts, steady-state median {f(aec3 and aec3['steady_median_db'])} dB",
            f"- Speex ERLE: {f(speex and speex['convergence_s'], 2)} s, steady-state median {f(speex and speex['steady_median_db'])} dB; first aligned {f(row['speex_first_s'], 2)} s after the first answer starts",
            f"- the gateway's GCC-PHAT against the independent delay: median |difference| {f(row['gateway_vs_independent_ms'], 2)} ms",
            f"- drift flag: {row['drift_flag']}; self-interruptions: {row['self_interruptions']}",
            f"- caller barge-ins: {[(round(b['latency_ms']), round(b['excess_ms'])) for b in row['barge_ins']]} (latency from the estimated caller onset, excess over the guard, ms)",
        ]
    out.append("")
    return "\n".join(out)


def burst_markdown(result: dict) -> str:
    out = ["", "## Residual-echo burst detector on injected leaks", "",
           "The no-echo capture (canceller bypassed) with delayed copies of the reference added to its input; the detector must find these and nothing else.", "",
           "| injected at (s) | duration (s) | lag (ms) | level vs reference (dB) |", "|---|---|---|---|"]
    out += [f"| {s:g} | {d:g} | {lag:g} | {g:g} |" for s, d, lag, g in result["injected"]]
    out += ["", "| found at (s) | duration (s) | lag (ms) | level vs reference (dB) | over floor (dB) | position |", "|---|---|---|---|---|---|"]
    out += [f"| {b['start_ms'] / 1000:.2f} | {b['duration_s']:.2f} | {b['lag_ms']:.0f} | {b['vs_reference_db']:.1f} | {b['over_floor_db']:.1f} | {b['position']} |" for b in result["found"]]
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "simulated.md")
    args = parser.parse_args()
    rows = [validate(directory) for directory in args.captures]
    quiet = next((d for d in args.captures if 'no_echo' in str(d)), None)
    bursts = injected_bursts(quiet) if quiet else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = markdown(rows) + (burst_markdown(bursts) if bursts else "")
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(analyze.clean({"scenarios": rows, "injected_bursts": bursts}), indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""Can a correlation of the raw input against our own played reference tell a leaked echo burst
from a real caller? Offline, on the hold measurement's fix-condition captures.

Reads captures/hold/rows/*-fix.json (ground truth: burst spans for A/Am, injected clip spans for
B/Be/C) and each capture's input.wav, ref_aec3.wav, aec3.wav and events.jsonl. Writes
results/hold_gcc.md. Numpy only. Run: uv run --locked python spikes/live_echo/hold_gcc.py
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).parent
FIX = HERE / "captures" / "hold" / "fix"
ROWS = HERE / "captures" / "hold" / "rows"
OUT = HERE / "results" / "hold_gcc.md"
RATE = 16_000
LAG_MAX_MS = 1000
BAND = (200.0, 3600.0)
SIDE = 32  # +-2 ms excluded around the peak for the PSR sides (as the gateway: 16 @ 8 kHz)
SPANS_MS = (400, 700, 1000)
PSR_T = (4, 6, 8, 10, 12, 15)
NCC_T = (0.05, 0.08, 0.1, 0.15, 0.2, 0.3)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768


def bandpass(x: np.ndarray) -> np.ndarray:
    n = 1 << (len(x) - 1).bit_length()
    f = np.fft.rfftfreq(n, 1 / RATE)
    spec = np.fft.rfft(x, n) * ((f >= BAND[0]) & (f <= BAND[1]))
    return np.fft.irfft(spec, n)[: len(x)]


def span_stats(x: np.ndarray, r: np.ndarray, s: int, e: int) -> dict[str, float] | None:
    """PHAT peak (PSR, lag) and best normalised correlation of x[s:e] against r over lags 0..1000 ms.
    x and r are band-passed, sample aligned. None when the reference is silent or out of range."""
    lag_max = LAG_MAX_MS * RATE // 1000
    if s - lag_max < 0 or e > len(x) or e - s < 800:
        return None
    seg = x[s:e]
    ref = r[s - lag_max : e]  # index k in ref pairs with seg[0]; lag = lag_max - k
    if np.sqrt(np.mean(ref**2)) < 10 ** (-60 / 20):  # reference silent over the whole lag range
        return None
    n = 1 << (len(ref) + len(seg)).bit_length()
    rf, xf = np.fft.rfft(ref, n), np.fft.rfft(seg, n)
    cross = rf * np.conj(xf)
    f = np.fft.rfftfreq(n, 1 / RATE)
    phat = np.where((f >= BAND[0]) & (f <= BAND[1]), cross / (np.abs(cross) + 1e-12), 0)
    mag = np.abs(np.fft.irfft(phat, n)[: lag_max + 1])  # k = 0..lag_max
    peak = int(np.argmax(mag))
    sides = np.concatenate((mag[: max(peak - SIDE, 0)], mag[peak + SIDE + 1 :]))
    psr = float((mag[peak] - sides.mean()) / (sides.std() + 1e-12))
    raw = np.fft.irfft(cross, n)[: lag_max + 1]
    energy = np.concatenate(([0.0], np.cumsum(ref**2)))
    window = energy[len(seg) : len(seg) + lag_max + 1] - energy[: lag_max + 1]
    ncc = raw / (np.linalg.norm(seg) * np.sqrt(window) + 1e-12)
    k = int(np.argmax(ncc))
    return {"psr": psr, "lag_ms": (lag_max - peak) * 1000 / RATE, "ncc": float(ncc[k]),
            "ncc_lag_ms": (lag_max - k) * 1000 / RATE}


def bypass_at(events: list[dict], t_ms: float) -> bool | None:
    state = None
    for e in events:
        if e["kind"] == "bypass" and e["t"] <= t_ms:
            state = e["on"]
    return state


def load_trials() -> list[dict]:
    trials = []
    for path in sorted(ROWS.glob("*-fix.json")):
        for row in json.loads(path.read_text())["rows"]:
            sc = row["scenario"]
            if sc not in ("A", "Am", "B", "Be", "C") or not row.get("capture_dir"):
                continue
            if sc in ("A", "Am"):
                if row["done_before_burst"]:
                    continue  # nothing is playing: the hold does not apply
                b = row["burst"]
                onset, end, delay = b["start_ms"], b["end_ms"], row["delay_s"] * 1000
            else:
                onset = row["n_inject"] * 20 + row["onset_s"] * 1000
                end = row["n_inject"] * 20 + row["onset_s"] * 1000 + (row["duration_s"] - row["onset_s"]) * 1000
                delay = None
            trials.append({"scenario": sc, "id": row["id"], "dir": FIX / row["capture_dir"], "onset": onset,
                           "end": end, "delay": delay})
    return trials


def analyse(trials: list[dict]) -> None:
    cache: dict[Path, tuple] = {}
    for t in trials:
        if t["dir"] not in cache:
            ev = [json.loads(line) for line in (t["dir"] / "events.jsonl").read_text().splitlines()]
            cache[t["dir"]] = (bandpass(read_wav(t["dir"] / "input.wav")), bandpass(read_wav(t["dir"] / "ref_aec3.wav")),
                               bandpass(read_wav(t["dir"] / "aec3.wav")), ev)
            cache[t["dir"]] += ([e for e in ev if e["kind"] == "gcc"],)
        x, r, y, ev, gcc = cache[t["dir"]]
        t["bypass"] = bool(bypass_at(ev, t["onset"]))  # no event yet: the canceller is active
        t["pre_lag"] = next((e["lag_ms"] for e in reversed(gcc) if e["t"] <= t["onset"] and e["valid"]), None)
        t["gcc"] = [e for e in gcc if e["t"] - 2000 < t["end"] and e["t"] > t["onset"]]
        t["gcc_mostly"] = [e for e in t["gcc"] if min(e["t"], t["end"]) - max(e["t"] - 2000, t["onset"]) >= 1000]
        s0 = round(t["onset"] * RATE / 1000)
        for T in SPANS_MS:
            e1 = min(s0 + T * RATE // 1000, round(t["end"] * RATE / 1000))
            t[f"in{T}"] = span_stats(x, r, s0, e1)
            t[f"aec{T}"] = span_stats(y, r, s0, e1)
        t["full"] = span_stats(x, r, s0, round(t["end"] * RATE / 1000))  # whole speech span: the lag reference


def lag_ref(t: dict) -> float | None:
    """The lag a decision should find: the whole burst's own peak (A/Am; the nominal delay is not a
    file lag, the playout model's shift varies per burst), the pre-onset logged lag (steady echo)."""
    if t["scenario"] in ("A", "Am"):
        return t["full"]["lag_ms"] if t["full"] and t["full"]["psr"] >= 8 else None
    if t["scenario"] == "Be":
        return t["pre_lag"]
    return None


def q(v: list[float], p: float) -> float:
    return float(np.percentile(v, p)) if v else float("nan")


def dist(v: list[float]) -> str:
    return "n/a" if not v else f"{q(v, 10):.1f} / {q(v, 50):.1f} / {q(v, 90):.1f}"


def table(head: list[str], body: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body]) + "\n"


def main() -> None:
    trials = load_trials()
    analyse(trials)
    groups = [("A leak (default model)", "A", None), ("Am leak (mild model)", "Am", None),
              ("B caller, no echo line", "B", None), ("Be caller, steady-echo line", "Be", None),
              ("C short caller", "C", None)]
    out = ["# Hold: does a correlation of the input against our played reference separate leaks from callers?\n",
           "Fix-condition captures of `results/hold.md`; ground truth from the rows (burst span for A/Am, injected clip "
           "span for B/Be/C); input-timeline ms. A1 bursts (answer already finished, nothing playing) are excluded. "
           "The leak model is a synthetic assumption (`bursts.py`); real phone leaks may differ.\n",
           "Lag reference (A/Am): the PHAT peak over the whole burst (PSR >= 8); the burst's nominal delay is not a lag in the "
           "reference file (the playout model's shift differs per burst), so it cannot be used. Be: the last valid logged "
           "lag before the onset.\n"]

    out.append("## 1. Counts and canceller state at the onset\n")
    body = []
    for name, sc, _ in groups:
        ts = [t for t in trials if t["scenario"] == sc]
        body.append([name, str(len(ts)), str(sum(t["bypass"] is True for t in ts)), str(sum(t["bypass"] is False for t in ts))])
    out.append(table(["group", "n", "bypass on", "bypass off"], body))

    out.append("## 2. Gateway's logged `gcc` windows overlapping the speech span\n")
    out.append("A logged window is 2 s ending at `t` (logged 0.3 s later); `mostly` = at least 1 s of it is speech. "
               "PSR cells: p10 / p50 / p90.\n")
    body = []
    for name, sc, _ in groups:
        ts = [t for t in trials if t["scenario"] == sc]
        for label, key in (("overlapping", "gcc"), ("mostly speech", "gcc_mostly")):
            w = [e for t in ts for e in t[key]]
            valid = [e for e in w if e["valid"]]
            cons = ""
            if sc in ("A", "Am"):
                ok = [abs(e["lag_ms"] - lag_ref(t)) <= 5 for t in ts for e in t[key] if e["valid"] and lag_ref(t) is not None]
                cons = f"{sum(ok)}/{len(ok)}"
            elif sc == "Be":
                ok = [abs(e["lag_ms"] - lag_ref(t)) <= 5 for t in ts for e in t[key] if e["valid"] and lag_ref(t) is not None]
                cons = f"{sum(ok)}/{len(ok)}"
            body.append([name, label, str(len(w)), f"{len(valid)}", dist([e["psr"] for e in w]), cons or "-"])
    out.append(table(["group", "windows", "n", "valid (PSR>=8)", "PSR p10/p50/p90", "lag consistent"], body))
    firsts = {}
    for name, sc, _ in groups:
        ts = [t for t in trials if t["scenario"] == sc and t["gcc_mostly"]]
        firsts[name] = median([min(e["t"] for e in t["gcc_mostly"]) - t["onset"] for t in ts]) if ts else float("nan")
    out.append("Median delay from the onset to the first logged window that is mostly speech: "
               + "; ".join(f"{k} {v:.0f} ms" for k, v in firsts.items()) + " (the window ends there, plus 0.3 s of reference wait).\n")

    out.append("## 3. Own short-span correlation (input vs reference, lags 0-1000 ms)\n")
    out.append("Band 200-3600 Hz. PSR = PHAT peak against the other lags (as the gateway's); NCC = best normalised "
               "cross-correlation over the lags. Cells: p10 / p50 / p90. `lag ok` = peak within 5 ms of the lag reference (A/Am vs whole-burst peak; Be vs pre-onset logged lag).\n")
    for T in SPANS_MS:
        body = []
        for name, sc, _ in groups:
            for bp in (True, False):
                ts = [t for t in trials if t["scenario"] == sc and t["bypass"] is bp and t[f"in{T}"]]
                if not ts:
                    continue
                lagok = ""
                if sc in ("A", "Am", "Be"):
                    ts2 = [t for t in ts if lag_ref(t) is not None]
                    lagok = str(sum(abs(t[f"in{T}"]["lag_ms"] - lag_ref(t)) <= 5 for t in ts2)) + f"/{len(ts2)}"
                body.append([name, "on" if bp else "off", str(len(ts)), dist([t[f"in{T}"]["psr"] for t in ts]),
                             dist([t[f"in{T}"]["ncc"] * 100 for t in ts]), lagok or "-"])
        out.append(f"### Span onset to onset + {T} ms\n")
        out.append(table(["group", "bypass", "n", "PSR", "NCC x100", "lag ok"], body))

    out.append("## 4. Threshold separation (share flagged as 'our own audio')\n")
    out.append("Flagged = metric >= threshold. Leaks should be flagged (high), callers should not. Spans clipped at the end "
               "of the speech (C is 0.4-0.5 s long). A/Am/B/C columns are bypass-on cases (the 5 bypass-off A/Am leaks "
               "are in section 5); `Be` is the steady-echo line (canceller active) where no leak was injected. The "
               "`aec3` columns correlate the canceller's output instead of the raw input.\n")
    for metric, ths, scale in (("psr", PSR_T, 1), ("ncc", NCC_T, 1)):
        for T in SPANS_MS:
            def share(sc: str, th: float, bp: bool | None = True, key: str = "in") -> str:
                ts = [t for t in trials if t["scenario"] == sc and t[f"{key}{T}"] and (bp is None or t["bypass"] is bp)]
                if not ts:
                    return "n/a"
                k = sum(t[f"{key}{T}"][metric] >= th for t in ts)
                return f"{k}/{len(ts)}"
            head = [f"{metric.upper()} >=", "A leaks", "Am leaks", "B callers", "C callers", "Be callers (input)",
                    "A leaks (aec3, bypass on)", "Be callers (aec3)"]
            body = [[str(th), share("A", th), share("Am", th), share("B", th), share("C", th),
                     share("Be", th, None), share("A", th, True, "aec"), share("Be", th, None, "aec")] for th in ths]
            out.append(f"### {metric.upper()}, span {T} ms\n")
            out.append(table(head, body))

    out.append("## 5. A/Am leaks with the canceller active (bypass off at the onset)\n")
    body = []
    for sc in ("A", "Am"):
        ts = [t for t in trials if t["scenario"] == sc and t["bypass"] is False]
        for T in SPANS_MS:
            vs = [t[f"in{T}"]["psr"] for t in ts if t[f"in{T}"]]
            body.append([sc, str(T), str(len(vs)), dist(vs)])
    out.append(table(["scenario", "span ms", "n", "PSR p10/p50/p90"], body))
    out.append("""## 6. Reading and caveats

- On bypass lines (no steady echo) a PSR >= 10 on the 700-1000 ms span flags most leaks and no caller (B, C); at 400 ms
  the leak PSR is lower and the callers' tail rises (see section 4). The misses are bursts whose lag is not found
  (the same bursts have a stray lag in the gateway's own windows).
- On the steady-echo line (Be) the raw input correlates with the reference whatever the speaker says (the steady
  echo itself), so the raw correlation cannot separate callers; only the canceller's output (`aec3`) is a candidate
  there, and no leak was injected on such a line, so that case is untested.
- The burst model (`bursts.py`: band-limited, clipped, gain-modulated copies of the played audio at a fixed delay)
  is a synthetic assumption built from the playout reference. A real leak is a copy by construction; a real phone
  leak may be less correlated, so these separations are optimistic for leaks.
- Callers are clean recorded clips with no room echo of the bot beyond the Be steady echo. A caller talking in a
  room with strong bot echo would correlate like Be.
- Small n (11-24 leaks, 22-30 callers per cell): treat shares as indications.
""")
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

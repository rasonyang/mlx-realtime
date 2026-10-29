"""Results of the hold measurement: reads captures/hold/rows/*.json (bursts.py run) and writes
results/hold.json (the per-event rows and the summaries) and results/hold.md."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mlx_realtime.text.similarity import overlap, units, unmatched  # noqa: E402

ROWS = HERE / "captures" / "hold" / "rows"
RESULTS = HERE / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f} % [{100 * lo:.0f}, {100 * hi:.0f}]"


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def lat(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{pct(values, 0.5):.0f} / {pct(values, 0.95):.0f} (n={len(values)})"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ROWS.glob("*.json")):
        smoke = os.environ.get("HOLD_SMOKE") == "1"  # a check of the pipeline: the smoke rows only
        if path.name == "meta.json" or path.name.endswith(".smoke.json") != smoke:
            continue
        rows.extend(json.loads(path.read_text())["rows"])
    return rows


# Derivations


def items_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("capture", {}).get("items", [])


def fix_outcome(row: dict[str, Any]) -> str:
    """What the hold did with the turn(s) whose gate fired in the row's window."""
    items = items_of(row)
    if not items:
        return "no_gate"
    held = [i for i in items if i["held"]]
    if not held:
        return "gate_not_held"
    released = [i for i in held if "hold_release" in i]
    if released:
        return "released:" + released[0]["hold_release"]["reason"]
    dropped = [i for i in held if "hold_drop" in i]
    if dropped:
        return "dropped_" + dropped[0]["hold_drop"]["reason"]
    if any("hold_cancel" in i for i in held):
        return "cancelled"
    return "held_unresolved"


def cut_frames(row: dict[str, Any]) -> int:
    return sum(f["frames_dropped"] for f in row.get("client", {}).get("flushes", []))


def derive_a(row: dict[str, Any]) -> dict[str, Any]:
    items = items_of(row)
    client = row["client"]
    return {
        "gate": bool(items),
        "ss_client": len(client["speech_started"]),
        "ss_with_audio": any(s["audio"] for s in client["ss_snap"]),
        "self_interrupt": cut_frames(row) > 0,
        "cut_ms": cut_frames(row) * 20,
        "fix_outcome": fix_outcome(row),
        "bargein": [i.get("barge_in_outcome", {}).get("outcome") for i in items],
        "bypass": row["capture"]["bypass"],
    }


# Tables


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def a_section(rows: list[dict[str, Any]], out: list[str], scn: str = "A") -> dict[str, Any]:
    a = [r for r in rows if r["scenario"] == scn and "error" not in r]
    for r in a:
        r["derived"] = derive_a(r)
    title = {"A": "A. Residual-echo bursts, default (strongly processed) leak model", "Am": "Am. Residual-echo bursts, mild (cleaner, nearly linear) leak model"}[scn]
    out.append(f"## {title} (no steady echo)\n")
    errors = [r for r in rows if r["scenario"] == scn and "error" in r]
    cnt = Counter((r["condition"], r["kind"]) for r in a)
    out.append(f"Bursts run: {len(a)} ({', '.join(f'{k[0]} {k[1]} {v}' for k, v in sorted(cnt.items()))}); trials with an error: {len(errors)}.\n")
    bypass = Counter((r["condition"], r["derived"]["bypass"]) for r in a)
    out.append(f"Canceller bypass at the start of each burst's window: {dict(bypass)}.\n")

    def cells(group: str, key: Any) -> list[list[str]]:
        body = []
        for value in sorted({key(r) for r in a}):
            line = [str(value)]
            for cond in ("baseline", "fix"):
                sel = [r for r in a if r["condition"] == cond and key(r) == value]
                n = len(sel)
                line += [
                    rate(sum(r["derived"]["gate"] for r in sel), n),
                    rate(sum(r["derived"]["self_interrupt"] for r in sel), n),
                ]
            body.append(line)
        return body

    head = ["cell", "base: gate fired", "base: self-interruption", "fix: gate fired", "fix: self-interruption"]
    out.append("Self-interruption = AICC flushed unplayed frames after a speech_started that no caller spoke (the harness's own cleanup flushes are excluded). Gate fired = the gateway's VAD rules let the burst start a turn (baseline: speech_started; fix: hold). Wilson 95 % intervals in brackets.\n")
    out.append("### By placement\n")
    out.append(table(head, cells("kind", lambda r: r["kind"]) + cells("all", lambda r: "all")))
    out.append("### By placement and delay\n")
    out.append(table(head, cells("kd", lambda r: f"{r['kind']} {int(r['delay_s'] * 1000)} ms")))
    out.append("### By placement and level (dB under the played audio)\n")
    out.append(table(head, cells("kl", lambda r: f"{r['kind']} {int(r['level_db'])} dB")))
    out.append("### By level\n")
    out.append(table(head, cells("l", lambda r: f"{int(r['level_db'])} dB")))

    out.append("### What the hold did with the bursts that fired the gate (fix)\n")
    fix = [r for r in a if r["condition"] == "fix"]
    outcomes = Counter((r["kind"], r["derived"]["fix_outcome"]) for r in fix)
    kinds = sorted({r["kind"] for r in fix})
    names = sorted({o for _, o in outcomes})
    out.append(table(["outcome", *kinds, "all"], [[n, *[str(outcomes[(k, n)]) for k in kinds], str(sum(outcomes[(k, n)] for k in kinds))] for n in names]))
    released = [r for r in fix if r["derived"]["fix_outcome"].startswith("released")]
    out.append("### Released bursts (fix): why\n")
    body = []
    for r in released:
        item = [i for i in items_of(r) if "hold_release" in i][0]
        checks = [c for c in item["checks"] if c["verdict"] != "failed"]
        last = checks[-1] if checks else {}
        body.append([r["id"], r["kind"], f"{int(r['delay_s'] * 1000)}", f"{int(r['level_db'])}", item["hold_release"]["reason"],
                     str(item["hold_release"]["held_ms"]), str(last.get("unmatched")), repr(last.get("text"))[:50], repr((last.get("played_text") or ""))[-40:],
                     "yes" if r["derived"]["self_interrupt"] else "no"])
    out.append(table(["trial", "kind", "delay", "level", "reason", "held ms", "unmatched", "ASR text", "played text (end)", "self-interruption"], body) if body else "No burst was released.\n")


    out.append("### How the ASR hears a held burst (fix): first check and last check before the decision\n")
    body = []
    for level in sorted({r["level_db"] for r in fix}):
        firsts, lasts, spans = [], [], []
        for r in fix:
            if r["level_db"] != level:
                continue
            for i in items_of(r):
                cs = [c for c in i["checks"] if c["verdict"] != "failed" and c["text"] is not None]
                if not i["held"] or not cs:
                    continue
                for store, c in ((firsts, cs[0]), (lasts, cs[-1])):
                    n = len(units(c["text"]))
                    store.append(1 - c["unmatched"] / n if n else float("nan"))
                spans.append(cs[0]["audio_end_ms"] - cs[0]["audio_start_ms"])
        clean = lambda xs: [x for x in xs if x == x]
        body.append([f"{int(level)} dB", str(len(firsts)), f"{pct(clean(firsts), .5):.2f}" if clean(firsts) else "n/a", f"{pct(clean(lasts), .5):.2f}" if clean(lasts) else "n/a", lat([float(s) for s in spans])])
    out.append(table(["level", "held bursts", "median share of ASR units found in the played text, first check", "same, last check", "audio of the first check, ms (P50 / P95)"], body))
    spans_rel = [c["audio_end_ms"] - c["audio_start_ms"] for r in released for i in items_of(r) if "hold_release" in i for c in i["checks"][-1:]]
    out.append(f"Audio covered by the check that released a burst (false release): {lat([float(s) for s in spans_rel])} ms.\n")
    counts = Counter(len([c for c in i["checks"] if c["verdict"] != "failed"]) for r in released for i in items_of(r) if "hold_release" in i and i["hold_release"]["reason"] == "caller")
    out.append(f"False releases by the number of checks run before the decision (1 = decided on the very first check): {dict(sorted(counts.items()))}.\n")
    base = [r for r in a if r["condition"] == "baseline"]
    out.append("### Baseline: what the burst turns became\n")
    out.append(f"Baseline bursts with a gate: {sum(r['derived']['gate'] for r in base)}; speech_started at the client while AICC had audio: {sum(r['derived']['ss_with_audio'] for r in base)}; barge-in outcomes of those turns: {dict(Counter(o for r in base for o in r['derived']['bargein']))}; mean audio cut by a self-interruption: "
               f"{(sum(r['derived']['cut_ms'] for r in base if r['derived']['self_interrupt']) / max(1, sum(r['derived']['self_interrupt'] for r in base))):.0f} ms.\n")
    realised = [r["burst"]["realised_db"] for r in a if r["burst"]["realised_db"] is not None]
    out.append(f"Realised burst level against the played audio (rms over the burst, dB): P50 {pct(realised, .5):.1f}, min {min(realised):.1f}, max {max(realised):.1f}.\n")
    tails = [r for r in a if r["kind"] == "A3" and r["answer_s"] - (r["burst"]["playback_pos_s"] + r["length_s"]) < 3.0]
    out.append(f"A3 bursts whose source ended less than 3 s before the end of the answer (not mid-answer; answers whose audio was cut before the end count as mid): {len(tails)}.\n")
    return {"n": len(a)}


def latency_rows(rows: list[dict[str, Any]], scenario: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"baseline": [], "fix": []}
    for r in rows:
        if r["scenario"] == scenario and "error" not in r:
            out[r["condition"]].append(r)
    return out


def b_section(rows: list[dict[str, Any]], out: list[str], scenario: str, title: str) -> None:
    sel = latency_rows(rows, scenario)
    out.append(f"## {title}\n")
    body = []
    for cond in ("baseline", "fix"):
        rs = sel[cond]
        got = [r for r in rs if r["ss"]]
        body.append([cond, str(len(rs)), rate(len(got), len(rs)), lat([r["ss"]["dt_ms"] for r in got]),
                     lat([r["flush_dt_ms"] for r in got if r["flush_dt_ms"] is not None]),
                     lat([r["transcript"]["dt_from_stopped_ms"] for r in got if r["transcript"] and r["transcript"]["dt_from_stopped_ms"] is not None]),
                     str(sum(1 for r in rs if not r["ss"]))])
    held = [i["speech_started"]["held_ms"] for r in sel["fix"] for i in items_of(r) if i.get("held") and "speech_started" in i and "held_ms" in i["speech_started"]]
    out.append(f"Added delay of the hold itself (fix, `speech_started.held_ms`, hold -> release): {lat([float(h) for h in held])}. The onset -> speech_started columns also contain the VAD's own delay, which varies from run to run (the bot's audio under the caller differs), so compare the held_ms.\n")
    out.append(table(["condition", "barge-ins", "speech_started at the client", "onset -> speech_started ms (P50 / P95)", "onset -> AICC flush ms (P50 / P95)", "speech_stopped -> transcript ms (P50 / P95)", "turns without speech_started"], body))
    spans = [float(c["audio_end_ms"] - c["audio_start_ms"]) for r in sel["fix"] for i in items_of(r) if i.get("held") and i.get("hold_release", {}).get("reason") == "caller" for c in i["checks"][-1:] if c["audio_end_ms"] is not None]
    nchecks = [float(len(i["checks"])) for r in sel["fix"] for i in items_of(r) if i.get("held")]
    out.append(f"Audio covered by the check that released a real caller (ms, P50 / P95): {lat(spans)}; ASR checks per held turn: {lat(nchecks)}.\n")
    gate_delay = {c: [i["gate_t"] - (r["n_inject"] * 20 + r["onset_s"] * 1000) for r in sel[c] for i in items_of(r)[:1]] for c in sel}
    out.append(f"VAD gate delay (onset in the injected audio -> the gate fired in the capture, input ms; P50 / P95): baseline {lat(gate_delay['baseline'])}, fix {lat(gate_delay['fix'])}. Same code in both; it varies with the bot audio under the caller.\n")
    # paired by clip
    base = {r["id"]: r for r in sel["baseline"]}
    fix = {r["id"]: r for r in sel["fix"]}
    pairs = [(base[k], fix[k]) for k in base if k in fix and base[k]["ss"] and fix[k]["ss"]]
    deltas = [f["ss"]["dt_ms"] - b["ss"]["dt_ms"] for b, f in pairs]
    if deltas:
        out.append(f"Paired by trial (same clip and offset), fix minus baseline speech_started latency: P50 {pct(deltas, .5):+.0f} ms, P95 {pct(deltas, .95):+.0f} ms, mean {sum(deltas) / len(deltas):+.0f} ms (n={len(deltas)}).\n")
    by_set: list[list[str]] = []
    for name in ("zh", "en", "mix"):
        for cond in ("baseline", "fix"):
            rs = [r for r in sel[cond] if r["clip"]["set"] == name]
            got = [r for r in rs if r["ss"]]
            by_set.append([name, cond, str(len(rs)), str(len(got)), lat([r["ss"]["dt_ms"] for r in got]), lat([r["flush_dt_ms"] for r in got if r["flush_dt_ms"] is not None])])
    out.append(table(["set", "condition", "n", "speech_started", "onset -> speech_started ms", "onset -> flush ms"], by_set))
    out.append("Fix outcomes (how each turn was released or dropped):\n")
    outcomes = Counter(fix_outcome(r) for r in sel["fix"])
    out.append(table(["outcome", "count"], [[k, str(v)] for k, v in sorted(outcomes.items())]))
    base_out = Counter((fix_outcome(r) if False else ("speech_started" if r["ss"] else "none")) for r in sel["baseline"])
    out.append(f"Baseline: {dict(base_out)}.\n")
    # transcripts of committed turns
    out.append("Committed-turn transcripts (baseline | fix), same clip:\n")
    body = []
    same = 0
    comparable = 0
    for k in sorted(base):
        if k not in fix:
            continue
        bt = (base[k]["transcript"] or {}).get("text", "")
        ft = (fix[k]["transcript"] or {}).get("text", "")
        if base[k]["ss"] and fix[k]["ss"]:
            comparable += 1
            same += bt == ft
        if bt != ft:
            body.append([k, base[k]["clip"]["id"], repr(bt), repr(ft), fix_outcome(fix[k])])
    out.append(f"Identical transcripts when both conditions committed the turn: {same}/{comparable}. Differences:\n")
    out.append(table(["trial", "clip", "baseline", "fix", "fix outcome"], body) if body else "None.\n")
    dropped = [r for r in sel["fix"] if not r["ss"]]
    out.append(f"Real turns dropped in the fix (no speech_started at the client): {len(dropped)}. " + ", ".join(f"{r['id']} ({fix_outcome(r)}, {r['clip']['id']})" for r in dropped) + "\n")


def c_section(rows: list[dict[str, Any]], out: list[str]) -> None:
    sel = latency_rows(rows, "C")
    out.append("## C. Short caller utterances over playback\n")
    out.append("Utterances synthesised with the gateway's Qwen3-TTS: Chinese in `uncle_fu`, English in `aiden` (the bot speaks `vivian`), trimmed, scaled to -26 dBov active level, played from a clean 16 kHz clip through the simulated line.\n")
    body = []
    for text in [t for t, _, _ in __import__("bursts").SHORT_UTTERANCES]:
        line = [text]
        for cond in ("baseline", "fix"):
            rs = [r for r in sel[cond] if r["utterance"] == text]
            got = [r for r in rs if r["ss"]]
            if cond == "baseline":
                line += [f"{len(got)}/{len(rs)}", lat([r["ss"]["dt_ms"] for r in got])]
            else:
                oc = Counter(fix_outcome(r) for r in rs)
                line += [f"{len(got)}/{len(rs)}", lat([r["ss"]["dt_ms"] for r in got]), ", ".join(f"{k} {v}" for k, v in sorted(oc.items()))]
        body.append(line)
    out.append(table(["utterance", "base: barge-ins", "base: onset -> speech_started ms", "fix: barge-ins", "fix: onset -> speech_started ms", "fix outcomes"], body))
    out.append("Totals by fix outcome (latency onset -> speech_started at the client, ms P50 / P95):\n")
    groups: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for r in sel["fix"]:
        o = fix_outcome(r)
        counts[o] += 1
        if r["ss"]:
            groups[o].append(r["ss"]["dt_ms"])
    out.append(table(["outcome", "count", "latency"], [[k, str(counts[k]), lat(groups.get(k, []))] for k in sorted(counts)]))
    base_ss = [r for r in sel["baseline"] if r["ss"]]
    out.append(f"Baseline: {len(base_ss)}/{len(sel['baseline'])} barge-ins, latency {lat([r['ss']['dt_ms'] for r in base_ss])}.\n")


def d_section(rows: list[dict[str, Any]], out: list[str]) -> None:
    sel = latency_rows(rows, "D")
    out.append("## D. Non-speech noise over playback\n")
    body = []
    for kind in ("cup", "keys", "clicks"):
        for cond in ("baseline", "fix"):
            rs = [r for r in sel[cond] if r["noise"] == kind]
            gate = sum(1 for r in rs if items_of(r))
            ss = sum(1 for r in rs if r["ss"])
            oc = Counter(fix_outcome(r) for r in rs)
            body.append([kind, cond, str(len(rs)), str(gate), str(ss), ", ".join(f"{k} {v}" for k, v in sorted(oc.items())) if cond == "fix" else "-"])
    out.append(table(["sound", "condition", "n", "gate fired", "barge-in (speech_started)", "fix outcomes"], body))
    out.append("Noise is synthesised (damped resonances, key clicks with a low thump, a 8 Hz click train) at a peak of -3, -12 or -24 dBFS; hold_lib.synth_noise.\n")


def e_section(rows: list[dict[str, Any]], out: list[str]) -> dict[str, Any]:
    out.append("## E. Cost of the checks\n")
    fix = [r for r in rows if r["condition"] == "fix" and "capture" in r]
    per_turn: list[int] = []
    latencies: list[float] = []
    finals: list[int] = []
    for r in fix:
        for item in items_of(r):
            if not item["held"]:
                continue
            checks = [c for c in item["checks"] if c["verdict"] != "failed"]
            per_turn.append(len(item["checks"]))
            finals.append(sum(1 for c in checks if c["final"]))
            for c in checks:
                if not c["final"] and c["audio_end_ms"] is not None and c["t"] is not None:
                    latencies.append(c["t"] - c["audio_end_ms"])
    out.append(f"Held turns in the fix runs: {len(per_turn)}; ASR checks per held turn: mean {sum(per_turn) / max(1, len(per_turn)):.1f}, P50 {pct([float(x) for x in per_turn], .5)}, P95 {pct([float(x) for x in per_turn], .95)}, max {max(per_turn) if per_turn else 0}; failed checks: "
               f"{sum(1 for r in fix for i in items_of(r) for c in i['checks'] if c['verdict'] == 'failed')}.\n")
    out.append(f"Check latency (request -> result), estimated as the capture time of the result less the end of the audio the check covered (non-final checks only; ms, P50 / P95): {lat(latencies)}.\n")
    by_scn = Counter()
    held_by = Counter()
    for r in fix:
        for item in items_of(r):
            if item["held"]:
                held_by[r["scenario"]] += 1
                by_scn[r["scenario"]] += len(item["checks"])
    out.append("Checks per held turn by scenario: " + ", ".join(f"{k}: {by_scn[k]}/{held_by[k]} = {by_scn[k] / held_by[k]:.1f}" for k in sorted(held_by)) + ".\n")
    out.append("### Commit -> transcript latency (speech_stopped at the client -> transcription.completed, ms P50 / P95)\n")
    body = []
    for scn in ("E0", "B", "Be", "C"):
        sel = latency_rows(rows, scn)
        line = [scn]
        for cond in ("baseline", "fix"):
            vals = [r["transcript"]["dt_from_stopped_ms"] for r in sel[cond] if r.get("transcript") and r["transcript"]["dt_from_stopped_ms"] is not None]
            line.append(lat(vals))
        body.append(line)
    out.append(table(["scenario", "baseline", "fix"], body))
    out.append("E0 are ordinary turns over silence (no hold possible): any difference there is run-to-run variation. B, Be and C turns were held, and the fix reuses the final check's transcript, so their commit -> transcript time can be shorter.\n")
    return {"checks_per_turn": per_turn, "check_latency_ms": latencies}


# What-ifs


def replay_item(item: dict[str, Any], n_min: int, *, echo_threshold: float = 0.7, echo_units: int = 4) -> dict[str, Any]:
    """The decision of the hold for ``n_min`` over the checks recorded for one held turn."""
    for c in item["checks"]:
        if c["verdict"] == "failed" or c["text"] is None:
            return {"decision": "failed", "t": c["t"]}
        played = c["played_text"] or ""
        new = unmatched(c["text"], played)
        count = len(units(c["text"]))
        if new >= n_min:
            return {"decision": "caller", "t": c["t"], "final": c["final"]}
        if c["final"]:
            if count == 0:
                return {"decision": "noise", "t": c["t"], "final": True}
            echo = count >= echo_units and bool(played) and overlap(c["text"], played) >= echo_threshold
            return {"decision": "echo" if echo else "turn_end", "t": c["t"], "final": True}
    return {"decision": "undecided", "t": item["checks"][-1]["t"] if item["checks"] else None}


def whatif_section(rows: list[dict[str, Any]], out: list[str]) -> dict[str, Any]:
    out.append("## Offline what-ifs: confirm_min_new_units = 1 and 3\n")
    out.append("Replayed from the fix runs' `hold_check` events (each check's ASR text and the assistant text played around it; units and the longest in-order match are the gateway's own `text.similarity`). Limits: the recorded sequence of a turn stops at the check where N=2 decided. N=1 can only decide earlier or at the same check, so its decision and latency are exact. N=3 agrees with N=2 only if that check already had at least 3 unmatched units; otherwise the later checks, which would have shown whether 3 is reached, do not exist (censored: N=3 releases at or after the recorded time, or never). Latencies are capture time of the deciding check less the onset in the injected audio; they exclude the delivery of the event.\n")
    fix = [r for r in rows if r["condition"] == "fix" and "capture" in r and "error" not in r]
    result: dict[str, Any] = {}
    # A: false releases
    for scn in ("A", "Am"):
        a = [r for r in fix if r["scenario"] == scn]
        if not a:
            continue
        held_a = [(r, i) for r in a for i in items_of(r) if i["held"]]
        out.append(f"### {scn} (bursts): false releases\n")
        result[scn] = whatif_a(a, held_a, out)
    return whatif_bc(fix, out, result)


def whatif_a(a: list[dict[str, Any]], held_a: list[Any], out: list[str]) -> list[list[str]]:
    body: list[list[str]] = []
    for n_min in (1, 2, 3):
        dec = [replay_item(i, n_min) for _, i in held_a]
        released = sum(1 for d in dec if d["decision"] == "caller")
        cens = 0
        if n_min == 3:  # N=2 released with < 3 unmatched at its last check: N=3 unknown
            cens = sum(1 for (r, i), d in zip(held_a, dec, strict=True) if d["decision"] == "undecided" and "hold_release" in i and i["hold_release"]["reason"] == "caller")
        share = f"{released}/{len(a)} bursts = {100 * released / max(1, len(a)):.0f} %"
        if n_min == 3:
            share += f" (upper bound {100 * (released + cens) / max(1, len(a)):.0f} %)"
        body.append([str(n_min), str(len(held_a)), str(released), str(cens), share])
    out.append(table(["N", "held burst turns", "released as caller", "censored (N=3 unknown)", "false-release rate per burst"], body))
    return body


def whatif_bc(fix: list[dict[str, Any]], out: list[str], result: dict[str, Any]) -> dict[str, Any]:
    for scn, label in (("B", "B (real barge-ins, no echo)"), ("Be", "Be (real barge-ins, steady echo)"), ("C", "C (short utterances)")):
        sel = [r for r in fix if r["scenario"] == scn]
        if not sel:
            continue
        out.append(f"### {label}: release by the checks\n")
        body = []
        for n_min in (1, 2, 3):
            lats: list[float] = []
            early = turn_end = other = cens = 0
            for r in sel:
                onset_ms = r["n_inject"] * 20 + r["onset_s"] * 1000
                for i in items_of(r):
                    if not i["held"]:
                        continue
                    d = replay_item(i, n_min)
                    actual = i.get("hold_release", {}).get("reason")
                    if d["decision"] == "caller":
                        if d.get("final"):
                            turn_end += 1
                        else:
                            early += 1
                        lats.append(d["t"] - onset_ms)
                    elif d["decision"] == "turn_end":
                        turn_end += 1
                        lats.append(d["t"] - onset_ms)
                    elif d["decision"] == "undecided" and actual == "caller" and n_min == 3:
                        cens += 1
                    else:
                        other += 1
            body.append([str(n_min), str(early), str(turn_end), str(other), str(cens), lat(lats)])
        out.append(table(["N", "released early (partial check)", "released at the final check", "dropped / other", "censored", "release latency from onset, ms (P50 / P95)"], body))
        result[scn] = body
    return result


# Setup


def setup_section(out: list[str]) -> dict[str, Any]:
    meta_path = ROWS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out.append("## Setup\n")
    for key in ("models", "pmset", "git", "config", "seed", "burst_model", "started", "wall_s"):
        if key in meta:
            value = meta[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            out.append(f"- **{key}**: {value}\n" if key != "pmset" else f"- **pmset -g** (at the start of the run):\n\n```\n{value}\n```\n")
    return meta


def report() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("no rows: run bursts.py first")
    out = ["# Hold measurement (issue #5)\n"]
    findings = RESULTS / "hold_findings.md"
    if findings.exists():
        out.append(findings.read_text() + "\n")
    meta = setup_section(out)
    summary: dict[str, Any] = {}
    for scn in ("A", "Am"):
        if any(r["scenario"] == scn for r in rows):
            summary[scn] = a_section(rows, out, scn)
    for scn, title in (("B", "B. Real barge-ins over playback, line without echo"), ("Be", "Be. Real barge-ins over playback, line with steady echo (ERL 10 dB, 100 / 180 ms)"), ("E0", "E0. Ordinary turns over silence")):
        if any(r["scenario"] == scn for r in rows):
            b_section(rows, out, scn, title)
    if any(r["scenario"] == "C" for r in rows):
        c_section(rows, out)
    if any(r["scenario"] == "D" for r in rows):
        d_section(rows, out)
    summary["E"] = e_section(rows, out)
    summary["whatif"] = whatif_section(rows, out)
    RESULTS.mkdir(exist_ok=True)
    if os.environ.get("HOLD_SMOKE") == "1":
        (RESULTS / "hold_smoke.md").write_text("\n".join(out))
        print("smoke report written to results/hold_smoke.md")
        return
    (RESULTS / "hold.md").write_text("\n".join(out))
    (RESULTS / "hold.json").write_text(json.dumps({"setup": meta, "summary": summary, "rows": rows}, ensure_ascii=False, indent=1, default=str))
    print(f"wrote {RESULTS / 'hold.md'} and hold.json ({len(rows)} rows)")

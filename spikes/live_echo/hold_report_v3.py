"""Results of the v3 hold measurement (bursts.py run with HOLD_DIR=hold_v3): baseline rows come from
the v1 run (captures/hold/rows, the same code path: confirm_while_playing false), the v1 fix rows
are kept for comparison, the v3 rows are captures/hold_v3/rows. Writes results/hold_v3.md and .json.

    uv run --locked python spikes/live_echo/hold_report_v3.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from hold_report import derive_a, fix_outcome, items_of, lat, pct, rate, table  # noqa: E402

V1 = HERE / "captures" / "hold" / "rows"
V3 = HERE / "captures" / "hold_v3" / "rows"
RESULTS = HERE / "results"
CONDS = ("baseline", "v1", "v3")
SCEN_V1 = {"A", "Am", "B", "Be", "C"}


def load() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(V1.glob("*.json")):
        if path.name == "meta.json" or path.name.endswith(".smoke.json"):
            continue
        for r in json.loads(path.read_text())["rows"]:
            if r["scenario"] not in SCEN_V1:
                continue
            r["condition"] = "baseline" if r["condition"] == "baseline" else "v1"
            rows.append(r)
    for path in sorted(V3.glob("*.json")):
        if path.name == "meta.json" or path.name.endswith(".smoke.json"):
            continue
        for r in json.loads(path.read_text())["rows"]:
            if r["condition"] == "fix":
                r["condition"] = "v3"
            rows.append(r)
    return rows


def sel(rows: list[dict[str, Any]], scn: str, cond: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["scenario"] == scn and r["condition"] == cond and "error" not in r]


def onset_ms(r: dict[str, Any]) -> float:
    return r["n_inject"] * 20 + r["onset_s"] * 1000


def a_section(rows: list[dict[str, Any]], out: list[str], scn: str, title: str) -> None:
    out.append(f"## {title}\n")
    data = {c: sel(rows, scn, c) for c in CONDS}
    for c in CONDS:
        for r in data[c]:
            r["derived"] = derive_a(r)
    out.append("Self-interruption = AICC flushed unplayed frames after a speech_started with no caller. Baseline and v1 are the v1 run; v3 is the current fix. [Wilson 95 %]\n")

    def cells(key: Any) -> list[list[str]]:
        body = []
        for value in sorted({key(r) for r in data["v3"]}):
            line = [str(value)]
            for c in CONDS:
                rs = [r for r in data[c] if key(r) == value]
                line.append(rate(sum(r["derived"]["self_interrupt"] for r in rs), len(rs)))
            body.append(line)
        return body

    head = ["cell", "baseline", "v1", "v3"]
    out.append(table(head, cells(lambda r: r["kind"]) + cells(lambda r: "all")))
    out.append("By placement and level:\n")
    out.append(table(head, cells(lambda r: f"{r['kind']} {int(r['level_db'])} dB")))
    out.append("By placement and delay:\n")
    out.append(table(head, cells(lambda r: f"{r['kind']} {int(r['delay_s'] * 1000)} ms")))
    out.append("Gate fired (the bursts that could have interrupted): " + ", ".join(
        f"{c} {rate(sum(r['derived']['gate'] for r in data[c]), len(data[c]))}" for c in CONDS) + "\n")
    for c in ("v1", "v3"):
        oc = Counter((r["kind"], r["derived"]["fix_outcome"]) for r in data[c])
        kinds = sorted({k for k, _ in oc})
        names = sorted({o for _, o in oc})
        out.append(f"{c}: what the hold did\n")
        out.append(table(["outcome", *kinds, "all"], [[n, *[str(oc[(k, n)]) for k in kinds], str(sum(oc[(k, n)] for k in kinds))] for n in names]))
    body = []
    for r in data["v3"]:
        for i in items_of(r):
            if i.get("hold_release", {}).get("reason") in (None, "playback_ended"):
                continue
            cs = [c for c in i["checks"] if c["verdict"] not in ("stale",)]
            last = cs[-1] if cs else {}
            body.append([r["id"], r["kind"], str(int(r["delay_s"] * 1000)), str(int(r["level_db"])), i["hold_release"]["reason"],
                         str(i["hold_release"]["held_ms"]), repr(last.get("text"))[:40], str(last.get("unmatched")), str(last.get("psr")),
                         str(last.get("psr_why")), "yes" if r["derived"]["self_interrupt"] else "no"])
    out.append("v3 releases other than at playback end:\n")
    out.append(table(["trial", "kind", "delay", "level", "reason", "held ms", "ASR text", "unmatched", "psr", "psr_why", "self-interruption"], body) if body else "None.\n")
    pb = Counter(i.get("hold_release", {}).get("reason") for r in data["v3"] for i in items_of(r) if "hold_release" in i)
    out.append(f"v3 release reasons: {dict(pb)}; drops: {dict(Counter(i['hold_drop']['reason'] + ('/' + i['hold_drop'].get('why', '') if i['hold_drop'].get('why') else '') for r in data['v3'] for i in items_of(r) if 'hold_drop' in i))}.\n")
    psr = [(c["psr"], c["psr_why"]) for r in data["v3"] for i in items_of(r) for c in i["checks"]]
    out.append(f"psr_why over all v3 checks: {dict(Counter(w for _, w in psr))}.\n")


def b_section(rows: list[dict[str, Any]], out: list[str], scn: str, title: str) -> None:
    out.append(f"## {title}\n")
    data = {c: sel(rows, scn, c) for c in CONDS}
    body = []
    for c in CONDS:
        rs = data[c]
        got = [r for r in rs if r["ss"]]
        held = [float(i["speech_started"]["held_ms"]) for r in rs for i in items_of(r) if i.get("held") and "held_ms" in i.get("speech_started", {})]
        body.append([c, str(len(rs)), rate(len(got), len(rs)), lat([r["ss"]["dt_ms"] for r in got]),
                     lat([r["flush_dt_ms"] for r in got if r["flush_dt_ms"] is not None]), lat(held),
                     lat([r["transcript"]["dt_from_stopped_ms"] for r in got if r["transcript"] and r["transcript"]["dt_from_stopped_ms"] is not None]),
                     str(sum(1 for r in rs if not r["ss"]))])
    out.append(table(["condition", "barge-ins", "speech_started", "onset -> speech_started ms (P50 / P95)", "onset -> flush ms", "held_ms", "speech_stopped -> transcript ms", "dropped (no speech_started)"], body))
    base = {r["id"]: r for r in data["baseline"]}
    for c in ("v1", "v3"):
        pairs = [(base[r["id"]], r) for r in data[c] if r["id"] in base and base[r["id"]]["ss"] and r["ss"]]
        d = [f["ss"]["dt_ms"] - b["ss"]["dt_ms"] for b, f in pairs]
        if d:
            out.append(f"Paired by trial, {c} minus baseline speech_started latency: P50 {pct(d, .5):+.0f} ms, P95 {pct(d, .95):+.0f} ms, mean {sum(d) / len(d):+.0f} (n={len(d)}).\n")
    for c in ("v1", "v3"):
        oc = Counter(fix_outcome(r) for r in data[c])
        out.append(f"{c} outcomes: {dict(oc)}\n")
        dropped = [r for r in data[c] if not r["ss"]]
        out.append(f"{c} real turns dropped: {len(dropped)} " + ", ".join(f"{r['id']} ({fix_outcome(r)})" for r in dropped) + "\n")
    same = comp = 0
    for r in data["v3"]:
        b = base.get(r["id"])
        if b and b["ss"] and r["ss"]:
            comp += 1
            same += (b["transcript"] or {}).get("text") == (r["transcript"] or {}).get("text")
    out.append(f"Identical committed transcripts, baseline vs v3: {same}/{comp}.\n")


def c_section(rows: list[dict[str, Any]], out: list[str]) -> None:
    out.append("## C. Short utterances over playback\n")
    data = {c: sel(rows, "C", c) for c in CONDS}
    texts = []
    for r in data["v3"]:
        if r["utterance"] not in texts:
            texts.append(r["utterance"])
    body = []
    for t in texts:
        line = [t]
        for c in CONDS:
            rs = [r for r in data[c] if r["utterance"] == t]
            got = [r for r in rs if r["ss"]]
            oc = Counter(fix_outcome(r) for r in rs)
            line.append(f"{len(got)}/{len(rs)}; {lat([r['ss']['dt_ms'] for r in got])}" + ("" if c == "baseline" else "; " + ", ".join(f"{k} {v}" for k, v in sorted(oc.items()))))
        body.append(line)
    out.append(table(["utterance", "baseline: barge-ins; onset -> speech_started ms", "v1", "v3"], body))
    for c in ("v1", "v3"):
        g: dict[str, list[float]] = {}
        for r in data[c]:
            if r["ss"]:
                g.setdefault(fix_outcome(r), []).append(r["ss"]["dt_ms"])
        out.append(f"{c} by outcome: " + "; ".join(f"{k}: n={len(v)}, {lat(v)}" for k, v in sorted(g.items())) + f"; dropped real: {sum(1 for r in data[c] if items_of(r) and not r['ss'] and fix_outcome(r).startswith('dropped'))}\n")


def max_psr(r: dict[str, Any]) -> float | None:
    vals = [c["psr"] for i in items_of(r) for c in i["checks"] if c.get("psr") is not None]
    return max(vals) if vals else None


def f_section(rows: list[dict[str, Any]], out: list[str]) -> None:
    out.append("## F. Double talk: a caller speaks while a leak burst is on the line\n")
    out.append("F1: the caller's speech starts with the burst; F2: 0.5-1 s into it. Caller clips as in B. Leak -5 / -20 dB, 316 / 506 ms. A caller 'dropped' = the turn never reached the client as speech_started.\n")
    data = {c: [r for r in rows if r["scenario"] == "F" and r["condition"] == c and "error" not in r] for c in ("baseline", "v3")}
    body = []
    for c in ("baseline", "v3"):
        for kind in ("F1", "F2"):
            for level in (-5.0, -20.0):
                rs = [r for r in data[c] if r["kind"] == kind and r["level_db"] == level]
                got = [r for r in rs if r["ss"]]
                oc = Counter(fix_outcome(r) for r in rs)
                psrs = [p for p in (max_psr(r) for r in rs) if p is not None]
                body.append([c, kind, f"{int(level)}", str(len(rs)), rate(len(got), len(rs)), lat([r["ss"]["dt_ms"] for r in got]),
                             ", ".join(f"{k} {v}" for k, v in sorted(oc.items())) if c == "v3" else "-",
                             f"{pct(psrs, .5):.1f} / {max(psrs):.1f}" if psrs else "n/a"])
    out.append(table(["condition", "kind", "leak dB", "n", "caller reached the client", "onset -> speech_started ms", "v3 outcomes", "max psr per trial (P50 / max)"], body))
    for c in ("baseline", "v3"):
        rs = data[c]
        got = [r for r in rs if r["ss"]]
        out.append(f"{c} overall: reached {rate(len(got), len(rs))}; dropped {len(rs) - len(got)}.\n")
    held = [float(i["speech_started"]["held_ms"]) for r in data["v3"] for i in items_of(r) if i.get("held") and "held_ms" in i.get("speech_started", {})]
    out.append(f"v3 hold time of released callers (held_ms): {lat(held)}.\n")
    body = []
    for r in data["v3"]:
        if r["ss"] and fix_outcome(r) == "released:caller" and False:
            continue
        o = fix_outcome(r)
        if o.startswith("released:caller") or not r["ss"]:
            pass
        i = (items_of(r) or [{}])[0]
        body.append([r["id"], r["kind"], str(int(r["level_db"])), o, str(i.get("hold_release", {}).get("held_ms") or i.get("hold_drop", {}).get("held_ms")),
                     str(max_psr(r)), str(len([c for c in i.get("checks", [])])), "-" if r["ss"] else repr((i.get("hold_drop") or {}).get("text"))[:40]])
    out.append("Per trial (v3):\n")
    out.append(table(["trial", "kind", "leak dB", "outcome", "held/drop ms", "max psr", "checks", "dropped text"], body))
    wrong = [r for r in data["v3"] if not r["ss"]]
    out.append(f"Callers dropped by v3: {[(r['id'], fix_outcome(r)) for r in wrong]}\n")


def e_section(rows: list[dict[str, Any]], out: list[str]) -> None:
    out.append("## Cost of the checks\n")
    body = []
    for c in ("v1", "v3"):
        for scn in ("A", "Am", "B", "Be", "C", "F"):
            if c == "v1" and scn == "F":
                continue
            held = [i for r in sel(rows, scn, c) for i in items_of(r) if i.get("held")]
            n = [len([k for k in i["checks"] if k["verdict"] != "stale"]) for i in held]
            body.append([c, scn, str(len(held)), f"{sum(n) / max(1, len(n)):.1f}", str(max(n) if n else 0)])
    out.append(table(["version", "scenario", "held turns", "checks per held turn (mean)", "max"], body))
    lat_ms = []
    kinds: Counter[str] = Counter()
    psr_n = Counter()
    for r in rows:
        if r["condition"] != "v3":
            continue
        for i in items_of(r):
            for c in i["checks"]:
                kinds[c.get("check")] += 1
                psr_n[c.get("psr_why")] += 1
                if c.get("check") == "partial" and c["audio_end_ms"] is not None and c["t"] is not None:
                    lat_ms.append(c["t"] - c["audio_end_ms"])
    out.append(f"v3 checks by kind: {dict(kinds)}; psr_why: {dict(psr_n)}. Partial-check latency (result less end of its audio, ms P50 / P95): {lat(lat_ms)}.\n")
    ords = []
    for scn in ("B", "Be"):
        for c in CONDS:
            ords.append([scn, c, lat([r["transcript"]["dt_from_stopped_ms"] for r in sel(rows, scn, c) if r.get("transcript") and r["transcript"]["dt_from_stopped_ms"] is not None])])
    out.append("speech_stopped -> transcript ms (P50 / P95):\n")
    out.append(table(["scenario", "condition", "latency"], ords))


# What-ifs on the recorded checks


def replay(item: dict[str, Any], psr_t: float | None, drop_t: float | None) -> dict[str, Any]:
    for c in item["checks"]:
        v = c["verdict"]
        if v == "stale":
            continue
        if v == "failed":
            return {"d": "release", "t": c["t"], "why": "failed"}
        if v in ("echo", "noise"):
            return {"d": "drop", "t": c["t"], "why": v}
        if v in ("caller", "echo_signal", "echo_signal_drop"):
            psr = c.get("psr")
            flagged = psr is not None and psr_t is not None and psr >= psr_t
            final = c.get("check") == "final"
            if not flagged:
                return {"d": "release", "t": c["t"], "why": c.get("check")}
            if not final:
                continue
            if drop_t is not None and psr >= drop_t:
                return {"d": "drop", "t": c["t"], "why": "echo_signal"}
            return {"d": "release", "t": c["t"], "why": "final"}
    rel = item.get("hold_release")
    if rel is not None and rel["reason"] == "playback_ended":
        return {"d": "release", "t": rel["t"], "why": "playback_ended"}
    if "hold_drop" in item:
        return {"d": "drop", "t": item["hold_drop"]["t"], "why": "recorded"}
    return {"d": "censored", "t": None, "why": "sequence ends"}


def whatif(rows: list[dict[str, Any]], out: list[str]) -> list[dict[str, Any]]:
    out.append("## Offline what-ifs: confirm_echo_psr and confirm_echo_drop_psr\n")
    out.append("Replayed from the v3 captures' per-check `psr` and verdicts (a would-be release is a check with verdict `caller`, `echo_signal` or `echo_signal_drop`). Limits: the recorded sequence of a turn stops at the decision made with 10 / 15. A lower `confirm_echo_psr` than 10 would hold some releases that were recorded as released, and the later checks that would decide them do not exist: those are counted as censored (unknown, between 'still held' and a later release). Higher thresholds and every `confirm_echo_drop_psr` are exact. A/Am: 'self-interruptions' = releases by a check (not at the playback's end) over the fix bursts; B, Be, F, C: drops of real callers and the release delay (check time less the onset in the injected audio, ms).\n")
    v3 = [r for r in rows if r["condition"] == "v3"]
    grid = []
    body = []
    for p in (8, 10, 12, 15):
        for d in (10, 15, 20, 25, None):
            row: dict[str, Any] = {"psr": p, "drop_psr": d}
            cells = [str(p), "null" if d is None else str(d)]
            for scn in ("A", "Am"):
                rs = [r for r in v3 if r["scenario"] == scn and "error" not in r]
                rel = cen = 0
                for r in rs:
                    for i in items_of(r):
                        if not i.get("held"):
                            continue
                        x = replay(i, p, d)
                        rel += x["d"] == "release" and x["why"] != "playback_ended"
                        cen += x["d"] == "censored"
                cells.append(f"{rel}/{len(rs)} (+{cen} cens.)")
                row[scn] = (rel, cen, len(rs))
            for scn in ("B", "Be", "C", "F"):
                rs = [r for r in v3 if r["scenario"] == scn and "error" not in r]
                drops = cen = 0
                delays: list[float] = []
                for r in rs:
                    if not r["ss"]:
                        drops += any(i.get("held") for i in items_of(r))  # held and never announced; no gate: not a drop
                        continue
                    its = [i for i in items_of(r) if i.get("held")]
                    if not its:
                        continue
                    x = replay(its[0], p, d)
                    if x["d"] == "drop":
                        drops += 1
                    elif x["d"] == "censored":
                        cen += 1
                    else:
                        delays.append(x["t"] - onset_ms(r))
                cells.append(f"{drops} drop, {cen} cens.; {lat(delays)}")
                row[scn] = (drops, cen, delays)
            grid.append(row)
            body.append(cells)
    out.append(table(["psr", "drop psr", "A self-int.", "Am self-int.", "B", "Be", "C", "F"], body))
    return grid


def setup(out: list[str]) -> dict[str, Any]:
    meta_path = V3 / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out.append("## Setup\n")
    for key in ("models", "pmset", "git", "config", "seed", "burst_model", "started"):
        if key in meta:
            v = meta[key]
            out.append(f"- **pmset -g** (at the start of the run):\n\n```\n{v}\n```\n" if key == "pmset" else f"- **{key}**: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}\n")
    return meta


def main() -> None:
    rows = load()
    out = ["# Hold measurement, v3 (issue #5)\n"]
    findings = RESULTS / "hold_v3_findings.md"
    if findings.exists():
        out.append(findings.read_text() + "\n")
    meta = setup(out)
    for scn, title in (("A", "A. Bursts, default leak model"), ("Am", "Am. Bursts, mild leak model")):
        if sel(rows, scn, "v3"):
            a_section(rows, out, scn, title)
    for scn, title in (("B", "B. Real barge-ins, no echo"), ("Be", "Be. Real barge-ins, steady echo")):
        if sel(rows, scn, "v3"):
            b_section(rows, out, scn, title)
    if sel(rows, "C", "v3"):
        c_section(rows, out)
    if any(r["scenario"] == "F" for r in rows):
        f_section(rows, out)
    e_section(rows, out)
    grid = whatif(rows, out)
    (RESULTS / "hold_v3.md").write_text("\n".join(out))
    (RESULTS / "hold_v3.json").write_text(json.dumps({"setup": meta, "whatif": grid, "rows": rows}, ensure_ascii=False, indent=1, default=str))
    print(f"wrote hold_v3.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()

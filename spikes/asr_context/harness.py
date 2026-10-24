"""ASR context-biasing harness (issue #1): every clip through POST /v1/audio/transcriptions with
and without a ``context`` form field, failure-path variants, scores and per-turn latency.

    uv run --locked --project spikes/asr_context python spikes/asr_context/harness.py run \
        --base-url http://127.0.0.1:8132 --out spikes/results/asr_context_stock.json [--limit 10]
    ... harness.py latency --base-url URL --out spikes/results/asr_context_latency.json
    ... harness.py huge    --base-url URL --out spikes/results/asr_context_huge.json
    ... harness.py report  spikes/results/asr_context_stock.json     # re-print the tables

Request: multipart ``file`` (WAV 16 kHz mono PCM16), ``model``, ``response_format=verbose_json``,
``language`` only with --language (the gateway omits it: auto-detect), and the context under
``--context-field`` (default ``context``). Modes (``--modes``):

  none      no context field (today's behaviour)
  context   terms.CONTEXT: the deployment's term list (~300 chars)
  context_noex  terms.CONTEXT_NOEX: context without the order-number format examples
  terms     terms.CONTEXT_TERMS: the bare comma-separated term list, no sentence
  empty     context="" (must behave as none)
  en_prompt terms.CONTEXT_EN: an English description of the call, no terms (other-language prompt)
  long1k    terms.CONTEXT_LONG: the term list padded to ~970 characters
  (huge10k  terms.CONTEXT_HUGE, 10 KiB: the ``huge`` subcommand, a few clips, records the backend's answer)

Metric definitions are in common.py (term hit rate, CER, non-term CER). ``run`` goes one request at
a time, the modes interleaved clip by clip, after 3 discarded warm-up requests. ``latency`` runs 1, 2
and 3 concurrent sessions (the gateway's asr.max_in_flight is 3), each session sending its turns back to
back; the per-turn latency is the client-side time from request to full response.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import statistics
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import httpx2
import numpy as np

import common as c
import terms as T

MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
CONTEXTS = {"none": None, "context": T.CONTEXT, "terms": T.CONTEXT_TERMS, "context_noex": T.CONTEXT_NOEX, "empty": "", "en_prompt": T.CONTEXT_EN,
            "long1k": T.CONTEXT_LONG, "huge10k": T.CONTEXT_HUGE}
DEFAULT_MODES = ["none", "context", "terms", "empty", "en_prompt", "long1k"]


class Client:
    def __init__(self, base_url: str, *, model: str, field: str, language: str | None, timeout: float = 60.0):
        self.http = httpx2.AsyncClient(base_url=base_url, timeout=timeout, trust_env=False)
        self.model, self.field, self.language = model, field, language

    async def aclose(self) -> None:
        await self.http.aclose()

    async def transcribe(self, wav: bytes, context: str | None) -> dict:
        form = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            form["language"] = self.language
        if context is not None:
            form[self.field] = context
        began = time.perf_counter()
        try:
            r = await self.http.post("/v1/audio/transcriptions", data=form,
                                     files={"file": ("audio.wav", wav, "audio/wav")})
            ms = (time.perf_counter() - began) * 1000
            if r.status_code != 200:
                return {"hyp": "", "ms": round(ms, 1), "status": r.status_code, "error": r.text[:300]}
            try:
                body = r.json()
                text = str(body.get("text", "")) if isinstance(body, dict) else ""
            except ValueError:
                text = r.text
            return {"hyp": text.strip(), "ms": round(ms, 1), "status": 200}
        except httpx2.HTTPError as error:
            return {"hyp": "", "ms": round((time.perf_counter() - began) * 1000, 1), "status": None,
                    "error": f"{type(error).__name__}: {error}"[:300]}


def load_wavs(rows: list[dict]) -> dict[str, bytes]:
    return {r["id"]: (c.DATA / "clips" / f"{r['id']}.wav").read_bytes() for r in rows}


def select(rows: list[dict], args) -> list[dict]:
    rows = [r for r in rows if (c.DATA / "clips" / f"{r['id']}.wav").exists()]
    if args.sets:
        rows = [r for r in rows if r["set"] in args.sets.split(",")]
    if args.ids:
        ids = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in ids]
    if args.limit:
        # a stratified few: round-robin over (set, degradation) so a short run touches every path
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["set"], r["degradation"])].append(r)
        out, k = [], 0
        while len(out) < args.limit and any(groups.values()):
            for g in list(groups.values()):
                if g and len(out) < args.limit:
                    out.append(g.pop(0))
            k += 1
        rows = out
    return rows


# ------------------------------------------------------------------------------------- scoring


def pooled(items: list[dict]) -> dict:
    s = defaultdict(int)
    for it in items:
        for k in ("ref_chars", "edits", "nt_ref_chars", "nt_edits", "terms", "hits"):
            s[k] += it[k]
    return {"n": len(items), "terms": s["terms"], "hits": s["hits"],
            "hit_rate": s["hits"] / s["terms"] if s["terms"] else None,
            "cer": s["edits"] / s["ref_chars"] if s["ref_chars"] else None,
            "nt_cer": s["nt_edits"] / s["nt_ref_chars"] if s["nt_ref_chars"] else None,
            "errors": 0}


def bootstrap_diff(a: list[dict], b: list[dict], key: str, n: int = 2000) -> list[float] | None:
    """95% paired-bootstrap CI of metric(b) - metric(a) over clips."""
    rnd = random.Random(0)
    vals = []
    for _ in range(n):
        idx = [rnd.randrange(len(a)) for _ in a]
        ma, mb = pooled([a[i] for i in idx])[key], pooled([b[i] for i in idx])[key]
        if ma is None or mb is None:
            return None
        vals.append(mb - ma)
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def summarize(results: dict) -> dict:
    rows = {r["id"]: r for r in results["rows"]}
    modes = results["modes"]
    scored: dict[str, dict[str, dict]] = {m: {} for m in modes}
    errors = defaultdict(int)
    for cid, per in results["clips"].items():
        r = rows[cid]
        for m in modes:
            o = per.get(m)
            if o is None:
                continue
            if o.get("status") != 200:
                errors[m] += 1
            scored[m][cid] = c.score(r["text"], o["hyp"], r["terms"]) | {"id": cid, "set": r["set"],
                                                                         "category": r["category"],
                                                                         "degradation": r["degradation"],
                                                                         "voice": r["voice"]}
    out: dict = {"by_mode": {}, "by_slice": {}, "diff_vs_none": {}, "errors": dict(errors),
                 "identical_to_none": {}}
    for m in modes:
        for scope, pred in (("tts", lambda s: s["set"] == "tts"), ("ascend", lambda s: s["set"] == "ascend")):
            out["by_mode"].setdefault(m, {})[scope] = pooled([s for s in scored[m].values() if pred(s)])
        out["by_mode"][m]["errors"] = errors[m]
        for dim in ("category", "degradation", "voice"):
            for val in sorted({s[dim] for s in scored[m].values() if s["set"] == "tts"}):
                out["by_slice"].setdefault(m, {}).setdefault(dim, {})[val] = pooled(
                    [s for s in scored[m].values() if s["set"] == "tts" and s[dim] == val])
        if m != "none" and "none" in modes:
            same = [cid for cid, per in results["clips"].items()
                    if m in per and "none" in per and per[m]["hyp"] == per["none"]["hyp"]]
            out["identical_to_none"][m] = f"{len(same)}/{sum(1 for p in results['clips'].values() if m in p)}"
            for scope in ("tts", "ascend"):
                ids = [cid for cid in scored[m] if cid in scored["none"] and scored[m][cid]["set"] == scope]
                if not ids:
                    continue
                a, b = [scored["none"][i] for i in ids], [scored[m][i] for i in ids]
                d = {}
                for key in ("hit_rate", "cer", "nt_cer"):
                    pa, pb = pooled(a)[key], pooled(b)[key]
                    d[key] = None if pa is None or pb is None else {
                        "diff": round(pb - pa, 4), "ci95": bootstrap_diff(a, b, key)}
                out["diff_vs_none"].setdefault(m, {})[scope] = d
    return out


def fmt(x, pct=True) -> str:
    return "-" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")


def print_report(results: dict, summary: dict) -> None:
    modes = results["modes"]
    print(f"\n== {results.get('label', '')}  {results['base_url']}  field={results['context_field']}  "
          f"clips={len(results['clips'])}")
    print("\nSynthetic corpus (terms in the reference)")
    print("| mode | clips | term hit | CER | non-term CER | errors | identical to none |")
    print("|---|---|---|---|---|---|---|")
    for m in modes:
        p = summary["by_mode"][m]["tts"]
        print(f"| {m} | {p['n']} | {fmt(p['hit_rate'])} ({p['hits']}/{p['terms']}) | {fmt(p['cer'])} | "
              f"{fmt(p['nt_cer'])} | {summary['by_mode'][m]['errors']} | {summary['identical_to_none'].get(m, '-')} |")
    if any(summary["by_mode"][m]["ascend"]["n"] for m in modes):
        print("\nASCEND human mixed speech (no term; CER = non-term CER)")
        print("| mode | clips | CER |")
        print("|---|---|---|")
        for m in modes:
            p = summary["by_mode"][m]["ascend"]
            print(f"| {m} | {p['n']} | {fmt(p['cer'])} |")
    for dim in ("category", "degradation", "voice"):
        vals = sorted({v for m in modes for v in summary["by_slice"].get(m, {}).get(dim, {})})
        if not vals:
            continue
        print(f"\nTerm hit rate by {dim}")
        print("| " + dim + " | " + " | ".join(modes) + " |")
        print("|---|" + "---|" * len(modes))
        for v in vals:
            cells = []
            for m in modes:
                p = summary["by_slice"][m][dim].get(v)
                cells.append("-" if not p else f"{fmt(p['hit_rate'])} (n={p['n']})")
            print(f"| {v} | " + " | ".join(cells) + " |")
    print("\nPaired difference vs none (95% bootstrap CI)")
    for m, d in summary["diff_vs_none"].items():
        for scope, k in d.items():
            print(f"  {m:10s} {scope:7s} " + "  ".join(
                f"{key}: {v['diff']:+.4f} {v['ci95']}" if v else f"{key}: -" for key, v in k.items()))


# ----------------------------------------------------------------------------------- subcommands


async def cmd_run(args) -> None:
    rows = select(c.load_manifest(), args)
    modes = args.modes.split(",")
    wavs = load_wavs(rows)
    client = Client(args.base_url, model=args.model, field=args.context_field, language=args.language)
    clips: dict[str, dict] = {}
    t0 = time.perf_counter()
    try:
        for r in rows[:3]:  # warm-up, discarded
            await client.transcribe(wavs[r["id"]], None)
        for i, r in enumerate(rows):
            clips[r["id"]] = {m: await client.transcribe(wavs[r["id"]], CONTEXTS[m]) for m in modes}
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(rows)}  {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    finally:
        await client.aclose()
    results = {"kind": "run", "label": args.label, "base_url": args.base_url, "model": args.model,
               "context_field": args.context_field, "language": args.language, "modes": modes,
               "context_lengths": {m: (None if CONTEXTS[m] is None else len(CONTEXTS[m])) for m in modes},
               "wall_s": round(time.perf_counter() - t0, 1), "rows": rows, "clips": clips}
    results["summary"] = summarize(results)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print_report(results, results["summary"])
    show_examples(results)


def show_examples(results: dict, n: int = 6) -> None:
    print("\nExamples (ref | none | context)")
    for cid, per in list(results["clips"].items())[:n]:
        r = next(x for x in results["rows"] if x["id"] == cid)
        print(f"  {cid} [{r['degradation']}] {r['text']} | {per['none']['hyp']} | {per.get('context', {}).get('hyp', '')}")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def cmd_latency(args) -> None:
    rows = select(c.load_manifest(), args)
    rows = [r for r in rows if r["set"] == "tts"]
    rows = rows[:: max(1, len(rows) // args.turns)][: args.turns]
    wavs = load_wavs(rows)
    client = Client(args.base_url, model=args.model, field=args.context_field, language=args.language)
    cells: dict[str, dict] = {}
    lmodes = args.modes.split(",")
    try:
        for r in rows[:3]:
            await client.transcribe(wavs[r["id"]], None)

        async def session(turns: list[dict], context: str | None, sink: list[float], errs: list[str]) -> None:
            for r in turns:  # a session sends its turns back to back
                o = await client.transcribe(wavs[r["id"]], context)
                if o["status"] == 200:
                    sink.append(o["ms"])
                else:
                    errs.append(str(o.get("error")))

        for rnd in range(args.rounds):
            for n in (1, 2, 3):
                for mode in (lmodes if rnd % 2 == 0 else lmodes[::-1]):
                    sink: list[float] = []
                    errs: list[str] = []
                    shuffled = random.Random(rnd * 10 + n).sample(rows, len(rows))
                    parts = [shuffled[k::n] for k in range(n)]
                    t0 = time.perf_counter()
                    await asyncio.gather(*(session(p, CONTEXTS[mode], sink, errs) for p in parts))
                    cell = cells.setdefault(f"{n}/{mode}", {"ms": [], "errors": [], "wall_s": []})
                    cell["ms"] += sink
                    cell["errors"] += errs
                    cell["wall_s"].append(round(time.perf_counter() - t0, 2))
    finally:
        await client.aclose()
    summary = {}
    print("\nPer-turn ASR latency (client side, ms)")
    print("| sessions | mode | turns | mean | p50 | p95 | max | errors |")
    print("|---|---|---|---|---|---|---|---|")
    for key, cell in cells.items():
        ms = cell["ms"]
        s = {"turns": len(ms), "mean": round(statistics.fmean(ms), 1), "p50": round(pct(ms, 0.5), 1),
             "p95": round(pct(ms, 0.95), 1), "max": round(max(ms), 1), "errors": len(cell["errors"])}
        summary[key] = s
        n, mode = key.split("/")
        print(f"| {n} | {mode} | {s['turns']} | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} | {s['errors']} |")
    Path(args.out).write_text(json.dumps({
        "kind": "latency", "label": args.label, "base_url": args.base_url, "context_field": args.context_field,
        "turns_per_round": len(rows), "rounds": args.rounds, "clips": [r["id"] for r in rows],
        "summary": summary, "raw": cells}, indent=1))


async def cmd_huge(args) -> None:
    """A 10 KiB context sent straight to the backend: what does it answer, and is it still alive?"""
    rows = select(c.load_manifest(), args)[: args.turns]
    wavs = load_wavs(rows)
    client = Client(args.base_url, model=args.model, field=args.context_field, language=args.language, timeout=args.timeout)
    out = {"kind": "huge", "label": args.label, "base_url": args.base_url, "context_field": args.context_field,
           "context_chars": len(T.CONTEXT_HUGE), "context_bytes": len(T.CONTEXT_HUGE.encode()), "requests": []}
    try:
        base = await client.transcribe(wavs[rows[0]["id"]], None)
        out["baseline_before"] = base
        for r in rows:
            plain = await client.transcribe(wavs[r["id"]], None)
            huge = await client.transcribe(wavs[r["id"]], T.CONTEXT_HUGE)
            out["requests"].append({"id": r["id"], "text": r["text"], "none": plain, "huge10k": huge})
            print(f"{r['id']}: none {plain['ms']} ms {plain['hyp']!r} | huge {huge['status']} {huge['ms']} ms "
                  f"{(huge['hyp'] or huge.get('error', ''))[:80]!r}")
        after = await client.transcribe(wavs[rows[0]["id"]], None)
        out["baseline_after"] = after
        try:
            alive = await client.http.get("/v1/models")
            out["models_status_after"] = alive.status_code
        except httpx2.HTTPError as error:
            out["models_status_after"] = repr(error)
        print("after: plain request", after["status"], after["ms"], "ms; /v1/models", out["models_status_after"])
    finally:
        await client.aclose()
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_report(args) -> None:
    results = json.loads(Path(args.path).read_text())
    if results["kind"] != "run":
        print(json.dumps(results["summary"], indent=1))
        return
    print_report(results, summarize(results))
    show_examples(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default=MODEL)
        p.add_argument("--context-field", default="context")
        p.add_argument("--language", default=None, help="omitted by default, as the gateway does")
        p.add_argument("--sets", default="", help="tts,ascend (default: both)")
        p.add_argument("--ids", default="")
        p.add_argument("--limit", type=int, default=0, help="a stratified subset of this many clips")
        p.add_argument("--label", default="")

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p = sub.add_parser("latency")
    common(p)
    p.add_argument("--turns", type=int, default=30, help="turns per cell and round")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--modes", default="none,context,long1k")
    p = sub.add_parser("huge")
    common(p)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120.0)
    p = sub.add_parser("report")
    p.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "report":
        cmd_report(a)
    else:
        asyncio.run({"run": cmd_run, "latency": cmd_latency, "huge": cmd_huge}[a.cmd](a))


if __name__ == "__main__":
    main()

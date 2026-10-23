"""Print the B2 report tables (markdown) from spikes/results/s3_*.json, s4_*.json and s9_*.json.

    uv run --locked --project spikes/audio python spikes/audio/b2_summary.py [section ...]
"""

from __future__ import annotations

import json
import statistics
import sys

import b2_common as c


def load(name: str) -> dict | None:
    p = c.RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=0):
    if x is None:
        return "–"
    return f"{x:.{nd}f}"


def pp(s: dict, nd=0) -> str:
    """p50 / p95 from a stats dict."""
    if not s or not s.get("n"):
        return "–"
    return f"{f(s['p50'], nd)} / {f(s['p95'], nd)}"



def sweep_measured(d: dict, tag: str = "") -> tuple[list, str]:
    """Measured sweep requests, minus the ones excluded by s3_sweep_check.py (another process used the GPU
    briefly), plus their re-runs from s3_sweep_topup.json. Default-model sweep only."""
    ok = lambda r: r.get("status") == 200 and not r.get("error") and r.get("ttfc_ms") is not None  # noqa: E731
    recs = d["records"]
    pert = load("s3_sweep_perturbation.json") if not tag else None
    excluded = {x["k"] for x in pert["replace"]} if pert else set()
    out = [r for k, r in enumerate(recs) if r["phase"] == "measured" and ok(r) and k not in excluded]
    note = "no exclusions"
    if pert:
        top = load("s3_sweep_topup.json")
        added = [r for r in (top or {}).get("records", []) if r["phase"] == "topup" and ok(r)]
        out += added
        note = (f"{len(excluded)} requests in {pert['window_local'][0][11:19]}-{pert['window_local'][1][11:19]} "
                f"excluded, {len(added)} re-run (s3_sweep_topup.json)")
    return out, note

def sweep(tag: str = "") -> None:
    d = load(f"s3_sweep{tag}.json")
    if not d:
        print(f"s3_sweep{tag}.json missing")
        return
    recs, note = sweep_measured(d, tag)
    print(f"\n### Sweep {tag or '(default model)'}: {d['conditions']['model']} — n={len(recs)} measured requests "
          f"({note}), errors={sum(1 for r in d['records'] if r.get('status') != 200 or r.get('error'))}")
    print("\nTTFC p50 / p95 (ms), both speakers pooled (n = 40 per cell unless noted); RTF p50 = total time / audio")
    print("\n| interval (chunk) | length | zh | en | mix | RTF p50 | steady RTF p50 | stalls |")
    print("|---|---|---|---|---|---|---|---|")
    for iv in d["conditions"]["intervals"]:
        for ln in d["conditions"]["lengths"]:
            row = []
            sel_all = [r for r in recs if r["interval"] == iv and r["length"] == ln]
            for lang in d["conditions"]["langs"]:
                sel = [r["ttfc_ms"] for r in sel_all if r["lang"] == lang]
                row.append(pp(c.stats(sel)))
            rtf = c.stats([r["rtf"] for r in sel_all])
            srtf = c.stats([r["steady_rtf"] for r in sel_all if r.get("steady_rtf")])
            stalls = sum(1 for r in sel_all if (r.get("min_slack_ms") or 0) < 0)
            print(f"| {iv} s | {ln} | " + " | ".join(row) + f" | {f(rtf.get('p50'), 3)} | {f(srtf.get('p50'), 3)} | {stalls}/{len(sel_all)} |")
    print("\nBy interval (all cells pooled):")
    print("\n| interval | TTFC p50 / p95 / max | total ms p50 | RTF p50 / p95 | chunk audio (s) | max gap between chunks p95 (ms) | min slack p50 (ms) |")
    print("|---|---|---|---|---|---|---|")
    for iv in d["conditions"]["intervals"]:
        sel = [r for r in recs if r["interval"] == iv]
        t = c.stats([r["ttfc_ms"] for r in sel])
        sizes = [ch[1] for r in sel for ch in r["chunks"][:-1]]
        mode = statistics.mode(sizes) / 48000 if sizes else None
        print(f"| {iv} | {f(t['p50'])} / {f(t['p95'])} / {f(t['max'])} | {f(c.stats([r['total_ms'] for r in sel])['p50'])} | "
              f"{pp(c.stats([r['rtf'] for r in sel]), 3)} | {f(mode, 2)} | {f(c.stats([r['max_gap_ms'] for r in sel if r.get('max_gap_ms')])['p95'])} | "
              f"{f(c.stats([r['min_slack_ms'] for r in sel if r.get('min_slack_ms') is not None])['p50'])} |")
    print("\nBy speaker:")
    for spk in d["conditions"]["speakers"]:
        for iv in d["conditions"]["intervals"]:
            sel = [r for r in recs if r["speaker"] == spk and r["interval"] == iv]
            print(f"- {spk} @ {iv}: TTFC {pp(c.stats([r['ttfc_ms'] for r in sel]))}, RTF p50 {f(c.stats([r['rtf'] for r in sel])['p50'], 3)}")
    first = [r for r in d["records"] if r["phase"] == "first_in_cell"]
    print(f"\nFirst request per cell (after 3 warmups; discarded): TTFC {pp(c.stats([r['ttfc_ms'] for r in first]))}, "
          f"max {f(c.stats([r['ttfc_ms'] for r in first])['max'])}")
    print(f"Warmup: {[(w['lang'], w['ttfc_ms']) for w in d['warmup']]}")
    print(f"Load average per round (before, after): first {d['round_loadavg1'][0]}, last {d['round_loadavg1'][-1]}, "
          f"max {max(max(x) for x in d['round_loadavg1'])}")
    print(f"Server log during run (non-access lines): {d['log_during_run'][:5]}")


def overhead() -> None:
    s, i = load("s3_sweep.json"), load("s3_inproc.json")
    if not (s and i):
        print("sweep or inproc missing")
        return
    srecs = [r for r in sweep_measured(s)[0] if r["speaker"] == "vivian"]
    irecs = [r for r in i["records"] if r["phase"] == "measured"]
    i_long = load("s3_inproc_long.json")
    if i_long:
        irecs += [r for r in i_long["records"] if r["phase"] == "measured"]
    print("\n### HTTP overhead on the first chunk (speaker vivian; server TTFC − in-process TTFC, p50 per cell)")
    print("\n| interval | length | zh server / in-proc (ms) | en | mix | Δp50 zh / en / mix |")
    print("|---|---|---|---|---|---|")
    deltas = []
    for iv in s["conditions"]["intervals"]:
        for ln in s["conditions"]["lengths"]:
            cells, ds = [], []
            for lang in s["conditions"]["langs"]:
                a = c.stats([r["ttfc_ms"] for r in srecs if (r["interval"], r["length"], r["lang"]) == (iv, ln, lang)])
                b = c.stats([r["ttfc_ms"] for r in irecs if (r["interval"], r["length"], r["lang"]) == (iv, ln, lang)])
                cells.append(f"{f(a.get('p50'))} / {f(b.get('p50'))}")
                if a.get("n") and b.get("n"):
                    ds.append(a["p50"] - b["p50"])
                    deltas.append((iv, ln, lang, a["p50"] - b["p50"], a["p95"] - b["p95"]))
            print(f"| {iv} | {ln} | " + " | ".join(cells) + " | " + " / ".join(f(x, 1) for x in ds) + " |")
    dp50 = [x[3] for x in deltas]
    dp95 = [x[4] for x in deltas]
    print(f"\nΔp50 over the {len(dp50)} cells: median {f(statistics.median(dp50), 1)} ms, min {f(min(dp50), 1)}, "
          f"max {f(max(dp50), 1)}; Δp95 median {f(statistics.median(dp95), 1)}, max {f(max(dp95), 1)}")
    pc = c.stats([r["pcm_convert_first_ms"] for r in irecs])
    print(f"In-process PCM conversion of the first chunk (audio_write): {pp(pc, 2)} ms")
    rt_s = c.stats([r["rtf"] for r in srecs])
    rt_i = c.stats([r["rtf"] for r in irecs])
    print(f"RTF p50 server {f(rt_s['p50'], 3)} vs in-process {f(rt_i['p50'], 3)}")
    print(f"In-process load_model {i['load_model_ms']} ms; first generate {i['cold_first_generate']}")


def simple(name: str) -> None:
    d = load(name)
    print(f"\n### {name}")
    if not d:
        print("missing")
        return
    print(json.dumps(d.get("summary"), ensure_ascii=False, indent=1)[:6000])


def cancel() -> None:
    d = load("s3_cancel.json")
    if not d:
        print("s3_cancel.json missing")
        return
    print("\n### Cancellation (traced server). Times in ms relative to the client's close of the old stream")
    print("\n| interval | new TTFC after cancel p50/p95 | control TTFC p50/p95 | old gen. end after close p50/p95/max | "
          "chunks emitted after close (max) | disconnect seen (handle.cancel) p50 |")
    print("|---|---|---|---|---|---|")
    for iv in sorted({t["interval"] for t in d["trials"]}):
        tr = [t for t in d["trials"] if t["interval"] == iv and t["kind"] == "trial"]
        co = [t for t in d["trials"] if t["interval"] == iv and t["kind"] == "control"]
        ge = c.stats([t.get("old_generation_end_ms_after_close") for t in tr])
        print(f"| {iv} | {pp(c.stats([t['new_ttfc_ms'] for t in tr]))} | {pp(c.stats([t['new_ttfc_ms'] for t in co]))} | "
              f"{f(ge.get('p50'))} / {f(ge.get('p95'))} / {f(ge.get('max'))} | "
              f"{f(c.stats([t.get('old_chunks_after_close') for t in tr]).get('max'))} | "
              f"{f(c.stats([t.get('old_handle_cancel_ms_after_close') for t in tr]).get('p50'), 1)} |")
    errs = [t for t in d["trials"] if t.get("new_error") or t.get("trace_problem")]
    print(f"problems: {len(errs)}")



def cold() -> None:
    d = load("s3_cold.json")
    if not d:
        print("s3_cold.json missing")
        return
    sm = d["summary"]
    print(f"\n### Cold start (n={len(d['runs'])} fresh servers)")
    print(f"- process start to /v1/models 200: {pp(sm['seconds_to_ready'], 1)} s")
    print(f"- first request (lazy load inside): TTFC {pp(sm['first_ttfc_ms'])} ms, total {pp(sm['first_total_ms'])} ms")
    print(f"- second request: TTFC {pp(sm['second_ttfc_ms'])} ms")
    print(f"- footprint after 2 requests: {[r['footprint_after'] for r in d['runs']]}")


def stream() -> None:
    d = load("s3_stream.json")
    if not d:
        print("s3_stream.json missing")
        return
    print("\n### Streaming check (zh, vivian)")
    print("\n| text | mode | TTFC p50 (ms) | total p50 (ms) | chunks | bytes/chunk | audio s |")
    print("|---|---|---|---|---|---|---|")
    keys = sorted({(r["length"], r["stream"], str(r["streaming_interval"])) for r in d["runs"]}, key=str)
    for ln, st, iv in keys:
        sel = [r for r in d["runs"] if (r["length"], r["stream"], str(r["streaming_interval"])) == (ln, st, iv)]
        sizes = sorted({ch[1] for r in sel for ch in r["chunks"]})
        print(f"| {ln} | stream={st} interval={iv} | {f(c.stats([r['ttfc_ms'] for r in sel])['p50'])} | "
              f"{f(c.stats([r['total_ms'] for r in sel])['p50'])} | {f(c.stats([r['n_chunks'] for r in sel])['p50'])} | "
              f"{sizes[:3]}{'…' if len(sizes) > 3 else ''} | {f(c.stats([r['audio_s'] for r in sel])['p50'], 2)} |")
    print(f"content-type {d['runs'][0]['content_type']}, transfer-encoding {d['runs'][0]['transfer_encoding']}, "
          f"even byte counts: {all(r['even_bytes'] for r in d['runs'])}")


def concurrency() -> None:
    d = load("s3_concurrency.json")
    if not d:
        print("s3_concurrency.json missing")
        return
    print("\n### Concurrency (zh medium, vivian, 0.32): TTFC by arrival rank, p50 / p95 ms")
    for k, v in d["summary"].items():
        print(f"- {k}: " + ", ".join(f"{rk.split('_')[0]} {pp(st)}" for rk, st in v.items()))


def pinning() -> None:
    d = load("s3_pinning.json")
    if not d:
        print("s3_pinning.json missing")
        return
    sm = d["sequence_summary"]
    print("\n### Warmup and pinning (traced server)")
    print(f"- warmup TTFC {d['warmup']['ttfc_ms']} ms; request #2 TTFC {sm['request_2']['ttfc_ms']}, last request TTFC "
          f"{list(sm.values())[1]['ttfc_ms']}; all {pp(sm['ttfc_ms'])}; first 10 {pp(sm['first10_ttfc_ms'])}; last 10 {pp(sm['last10_ttfc_ms'])}")
    print(f"- footprint: before warmup {d['footprint_before_warmup']}, after {d['footprint_after_warmup']}, after sequence {d['footprint_after_sequence']}")
    print(f"- swap before {d['swap_before']} / after {d['swap_after']}")
    print(f"- model.load trace lines: {d['model_load_trace_lines']}")
    print(f"- stock load lines: {d['load_lines_stock']}")
    for g in d["idle_gaps"]:
        print(f"- idle {g['idle_s']} s rep {g['rep']}: TTFC {g['after_idle_ttfc_ms']} then {g['next_ttfc_ms']} ms (total {g['after_idle_total_ms']} / {g['next_total_ms']})")


def speakers() -> None:
    d = load("s3_speakers.json")
    if not d:
        print("s3_speakers.json missing")
        return
    print("\n### Speakers (10 sentences x 2 runs each; ASR auto language)")
    print("\n| text | speaker | error rate | TTFC p50 | units/voiced s p50 (min) | ASR language |")
    print("|---|---|---|---|---|---|")
    for k, v in d["summary"].items():
        lang, spk = k.split("|")
        print(f"| {lang} | {spk} | {f(v['error']['rate'] * 100, 1)}% ({v['error']['edits']}/{v['error']['ref_tokens']}) | "
              f"{f(v['ttfc_ms']['p50'])} | {f(v['units_per_voiced_s']['p50'], 2)} ({f(v['units_per_voiced_s']['min'], 2)}) | {v['asr_language_counts']} |")


def guardrail() -> None:
    d = load("s3_guardrail.json")
    if not d:
        print("s3_guardrail.json missing")
        return
    sm = d["summary"]
    print("\n### Quality guardrail")
    print(f"- runs {sm['n_runs']}; rate/median {pp(sm['ratio_to_median'], 3)} min {sm['ratio_to_median']['min']} max {sm['ratio_to_median']['max']}")
    print(f"- histogram {sm['ratio_histogram']}")
    print(f"- outliers < 0.6x: {sm['outliers_below_0.6']}")
    print(f"- rate by lang: {sm['rate_by_lang']}")
    print(f"- back-transcribed {sm['back_transcribed']}")
    print(f"- TTFC {pp(sm['ttfc_ms'])}")
    for si, v in sm["per_sentence"].items():
        print(f"  - s{si} {v['lang']}: rate {pp(v['rate'], 2)} min {v['rate']['min']} max {v['rate']['max']}")


def kokoro() -> None:
    d = load("s3_kokoro.json")
    if not d:
        print("s3_kokoro.json missing")
        return
    print("\n### D13: Kokoro vs Qwen3-TTS (20 sentences per set, ASR back-transcription, auto language)")
    print("\n| set | engine | error rate | sentences with errors | TTFC p50 / p95 (ms) | total p50 (ms) | RTF p50 | English words recovered |")
    print("|---|---|---|---|---|---|---|---|")
    for lang in ("zh", "en", "mix"):
        for eng in ("kokoro", "qwen"):
            v = d["summary"][f"{eng}|{lang}"]
            rec = v.get("en_word_recall")
            print(f"| {lang} | {eng} | {f(v['error']['rate'] * 100, 1)}% | {v['sentences_with_errors']}/20 | {pp(v['ttfc_ms'])} | "
                  f"{f(v['total_ms']['p50'])} | {f(v['rtf']['p50'], 3)} | {str(rec['found']) + '/' + str(rec['total']) if rec else '–'} |")
    print(f"Kokoro cold: {d['kokoro_cold']}")
    print(f"Kokoro cold log: {d['kokoro_cold_log'][:8]}")


def s4_latency(tag: str = "") -> None:
    d, i = load(f"s4_latency{tag}.json"), load("s4_inproc.json")
    if not d:
        print("s4_latency missing")
        return
    print(f"\n### S4 latency {d['model']['repo']} (ms, client side, p50 / p95; n = 20 per cell)")
    print("\n| clip | zh auto | zh lang | en auto | en lang | mix auto | mix lang | in-process p50 (all sets, lang) |")
    print("|---|---|---|---|---|---|---|---|")
    for T_ in (2, 5, 10):
        row = [pp(d["summary"][f"{s}|{T_}s|{g}"]["ms"]) for s in ("zh", "en", "mix") for g in ("auto", "lang")]
        ip = "–"
        if i:
            ip = f(c.stats([r["ms"] for r in i["records"] if r["seconds"] == T_ and r["language_given"]])["p50"])
        print(f"| {T_} s | " + " | ".join(row) + f" | {ip} |")
    if i:
        for T_ in (2, 5, 10):
            for g in (False, True):
                a = c.stats([r["ms"] for r in d["records"] if r["seconds"] == T_ and r["language_given"] == g])
                b = c.stats([r["ms"] for r in i["records"] if r["seconds"] == T_ and r["language_given"] == g])
                print(f"- {T_} s lang={g}: HTTP p50 {a['p50']} vs in-process p50 {b['p50']} -> Δ {round(a['p50'] - b['p50'], 1)} ms")
    print(f"cold first request {d.get('cold_first_request')} ms; warmup {d['warmup']}")


def s4_accuracy(tag: str = "") -> None:
    d = load(f"s4_accuracy{tag}.json")
    if not d:
        print("s4_accuracy missing")
        return
    sm = d["summary"]
    print(f"\n### S4 accuracy {d['model']['repo']} (error rate %, no-number rows / number rows / all)")
    print("\n| set (metric) | language | wideband | telephone | Δ telephone (no-number rows) |")
    print("|---|---|---|---|---|")
    for s in ("zh", "en", "mix"):
        for g in ("auto", "lang"):
            wb, tel = sm[f"{s}|16k|{g}"], sm[f"{s}|tel16k|{g}"]
            fmt = lambda v: f"{f(v['no_number']['rate'] * 100 if v['no_number']['rate'] is not None else None, 1)} / {f(v['has_number']['rate'] * 100 if v['has_number']['rate'] is not None else None, 1)} / {f(v['all']['rate'] * 100, 1)}"  # noqa: E731
            dl = sm[f"{s}|telephone_delta|{g}"]["no_number"]
            print(f"| {s} ({wb['metric']}) | {g} | {fmt(wb)} | {fmt(tel)} | {f(dl * 100 if dl is not None else None, 1)} |")
    print("language detected (auto, wideband): " + "; ".join(f"{s}: {sm[f'{s}|16k|auto']['language_detected']}" for s in ("zh", "en", "mix")))


def s4_shape() -> None:
    d = load("s4_shape.json")
    if not d:
        print("s4_shape.json missing")
        return
    print("\n### S4 output shape")
    for p in d["probes"]:
        js = p.get("json")
        body = json.dumps(js, ensure_ascii=False)[:300] if js is not None else p.get("body", "")[:200]
        print(f"- {p['id'][:40]} fmt={p.get('response_format', 'verbose_json')} lang={p.get('language')} "
              f"{p.get('seconds', '')} -> {p['status']} {p.get('content_type')}: {body}")


def main() -> None:
    sections = sys.argv[1:] or ["sweep", "overhead", "cancel"]
    funcs = {"overhead": overhead, "cancel": cancel, "cold": cold, "stream": stream, "concurrency": concurrency,
             "pinning": pinning, "speakers": speakers, "guardrail": guardrail, "kokoro": kokoro,
             "s4_latency": s4_latency, "s4_accuracy": s4_accuracy, "s4_shape": s4_shape}
    for s in sections:
        name, _, tag = s.partition(":")
        if name == "sweep":
            sweep(tag)
        elif name in ("s4_latency", "s4_accuracy"):
            funcs[name](tag)
        elif name in funcs:
            funcs[name]()
        else:
            simple(s)


if __name__ == "__main__":
    main()

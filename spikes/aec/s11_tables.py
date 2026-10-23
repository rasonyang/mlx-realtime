"""Print the S11 part 2 report tables (markdown) from spikes/results/s11_*.json. Throwaway.

    cd spikes/aec && uv run python s11_tables.py [core|timeline|slices|dt|cpu|cer ...]
"""

from __future__ import annotations

import s11_common as c

import json  # noqa: E402
import sys  # noqa: E402


def load(name: str) -> dict | None:
    p = c.RESULTS / f"s11_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=1, pct=False):
    if x is None:
        return "–"
    if pct:
        return f"{100 * x:.{nd}f}%"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def core_table(d: dict, title: str) -> None:
    print(f"\n### {title}\n")
    print("| Variant | ERLE_ss p50 / p10 | conv. 20 dB ≤ 1 s: start / change | ERL+ERLE_ss ≥ 27 | Silero after 1 s: p50 / p90 / calls ≤ 1% "
          "| false starts 150 ms after 1 s (calls) | 400 ms | 1st s of playbacks rate p50 | after change 2 s rate p50 | C1 calls | C3 calls | CPU ms/s |")
    print("|" + "---|" * 13)
    for v, s in d["variants"].items():
        if "erle_ss_db" not in s:
            print(f"| {v} | no results: {s.get('errors')} |")
            continue
        fs, fs4 = s["false_starts_150"], s["false_starts_400"]
        print(f"| {v} | {f(s['erle_ss_db']['p50'])} / {f(s['erle_ss_db']['p10'])} | "
              f"{f(s['conv20_s']['frac_le_1s'], 0, True)} / {f(s['conv20_after_change_s']['frac_le_1s'], 0, True)} | "
              f"{f(s['erl_plus_erle_ss_ge_27'], 0, True)} | "
              f"{f(s['silero_rate_after_1s']['p50'], 2, True)} / {f(s['silero_rate_after_1s']['p90'], 2, True)} / "
              f"{f(s['silero_rate_after_1s']['frac_calls_le_1pct'], 0, True)} | {fs['after_1s']} ({fs['calls_with_any_after_1s']}) | "
              f"{fs4['after_1s']} ({fs4['calls_with_any_after_1s']}) | {f(s['silero_rate_first_1s_of_playbacks']['p50'], 2, True)} | "
              f"{f(s['silero_rate_change_first_2s']['p50'], 2, True)} | {s['criteria']['c1_echo_only_calls_pass']} | "
              f"{s['criteria']['c3_convergence_calls_pass']} | {f(s['cpu']['engine_ms_per_audio_s_p50'], 2)} |")


def by_table(d: dict, key: str, variants: list[str]) -> None:
    print(f"\n#### Silero rate after 1 s (p50, calls ≤ 1 %) and false starts (150 ms) by {key}\n")
    groups = sorted({g for v in variants if v in d["variants"] for g in d["variants"][v].get(key, {})},
                    key=lambda g: int("".join(ch for ch in g if ch.isdigit())))
    print("| Variant | " + " | ".join(groups) + " |")
    print("|---|" + "---|" * len(groups))
    for v in variants:
        if v not in d["variants"]:
            continue
        row = []
        for g in groups:
            s = d["variants"][v][key].get(g)
            if not s or "silero_rate_after_1s" not in s:
                row.append("–")
                continue
            row.append(f"{f(s['silero_rate_after_1s']['p50'], 2, True)}, {f(s['silero_rate_after_1s']['frac_calls_le_1pct'], 0, True)}, "
                       f"{s['false_starts_150']['after_1s']} | ERLE {f(s['erle_ss_db']['p50'])}")
        print(f"| {v} | " + " | ".join(row) + " |")


def near_table(d: dict) -> None:
    print("\n#### Near end and double talk (core grid)\n")
    print("| Variant | near single talk gain p50 / worst-utt p10 (dB) | DT SDR out / in p50 (dB) | post-DT ERLE drop p50 / max (dB) | "
          "false starts since playback onset p50 / p90 / max (s) |")
    print("|---|---|---|---|---|")
    for v, s in d["variants"].items():
        if "near_single_talk_gain_db" not in s:
            continue
        so = s["false_starts_150"]["since_playback_onset_s"]
        print(f"| {v} | {f(s['near_single_talk_gain_db']['p50'], 2)} / {f(s['near_single_talk_gain_db']['worst_utt_p10'], 2)} | "
              f"{f(s['dt_sdr_db']['out_p50'])} / {f(s['dt_sdr_db']['in_p50'])} | {f(s['post_dt_drop_db']['p50'])} / "
              f"{f(s['post_dt_drop_db']['max'])} | {f(so['p50'], 2)} / {f(so['p90'], 2)} / {f(so['max'], 2)} (n={so['n']}) |")


def curve_table(d: dict) -> None:
    bins = ["0-0.5", "0.5-1", "1-2", "2-3", "3-5", "5-10", "10-15", "15+"]
    print("\n#### ERLE over time: median over calls, by far-end-active time since the onset | since the path change (dB)\n")
    print("| Variant | " + " | ".join(bins) + " | conv20 p50 / p90 (s) | after change: " + " | ".join(bins[:6]) + " | conv20 p50 / p90 (s) |")
    print("|---|" + "---|" * (len(bins) + 1) + "---|" * 7)
    for v, s in d["variants"].items():
        if "erle_curve_p50" not in s:
            continue
        a = s["erle_curve_p50"]
        b = (s.get("erle_curve_after_change_p50") or [None] * 8)[:6]
        print(f"| {v} | " + " | ".join(f(x) for x in a) + f" | {f(s['conv20_s']['p50'], 2)} / {f(s['conv20_s']['p90'], 2)} | "
              + " | ".join(f(x) for x in b) + f" | {f(s['conv20_after_change_s']['p50'], 2)} / {f(s['conv20_after_change_s']['p90'], 2)} |")


def guard_table(d: dict, variants: list[str] | None = None) -> None:
    print("\n#### False speech_started left after an onset guard (from the modelled playback start), all calls\n")
    gs = ["0ms", "500ms", "1000ms", "1500ms", "2000ms", "2500ms", "3000ms", "4000ms"]
    print("| Variant | min_speech | " + " | ".join(gs) + " |")
    print("|---|---|" + "---|" * len(gs))
    for v, s in d["variants"].items():
        if variants and v not in variants or "false_starts_vs_echo_guard" not in s:
            continue
        for ms in ("150", "400"):
            g = s["false_starts_vs_echo_guard"][ms]
            print(f"| {v} | {ms} | " + " | ".join(f"{g[k]['false_starts']} ({g[k]['calls']})" for k in gs) + " |")


def dt_table(d: dict) -> None:
    print("\n### Double-talk calls (32 calls, 320 near-end utterances; D 120 ms; paths m1-m8)\n")
    print("| Variant | ERLE_ss p50 | spectral SDR in DT: out / in p50 (dB) | spectral SDR near-only: out p50 (dB) | "
          "post-DT ERLE drop p50 / p90 / max (dB) | Silero after 1 s p50 | false starts 150 ms |")
    print("|---|---|---|---|---|---|---|")
    for v, s in d["variants"].items():
        if "erle_ss_db" not in s:
            print(f"| {v} | no results |")
            continue
        sp = s.get("spec", {})
        print(f"| {v} | {f(s['erle_ss_db']['p50'])} | {f(sp.get('dt_out_p50'))} / {f(sp.get('dt_in_p50'))} | "
              f"{f(sp.get('st_out_p50'))} | {f(s['post_dt_drop_db']['p50'])} / {f(s['post_dt_drop_db']['p90'])} / "
              f"{f(s['post_dt_drop_db']['max'])} | {f(s['silero_rate_after_1s']['p50'], 2, True)} | "
              f"{s['false_starts_150']['total']} |")


def timeline_table(d: dict) -> None:
    print("\n### Timeline sensitivity (m4/m5, ERL 10; D 20/120/300 ms pooled per eps)\n")
    eps_keys = ["eps-200", "eps-50", "eps+0", "eps+50", "eps+200", "step-200", "step-50", "step+50", "step+200"]
    print("| Variant | " + " | ".join(eps_keys) + " |")
    print("|---|" + "---|" * len(eps_keys))
    for v, s in d["variants"].items():
        row = []
        for k in eps_keys:
            g = s.get("by_eps", {}).get(k)
            if not g or "erle_ss_db" not in g:
                row.append("–")
                continue
            row.append(f"ERLE {f(g['erle_ss_db']['p50'])}; S1 {f(g['silero_rate_after_1s']['p50'], 1, True)}; "
                       f"S2 {f(g['silero_rate_after_2s']['p50'], 1, True)}; fs2 {g['false_starts_150']['after_2s']}")
        print(f"| {v} | " + " | ".join(row) + " |")
    print("\nS1/S2 = Silero rate after the first 1 s / 2 s of far-end speech (p50 over the 6 or 2 calls); "
          "fs2 = false speech_started (150 ms) after 2 s, summed.")


def timeline_matrix(d: dict) -> None:
    print("\n#### Timeline per call (m4 | m5): Silero rate after 2 s of far-end speech, false starts (150 ms) after 2 s, ERLE_ss\n")
    cols = [(dd, e) for e in ("-200", "-50", "+0", "+50", "+200") for dd in ("20", "120", "300")]
    cols += [("120", f"step{e}") for e in ("-200", "-50", "+50", "+200")]
    print("| Variant | " + " | ".join(f"D{dd} {e if e.startswith('step') else 'eps' + e}" for dd, e in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for v, s in d["variants"].items():
        bc = s.get("by_cond", {})
        row = []
        for dd, e in cols:
            cells = []
            for p in ("m4", "m5"):
                cid = f"tl_{p}_d{dd}_{e if e.startswith('step') else 'eps' + e}"
                b = bc.get(cid)
                if not b or "erle_ss" not in b:
                    cells.append("–")
                    continue
                cells.append(f"{f(b['silero_after_2s'], 1, True)},{b['false_starts_150_after_2s']},{f(b['erle_ss'], 0)}")
            row.append(" / ".join(cells))
        print(f"| {v} | " + " | ".join(row) + " |")


def slices_table(d: dict) -> None:
    print("\n### Slices (per call)\n")
    print("| Variant | " + " | ".join(sorted(next(iter(d["variants"].values()))["by_cond"])) + " |")
    conds = sorted(next(iter(d["variants"].values()))["by_cond"])
    print("|---|" + "---|" * len(conds))
    for v, s in d["variants"].items():
        row = []
        for k in conds:
            b = s["by_cond"].get(k, {})
            if "erle_ss" not in b:
                row.append(str(b.get("error", "–"))[:30])
                continue
            row.append(f"{f(b['erle_ss'])} dB; {f(b['silero_after_1s'], 1, True)}; fs {b['false_starts_150_after_1s']}")
        print(f"| {v} | " + " | ".join(row) + " |")


def cpu_table(d: dict) -> None:
    print("\n### CPU (one 60 s call, 6000 timed 10 ms frames; 8 sessions on ThreadPoolExecutor(8))\n")
    print("| Engine | CPU ms per audio s (thread / process) | frame CPU p50 / p95 / p99 / max (µs) | 1 session x RT | 8 sessions x RT | scaling 8 vs 1 | load 1-min before → after |")
    print("|---|---|---|---|---|---|---|")
    for e, r in d["engines"].items():
        if "single" not in r:
            print(f"| {e} | {r} |")
            continue
        s, cc = r["single"], r["concurrency"]
        fc = s["frame_cpu_us"]
        print(f"| {e} | {s['cpu_ms_per_audio_s_thread']} / {s['cpu_ms_per_audio_s_process']} | {fc['p50']} / {fc['p95']} / {fc['p99']} / {fc['max']} | "
              f"{cc['1']['throughput_x_realtime']} | {cc['8']['throughput_x_realtime']} | {cc['scaling_8_vs_1']} | "
              f"{r['load_before'][0]} → {cc['8']['load_after'][0]} |")
    g = d["engines"].get("none_pf", {}).get("gcc_phat")
    if g:
        print(f"\nGCC-PHAT tracker: {g['cpu_ms_per_audio_s_thread']} ms CPU per audio second ({g['windows']} windows).")


def cer_table(d: dict) -> None:
    print("\n### Near-end ASR during double talk (Qwen3-ASR-1.7B-8bit; zh CER, en WER; dCER vs the clean chain, 95 % CI)\n")
    keys = ["zh|near0", "zh|near-15", "en|near0", "en|near-15"]
    print("| Variant | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for v, s in d["summary"].items():
        row = []
        for k in keys:
            x = s.get(k, {})
            vc = x.get("vs_clean")
            row.append(f"{f(x.get('rate'), 1, True)}" + (f" (Δ {vc['delta_points']:+.1f} [{vc['ci95_points'][0]:+.1f}, {vc['ci95_points'][1]:+.1f}])" if vc else ""))
        print(f"| {v} | " + " | ".join(row) + " |")
    print("\nBy ERL:")
    for v, s in d["summary"].items():
        row = []
        for k in sorted(k for k in s if k.count("|") == 2):
            vc = s[k].get("vs_clean")
            row.append(f"{k} {f(s[k].get('rate'), 1, True)}" + (f" Δ{vc['delta_points']:+.1f}" if vc else ""))
        print(f"- {v}: " + "; ".join(row))


def main() -> None:
    parts = sys.argv[1:] or ["core", "timeline", "slices", "dt", "cpu", "cer"]
    for p in parts:
        d = load(p)
        if d is None:
            print(f"\n(s11_{p}.json missing)")
            continue
        if p == "core":
            core_table(d, "Core grid (m1-m8 x ERL 6/10/15/20 x D 20/120/300, 96 calls per variant)")
            near_table(d)
            curve_table(d)
            guard_table(d)
            vs = list(d["variants"])
            by_table(d, "by_erl", vs)
            by_table(d, "by_d", vs)
        elif p == "timeline":
            timeline_table(d)
            timeline_matrix(d)
        elif p == "slices":
            slices_table(d)
        elif p == "dt":
            dt_table(d)
        elif p == "cpu":
            cpu_table(d)
        elif p == "cer":
            cer_table(d)


if __name__ == "__main__":
    main()

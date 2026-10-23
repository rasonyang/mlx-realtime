"""Print compact tables from spikes/results/s1_*.json and s2_*.json (for the S1/S2 report).

    uv run --project spikes/llm python spikes/llm/summarize.py [qwen3.6|qwen3.5]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(s: dict | None, k: str = "p50") -> str:
    if not s or not s.get("n"):
        return "-"
    return f"{s[k]:.3f}"


def triple(s: dict | None) -> str:
    if not s or not s.get("n"):
        return "-"
    return f"{s['p50']:.2f} / {s['p95']:.2f} / {s['max']:.2f} (n={s['n']})"


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6"
    d = load(f"s1_4_{model}.json")
    if d:
        for v, run in d["data"]["runs"].items():
            print(f"S1.4 {v}: scores {run['score_per_rep_out_of_20']} drops {run['silent_drops']} "
                  f"split {run['calls_split_over_chunks']}")
            for pc in run["per_case"]:
                bad = [x["why"] for x in pc["runs"] if not x["ok"]]
                if bad:
                    print(f"   {pc['id']}: {pc['ok_runs']}/{len(pc['runs'])} {bad}")
    d = load(f"s1_5_{model}.json")
    if d:
        for x in d["data"]["interleaved"]:
            print(f"S1.5 {x['conv']}{x['turn']} prompt {x['prompt_tokens']} cached {x['cached_tokens']} "
                  f"ttft {x['ttft']:.3f}")
    d = load(f"s1_7_{model}.json")
    if d:
        for k, v in d["data"].items():
            print(f"S1.7 {k}: ttft {triple(v['ttft'])} tps/stream {triple(v['decode_tps_per_stream'])} "
                  f"agg {v['aggregate_tps_p50']}")
    d = load(f"s2_grid_{model}.json")
    if d:
        for size, v in d["data"].items():
            for cw in ("cold", "warm"):
                s = v[cw]["stats"]
                print(f"S2 grid {size} {cw} (prompt {v[cw]['prompt_tokens']}): ttft {triple(s['ttft'])} "
                      f"tps {triple(s.get('decode_tps'))}")
    d = load(f"s2_aicc_{model}.json")
    if d:
        from llmlib import pq

        def show(name, block):
            s = block["stats"]
            rows = block["rows"]
            sp = [x for x in rows if not x["tool_calls"]]
            tl = [x for x in rows if x["tool_calls"]]
            print(f"S2 aicc {name} SPOKEN n={len(sp)}: ttft {triple(pq([x['ttft'] for x in sp]))} | boundary "
                  f"{triple(pq([x['t_first_boundary'] for x in sp]))} || TOOL n={len(tl)}: done "
                  f"{triple(pq([x['t_done'] for x in tl]))}")
            print(f"S2 aicc {name}: ttft {triple(s.get('ttft'))} | boundary {triple(s.get('t_first_boundary'))} "
                  f"| tool {triple(s.get('t_first_tool_call'))} | done {triple(s.get('t_done'))} "
                  f"| uncached p50 {fmt(s.get('uncached'))}")
        for k, v in d["data"].items():
            show(k, v)
            for sub in ("next_turn_after_change", "next_normal_response"):
                if sub in v:
                    show(f"{k} -> {sub}", v[sub])
    d = load(f"s2_warming_{model}.json")
    if d:
        for k, v in d["data"].items():
            s = v["stats"]
            print(f"S2 warming {k}: ttft {triple(s.get('ttft'))} uncached {fmt(s.get('uncached'))} "
                  f"warm_done {fmt(s.get('warm_done_s'))} from_warm_start {triple(s.get('real_ttft_from_warm_start'))}")
    d = load(f"s2_sustained_{model}.json")
    if d:
        x = d["data"]
        print(f"S2 sustained: tps {triple(x['decode_tps'])} first/last fifth {x['first_fifth_mean']:.1f}/"
              f"{x['last_fifth_mean']:.1f} ratio {x['ratio_last_first']:.3f}")
    for tag in ("", "_res1000"):
        d = load(f"s2_idle{tag}_{model}.json")
        if d:
            x = d["data"]
            print(f"S2 idle{tag} {x['idle_minutes']} min: baseline aicc {triple(x['baseline_aicc_ttft'])} "
                  f"plain {triple(x['baseline_plain_ttft'])}; after: aicc {x['after_idle_first_aicc']['ttft']:.3f} "
                  f"plain {x['after_idle_plain']['ttft']:.3f} aicc2 {x['after_idle_aicc_again']['ttft']:.3f}")


if __name__ == "__main__":
    main()

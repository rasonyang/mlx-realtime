"""The lenient tool-call parser, live: B1's S1.4 transfer probe against ``mlxrt_llm.serve``.

B1's probe (``spikes/llm/s1_toolparse_probe.py``) found that stock mlx_lm.server drops 11% of the
calls Qwen3.6 makes on the realistic turn "转人工，把情况说清楚" (the model writes
``<transfer_to_agent>`` for ``<function=transfer_to_agent>``). This replays the same turn, built
the same way (B1's ``llmlib`` / ``s2_perf`` helpers, imported from ``spikes/llm``), against the
wrapper on 127.0.0.1:8101, at B1's sampling (0.7 / 0.8) and at the G2 tool sampling (greedy),
and counts per run: calls delivered, calls dropped (``finish_reason: tool_calls`` and none), and
the wrapper's "recovered" log lines. Writes ``spikes/results/llm_hardening_parser.json``.

    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm_hardening python \\
        spikes/llm_hardening/probe_parser.py --n 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llm"))  # B1's helpers: plain http.client, no MLX

import llmlib as L  # noqa: E402
import s2_perf as P  # noqa: E402

from run import G2_FLAGS, RESULTS, TRANSFER_TURN, start_backend  # noqa: E402

RECOVERED = "tool call written as <NAME> instead of <function=NAME>: recovered"
DROPPED = "Failed to parse tool call"


def probe(n: int, sampling: dict[str, float], seed: int, backend_log: Path) -> dict[str, object]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        hist, _ = P.converse(P.A["instructions"], [P.CALLER[i % 20], rnd.choice(P.FOLLOW)])
        hist.append(P.u(TRANSFER_TURN))
        before = backend_log.read_text(errors="replace")
        r = L.stream_chat(L.gateway_body(hist, P.A["tools"], sampling=sampling))
        time.sleep(0.2)
        log = backend_log.read_text(errors="replace")[len(before) :]
        rows.append(
            {
                "i": i,
                "finish_reason": r.finish_reason,
                "calls": [c["function"]["name"] for c in r.tool_calls],
                "dropped": r.finish_reason == "tool_calls" and not r.tool_calls,
                "recovered_in_log": log.count(RECOVERED),
                "parse_failures_in_log": log.count(DROPPED),
                "t_done": r.t_done,
                "content": r.content[:120],
            }
        )
        print(i, rows[-1]["calls"], "DROP" if rows[-1]["dropped"] else "", flush=True)
    return {
        "sampling": sampling,
        "n": n,
        "called_transfer": sum(row["calls"] == ["transfer_to_agent"] for row in rows),
        "dropped": sum(bool(row["dropped"]) for row in rows),
        "recovered": sum(int(row["recovered_in_log"]) for row in rows),
        "parse_failures": sum(int(row["parse_failures_in_log"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    # B1's conversation helper samples the earlier turns at its own gateway setting.
    backend = start_backend([])
    try:
        runs = {
            "b1_sampling_0.7": probe(args.n, {"temperature": 0.7, "top_p": 0.8}, 5, backend.log),
            "g2_tool_sampling_0": probe(args.n, {"temperature": 0.0, "top_p": 1.0}, 6, backend.log),
        }
    finally:
        backend.stop()
    result = {
        "spike": "LLM hardening: the lenient tool-call parser on B1's transfer probe",
        "conditions": {"server": " ".join(backend.argv), "server_flags": G2_FLAGS,
                       "model": L.MODELS["qwen3.6"], "log": str(backend.log)},  # fmt: skip
        "runs": runs,
    }
    path = RESULTS / "llm_hardening_parser.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for name, run in runs.items():
        summary = {key: run[key] for key in ("n", "called_transfer", "dropped", "recovered")}
        print(name, summary)
    print("->", path)


if __name__ == "__main__":
    main()

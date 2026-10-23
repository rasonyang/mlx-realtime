"""S1.4 follow-up: why does Qwen3.6 sometimes end with finish_reason tool_calls but no parsed call?

Starts mlx_lm.server with the base flags plus --log-level DEBUG (the handler then logs every generated
token, including the tool-call span the client never sees), sends the transfer_to_agent prompt of
s2_perf.part_transfer n times, and keeps the raw generated text of every response whose call was dropped
("Failed to parse tool call" in the log). Writes spikes/results/s1_toolparse_<model>.json.

    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm python spikes/llm/s1_toolparse_probe.py --model qwen3.6 --n 30
"""

from __future__ import annotations

import argparse
import random
import re
import time

import llmlib as L
import s2_perf as P

DEBUG_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ - (DEBUG|INFO|WARNING) - ", re.M)


def generated_text(log: str) -> str:
    """Join the DEBUG token lines after 'Starting stream:' (tokens can contain newlines)."""
    start = log.find("Starting stream:")
    if start < 0:
        return ""
    body = log[start + len("Starting stream:"):]
    parts = DEBUG_LINE.split(body)
    # split -> [pre, level, text, level, text, ...]
    out = []
    for level, text in zip(parts[1::2], parts[2::2]):
        if level == "INFO":  # prompt-progress lines interleave with the tokens
            continue
        if level != "DEBUG" or text.startswith("Incoming Request Body"):
            break
        out.append(text[:-1] if text.endswith("\n") else text)
    return "".join(out)


def reparse(log_path: str, json_path: str) -> None:
    """Re-extract the raw text of every dropped call from a kept server log (one per parse warning)."""
    import json
    from pathlib import Path

    log = Path(log_path).read_text(errors="replace")
    raws = []
    for m in re.finditer(r"Failed to parse tool call", log):
        start = log.rfind("Starting stream:", 0, m.start())
        raws.append(generated_text(log[start:m.start()]))
    d = json.loads(Path(json_path).read_text())
    d["dropped_raw_from_log"] = raws
    Path(json_path).write_text(json.dumps(d, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6", choices=list(L.MODELS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reparse-log", default=None, help="only re-extract raw text from this kept log")
    args = ap.parse_args()
    if args.reparse_log:
        reparse(args.reparse_log, str(L.RESULTS / f"s1_toolparse_{args.model}.json"))
        return
    srv = L.Server(L.MODELS[args.model], extra_flags=["--log-level", "DEBUG"], tag=f"toolparse_{args.model}")
    srv.start()
    rnd = random.Random(5)
    rows = []
    try:
        for i in range(args.n):
            hist, _ = P.converse(P.A["instructions"], [P.CALLER[i % 20], rnd.choice(P.FOLLOW)])
            hist.append(P.u("你说的我都试过了没用，给我转人工客服，把我的情况跟他们说清楚。"))
            time.sleep(0.3)
            off = srv.log_offset()
            r = P.aicc_req(hist)
            time.sleep(0.5)
            log = srv.log_since(off)
            dropped = r.finish_reason == "tool_calls" and not r.tool_calls
            rows.append({"i": i, "finish_reason": r.finish_reason, "dropped": dropped,
                         "parse_warning": "Failed to parse tool call" in log,
                         "tool_calls": r.tool_calls, "content": r.content,
                         "raw_generated": generated_text(log), "load1": round(L.loadavg(), 1)})
            print(i, "DROP" if dropped else "ok", flush=True)
    finally:
        srv.stop()
    L.write_json(f"s1_toolparse_{args.model}.json", {
        "spike": "S1.4 tool-call parse probe", "conditions": srv.conditions(),
        "n": len(rows), "dropped": sum(r["dropped"] for r in rows), "rows": rows})


if __name__ == "__main__":
    main()

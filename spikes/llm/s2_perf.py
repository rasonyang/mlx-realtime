"""S2 — LLM performance of mlx_lm.server 0.32.2 with Qwen3.x-35B-A3B-4bit (M0, docs/SPIKE.md §S2).

Starts its own server on 127.0.0.1:8101 (HF_HUB_OFFLINE=1, --model <snapshot dir>, llmlib.BASE_FLAGS) and
runs the selected parts. All requests are gateway-shaped (llmlib.gateway_body). Writes
spikes/results/s2_<part>_<model>.json.

    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm python spikes/llm/s2_perf.py --model qwen3.6 \
        --parts first,grid,aicc,warming,sustained
    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm python spikes/llm/s2_perf.py --model qwen3.6 \
        --parts idle --idle-min 15 [--residency-ms 1000]

Parts:
  first      the first AICC request after the 1-token warmup (model-cold kernels), n=1
  grid       TTFT and decode tok/s at ~300/1k/4k/16k prompt tokens, prefix cache cold and warm
  aicc       the AICC prompt (line_session_zh.json): cold, new call with cached instructions, warm append,
             instructions change (unseen / seen before), per-response instructions override and the
             response after it, and time to a complete tool call
  warming    1-token cache warming after an instructions change, at several gaps
  sustained  back-to-back decode for ~3 minutes, to see GPU clock step-down
  idle       TTFT after N minutes idle (optionally with MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS)
"""

from __future__ import annotations

import argparse
import random
import threading
import time
import uuid

import llmlib as L

A = L.aicc()
CALLER = [
    "你好，我想问一下手机怎么截图。", "我的手机充电特别慢，怎么回事？", "手机存储空间满了怎么清理？",
    "怎么把手机系统升级到最新版本？", "我手机老是自动重启，怎么办？", "怎么关闭手机的自动亮度？",
    "手机发烫正常吗？", "怎么设置来电铃声？", "手机连不上无线网怎么办？", "怎么恢复出厂设置？",
    "我的手机屏幕有点发黄，能调吗？", "怎么开启省电模式？", "蓝牙耳机连不上手机怎么办？",
    "怎么备份手机里的照片？", "手机信号不好怎么办？", "怎么查看手机还剩多少存储？",
    "手机开不了机了怎么办？", "怎么关掉应用通知？", "手机突然没声音了怎么回事？", "怎么设置指纹解锁？",
    "那录屏呢？", "好的，那我试一下。", "还有别的办法吗？", "这个要花钱吗？", "大概要多久？",
]
FOLLOW = ["那还有别的办法吗？", "好的，我试试。那如果还不行呢？", "明白了，那需要送修吗？", "这样会不会丢数据？",
          "嗯，那大概多久能好？", "谢谢，那我顺便再问一下怎么截图？"]
LONG_Q = "请尽量详细地回答，至少写三百字：手机电池怎样保养才能用得更久？"
FILL = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget and the "
        "engineers discuss latency, throughput and the cost of memory bandwidth. ")


def sysmsg(text: str) -> dict:
    return {"role": "system", "content": text}


def u(text: str) -> dict:
    return {"role": "user", "content": text}


def asst(text: str) -> dict:
    return {"role": "assistant", "content": text}


def row(r: L.Result, **extra) -> dict:
    return {"ttft": r.t_first_token, "t_first_boundary": r.t_first_boundary, "t_first_content": r.t_first_content,
            "t_first_tool_call": r.t_first_tool_call, "t_done": r.t_done, "prompt_tokens": r.prompt_tokens,
            "cached": r.cached_tokens, "uncached": (r.prompt_tokens or 0) - (r.cached_tokens or 0),
            "completion_tokens": r.completion_tokens, "decode_tps": r.decode_tps, "finish_reason": r.finish_reason,
            "status": r.status, "content": r.content[:200],
            "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in r.tool_calls],
            "wall_clock": time.strftime("%H:%M:%S"), "load1": round(L.loadavg(), 1), **extra}


def stats(rows: list[dict], keys=("ttft", "t_first_boundary", "t_first_tool_call", "t_done", "uncached",
                                  "decode_tps")) -> dict:
    return {k: L.pq([x.get(k) for x in rows]) for k in keys if any(x.get(k) is not None for x in rows)}


def aicc_req(msgs: list[dict], **kw) -> L.Result:
    return L.stream_chat(L.gateway_body(msgs, A["tools"], **kw))


# ---------------------------------------------------------------- parts


def part_first(srv: L.Server) -> dict:
    msgs = [sysmsg(A["instructions"]), asst(A["announce"]), u(CALLER[0])]
    r = aicc_req(msgs)
    r2 = aicc_req([sysmsg("(x)" + A["instructions"]), asst(A["announce"]), u(CALLER[1])])
    return {"first_aicc_request_after_warmup": row(r), "second_cold_prefill_request": row(r2),
            "server_ready_s": srv.ready_s, "server_first_request_done_s": srv.first_request_s}


def synth(n_tokens: int, nonce: str | None) -> list[dict]:
    n = max(1, (n_tokens - 60) // 36)
    text = "".join(f"[{i}] {FILL}" for i in range(n))
    head = f"Session {nonce}. " if nonce else "Session fixed. "
    return [sysmsg(head + "You are a helpful assistant."),
            u(text + "\n\nIgnore the document above. Write a long, detailed story about a lighthouse keeper.")]


def part_grid(srv: L.Server, n: int, sizes=(300, 1000, 4000, 16000)) -> dict:
    out = {}
    for size in sizes:
        cold, warm = [], []
        for _ in range(n):
            r = L.stream_chat(L.gateway_body(synth(size, uuid.uuid4().hex), max_completion_tokens=128,
                                             sampling={"temperature": 0.0}), timeout=600)
            cold.append(row(r))
        fixed = synth(size, None)
        L.stream_chat(L.gateway_body(fixed, max_completion_tokens=128, sampling={"temperature": 0.0}), timeout=600)
        for _ in range(n):
            r = L.stream_chat(L.gateway_body(fixed, max_completion_tokens=128, sampling={"temperature": 0.0}),
                              timeout=600)
            warm.append(row(r))
        out[str(size)] = {"cold": {"stats": stats(cold), "prompt_tokens": cold[0]["prompt_tokens"], "rows": cold},
                          "warm": {"stats": stats(warm), "prompt_tokens": warm[0]["prompt_tokens"], "rows": warm}}
        print(f"  grid {size}: cold ttft p50 {out[str(size)]['cold']['stats']['ttft']['p50']} "
              f"warm {out[str(size)]['warm']['stats']['ttft']['p50']}", flush=True)
    return out


def converse(instr: str, turns: list[str], start: list[dict] | None = None) -> tuple[list[dict], list[L.Result]]:
    hist = start or [sysmsg(instr), asst(A["announce"])]
    res = []
    for t in turns:
        hist.append(u(t))
        r = aicc_req(hist)
        res.append(r)
        if r.tool_calls:
            tc = r.tool_calls[0]
            hist.append({"role": "assistant", "content": r.content, "tool_calls": [
                {"id": tc["id"], "type": "function", "function": tc["function"]}]})
            hist.append({"role": "tool", "tool_call_id": tc["id"], "content": '{"found":1,"status":"维修中"}'})
            r2 = aicc_req(hist)
            hist.append(asst(r2.content))
        else:
            hist.append(asst(r.content))
    return hist, res


def part_aicc(srv: L.Server, n: int) -> dict:
    rnd = random.Random(7)
    out: dict = {}
    # 1. cold: instructions never seen (a unique tag in front), first caller turn.
    rows = []
    for i in range(n):
        instr = f"(通话 {uuid.uuid4().hex[:8]})\n" + A["instructions"]
        r = aicc_req([sysmsg(instr), asst(A["announce"]), u(CALLER[i % 20])])
        rows.append(row(r))
    out["1_cold"] = {"stats": stats(rows), "rows": rows}
    print(f"  1_cold ttft {out['1_cold']['stats']['ttft']}", flush=True)
    # 1b/2/tool: n calls with the session instructions (cached across calls), 4 requests each.
    new_call, warm, tool = [], [], []
    for i in range(n):
        hist = [sysmsg(A["instructions"]), asst(A["announce"])]
        turns = [CALLER[i % 20], rnd.choice(FOLLOW), rnd.choice(FOLLOW)]
        for j, t in enumerate(turns):
            hist.append(u(t))
            r = aicc_req(hist)
            (new_call if j == 0 else warm).append(row(r, turn=j + 1))
            hist.append(asst(r.content))
            if r.tool_calls:  # keep the history valid if it called anyway
                hist[-1] = {"role": "assistant", "content": r.content, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": c["function"]} for c in r.tool_calls]}
                for c in r.tool_calls:
                    hist.append({"role": "tool", "tool_call_id": c["id"], "content": '{"ok":1}'})
                hist.append(u("嗯。"))
                r = aicc_req(hist)
                hist.append(asst(r.content))
        hist.append(u(f"我的维修单号是RMA{rnd.randint(1000, 9999)}，帮我查一下进度。"))
        r = aicc_req(hist)
        tool.append(row(r))
    out["1b_new_call_cached_instructions"] = {"stats": stats(new_call), "rows": new_call}
    out["2_warm_append"] = {"stats": stats(warm), "rows": warm}
    out["tool_call_warm"] = {"stats": stats(tool), "rows": tool,
                             "called": sum(1 for x in tool if x["tool_calls"])}
    print(f"  2_warm ttft {out['2_warm_append']['stats']['ttft']}", flush=True)
    print(f"  tool {out['tool_call_warm']['stats'].get('t_done')}", flush=True)
    # 3. instructions-only change (a new phase): unseen text (unique tag) and seen-before text.
    for name, unseen in (("3_instructions_change_unseen", True), ("3b_instructions_change_seen_before", False)):
        rows, after, tool_after = [], [], []
        for i in range(n):
            hist, _ = converse(A["instructions"], [CALLER[i % 20], rnd.choice(FOLLOW)])
            phase = L.PHASE_CLOSING + (f"(变更 {uuid.uuid4().hex[:6]})" if unseen else "")
            hist[0] = sysmsg(L.aicc_phase_variant(A["instructions"], phase))
            hist.append(u("好的，没有别的问题了。" if i % 2 else "那我再问一下，手机怎么截图？"))
            r = aicc_req(hist)
            rows.append(row(r))
            hist.append(asst(r.content))
            hist.append(u(rnd.choice(FOLLOW)))
            r = aicc_req(hist)
            after.append(row(r))
        out[name] = {"stats": stats(rows), "rows": rows, "next_turn_after_change": {"stats": stats(after),
                                                                                     "rows": after}}
        print(f"  {name} ttft {out[name]['stats']['ttft']}", flush=True)
    # Tool call right after an unseen instructions change (full re-prefill + tool span).
    rows = []
    for i in range(n):
        hist, _ = converse(A["instructions"], [CALLER[i % 20]])
        hist[0] = sysmsg(L.aicc_phase_variant(A["instructions"], L.PHASE_MESSAGE + f"(变更 {uuid.uuid4().hex[:6]})"))
        hist.append(u(f"我要查维修进度，单号RMA{rnd.randint(1000, 9999)}。"))
        rows.append(row(aicc_req(hist)))
    out["tool_call_after_instructions_change"] = {"stats": stats(rows), "rows": rows,
                                                  "called": sum(1 for x in rows if x["tool_calls"])}
    # 4. per-response instructions override replacing the system message, then the next normal response.
    rows, after = [], []
    overrides = ["请用一句话亲切地问来电者还有没有其它需要帮忙的。", "请简短地向来电者道歉让他久等了，然后问他是否还在线。",
                 "用一句话提醒来电者可以随时按0转人工，然后继续回答他的问题。"]
    for i in range(n):
        hist, _ = converse(A["instructions"], [CALLER[i % 20], rnd.choice(FOLLOW)])
        ov = [sysmsg(overrides[i % 3])] + hist[1:]
        r = aicc_req(ov + [u("嗯。")]) if ov[-1]["role"] != "user" else aicc_req(ov)
        rows.append(row(r))
        hist.append(u("嗯。"))
        hist.append(asst(r.content))
        hist.append(u(rnd.choice(FOLLOW)))
        after.append(row(aicc_req(hist)))
    out["4_instructions_override"] = {"stats": stats(rows), "rows": rows,
                                      "next_normal_response": {"stats": stats(after), "rows": after}}
    print(f"  4_override ttft {out['4_instructions_override']['stats']['ttft']}", flush=True)
    return out


def part_warming(srv: L.Server, n: int, n_secondary: int) -> dict:
    """After an unseen instructions change: no warming vs a 1-token warming request, at several gaps.

    W2 warms [new system + history up to the last user turn] (both checkpoints); W3 warms [new system +
    synthetic cue] (system checkpoint only). gap: "after_done" sends the real request right after the
    warming request's [DONE]; a number sends it that many seconds after the warming request started.
    """
    rnd = random.Random(11)
    out: dict = {}
    plan = [("none", None, n), ("W2", "after_done", n), ("W2", 0.0, n), ("W2", 0.3, n_secondary),
            ("W2", 0.8, n_secondary), ("W3", "after_done", n), ("W3", 0.0, n_secondary),
            ("W2", "after_done+1s", n_secondary)]
    for mode, gap, reps in plan:
        rows = []
        for i in range(reps):
            hist, _ = converse(A["instructions"], [CALLER[(i + 3) % 20], rnd.choice(FOLLOW)])
            new_sys = sysmsg(L.aicc_phase_variant(A["instructions"], L.PHASE_CLOSING + f"(变更 {uuid.uuid4().hex[:6]})"))
            hist[0] = new_sys
            real = hist + [u("好的，没有别的问题了。" if i % 2 else "那我再问一下，手机怎么截图？")]
            warm_msgs = None
            if mode == "W2":
                last_user = max(k for k, m in enumerate(hist) if m["role"] == "user")
                warm_msgs = hist[:last_user + 1]
            elif mode == "W3":
                warm_msgs = [new_sys, u(L.CUE_GATEWAY)]
            wres: dict = {}
            if warm_msgs is None:
                r = aicc_req(real)
                rows.append(row(r, gap=None))
                continue
            t0 = time.perf_counter()

            def do_warm() -> None:
                wres["r"] = L.stream_chat(L.gateway_body(warm_msgs, A["tools"], max_completion_tokens=1))
                wres["t_end"] = time.perf_counter() - t0

            th = threading.Thread(target=do_warm)
            th.start()
            if isinstance(gap, float):
                time.sleep(gap)
            else:
                th.join()
                if gap == "after_done+1s":
                    time.sleep(1.0)
            sent = time.perf_counter() - t0
            r = aicc_req(real)
            th.join()
            rows.append(row(r, gap=gap, real_sent_after_s=round(sent, 3), warm_done_s=round(wres["t_end"], 3),
                            warm_ttft=wres["r"].t_first_token, warm_prompt_tokens=wres["r"].prompt_tokens,
                            warm_cached=wres["r"].cached_tokens,
                            real_ttft_from_warm_start=round(sent + (r.t_first_token or 0), 3)))
        key = f"{mode}_gap_{gap}"
        out[key] = {"stats": stats(rows, keys=("ttft", "t_first_boundary", "uncached", "warm_done_s",
                                               "real_ttft_from_warm_start")), "rows": rows}
        print(f"  warming {key}: ttft {out[key]['stats']['ttft']}", flush=True)
    return out


def part_transfer(srv: L.Server, n: int) -> dict:
    """Time to a complete transfer_to_agent call (its summary argument makes the longest tool span), warm and
    right after an unseen instructions change."""
    rnd = random.Random(5)
    out = {}
    for name, change in (("transfer_warm", False), ("transfer_after_instructions_change", True)):
        rows = []
        for i in range(n):
            hist, _ = converse(A["instructions"], [CALLER[i % 20], rnd.choice(FOLLOW)])
            if change:
                hist[0] = sysmsg(L.aicc_phase_variant(A["instructions"], L.PHASE_CLOSING + f"(变更 {uuid.uuid4().hex[:6]})"))
            hist.append(u("你说的我都试过了没用，给我转人工客服，把我的情况跟他们说清楚。"))
            rows.append(row(aicc_req(hist)))
        out[name] = {"stats": stats(rows), "rows": rows,
                     "called_transfer": sum(1 for x in rows if x["tool_calls"] and x["tool_calls"][0][0] ==
                                            "transfer_to_agent")}
        print(f"  {name}: done {out[name]['stats'].get('t_done')}", flush=True)
    return out


def part_sustained(srv: L.Server, seconds: float = 180) -> dict:
    rows = []
    t0 = time.time()
    msgs = [sysmsg("You are a storyteller."), u("Write a very long story about a lighthouse keeper.")]
    while time.time() - t0 < seconds:
        r = L.stream_chat(L.gateway_body(msgs, max_completion_tokens=400, sampling={"temperature": 0.7}))
        rows.append({"t": round(time.time() - t0, 1), "decode_tps": r.decode_tps, "ttft": r.t_first_token,
                     "completion_tokens": r.completion_tokens})
    tps = [x["decode_tps"] for x in rows if x["decode_tps"]]
    first, last = tps[: max(1, len(tps) // 5)], tps[-max(1, len(tps) // 5):]
    return {"rows": rows, "decode_tps": L.pq(tps), "first_fifth_mean": sum(first) / len(first),
            "last_fifth_mean": sum(last) / len(last), "ratio_last_first": (sum(last) / len(last)) / (sum(first) / len(first))}


def part_idle(srv: L.Server, minutes: float) -> dict:
    msgs = [sysmsg(A["instructions"]), asst(A["announce"]), u(CALLER[0])]
    plainm = [sysmsg("You are brief."), u("Say hello in one short sentence.")]
    base_aicc = [row(aicc_req(msgs)) for _ in range(5)]
    base_plain = [row(L.stream_chat(L.gateway_body(plainm, max_completion_tokens=20))) for _ in range(5)]
    fp_before = srv.footprint_gb()
    la0 = L.loadavg()
    print(f"  idle {minutes} min from {time.strftime('%H:%M:%S')}", flush=True)
    time.sleep(minutes * 60)
    fp_after = srv.footprint_gb()
    la1 = L.loadavg()
    after_aicc = row(aicc_req(msgs))
    after_plain = row(L.stream_chat(L.gateway_body(plainm, max_completion_tokens=20)))
    after_aicc2 = row(aicc_req(msgs))
    return {"idle_minutes": minutes, "baseline_aicc_cached": base_aicc, "baseline_plain": base_plain,
            "baseline_aicc_ttft": L.pq([x["ttft"] for x in base_aicc]),
            "baseline_plain_ttft": L.pq([x["ttft"] for x in base_plain]),
            "after_idle_first_aicc": after_aicc, "after_idle_plain": after_plain, "after_idle_aicc_again": after_aicc2,
            "footprint_gb_before": fp_before, "footprint_gb_after": fp_after,
            "loadavg_start_idle": la0, "loadavg_end_idle": la1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6", choices=list(L.MODELS))
    ap.add_argument("--parts", default="first,grid,aicc,warming,sustained")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-secondary", type=int, default=10)
    ap.add_argument("--grid-sizes", default="300,1000,4000,16000")
    ap.add_argument("--idle-min", type=float, default=15)
    ap.add_argument("--residency-ms", type=int, default=None)
    ap.add_argument("--sustained-s", type=float, default=180)
    args = ap.parse_args()
    env = {"MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS": str(args.residency_ms)} if args.residency_ms else {}
    tag = f"_res{args.residency_ms}" if args.residency_ms else ""
    srv = L.Server(L.MODELS[args.model], env=env, tag=f"s2_{args.model}{tag}")
    srv.start()
    try:
        for part in args.parts.split(","):
            la0 = L.loadavg()
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] part {part} ...", flush=True)
            if part == "first":
                data = part_first(srv)
            elif part == "grid":
                data = part_grid(srv, args.n, tuple(int(s) for s in args.grid_sizes.split(",")))
            elif part == "aicc":
                data = part_aicc(srv, args.n)
            elif part == "warming":
                data = part_warming(srv, args.n, args.n_secondary)
            elif part == "transfer":
                data = part_transfer(srv, args.n)
            elif part == "sustained":
                data = part_sustained(srv, args.sustained_s)
            elif part == "idle":
                data = part_idle(srv, args.idle_min)
            else:
                raise SystemExit(f"unknown part {part}")
            res = {"spike": f"S2.{part}", "conditions": {
                **srv.conditions(), "concurrency": 1, "warm_or_cold": "see keys", "loadavg_1m_before": la0,
                "loadavg_1m_after": L.loadavg(), "wall_s": round(time.time() - t0, 1), "env": L.env_info(),
                "footprint_gb": srv.footprint_gb(), "sampling": L.GATEWAY_SAMPLING,
                "max_completion_tokens": L.GATEWAY_MAX_COMPLETION}, "data": data}
            p = L.write_json(f"s2_{part}{tag}_{args.model}.json", res)
            print(f"  -> {p} ({res['conditions']['wall_s']} s)", flush=True)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()

"""S1 — LLM behaviour of mlx_lm.server 0.32.2 with Qwen3.x-35B-A3B-4bit (M0, docs/SPIKE.md §S1).

Starts its own server on 127.0.0.1:8101 (HF_HUB_OFFLINE=1, --model <snapshot dir>, base flags in
llmlib.BASE_FLAGS), runs the selected items and writes spikes/results/s1_<item>_<model>.json.

    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm python spikes/llm/s1_behaviour.py \
        --model qwen3.6 --items 1,2,3,4,5,6,7
    SPIKE_LOG_DIR=<scratch> uv run --project spikes/llm python spikes/llm/s1_behaviour.py \
        --model qwen3.5 --items 1,4,7

Items: 1 thinking off/on, 2 request fields, 3 template edge cases + AICC history shapes, 4 tool calls,
5 prompt cache across conversations, 6 cancellation, 7 concurrency.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid

import llmlib as L

A = L.aicc()
SYS_PLAIN = "You are a helpful voice assistant. Answer in one or two short spoken sentences, in the user's language."

THINK_PROMPTS = [
    "请用一两句话介绍一下杭州。", "手机电池不耐用有什么办法？", "什么是5G网络？", "怎么把照片从手机传到电脑？",
    "我的手机充不进电了，可能是什么原因？", "推荐一个周末放松的方式。", "解释一下什么是云存储。",
    "13乘以17等于多少？", "如何设置手机的指纹解锁？", "用一句话说说你能帮我做什么。",
    "What is the capital of Australia?", "How do I free up storage on my phone?",
    "Explain what a SIM card does in one sentence.", "What's 15% of 80?", "Give me a tip for better sleep.",
    "Why does my phone get hot when charging?", "What does airplane mode do?",
    "Tell me a fun fact about octopuses.", "How do I take a screenshot on Android?",
    "Is it safe to leave my phone charging overnight?",
]


def plain(prompt: str, system: str = SYS_PLAIN) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def aicc_msgs(*turns: dict, instructions: str | None = None) -> list[dict]:
    return [{"role": "system", "content": instructions or A["instructions"]},
            {"role": "assistant", "content": A["announce"]}, *turns]


def u(text: str) -> dict:
    return {"role": "user", "content": text}


def call(name: str, args: dict, cid: str | None = None) -> dict:
    return {"id": cid or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


# ---------------------------------------------------------------- item 1


def item1(srv: L.Server) -> dict:
    off = []
    for p in THINK_PROMPTS:
        r = L.stream_chat(L.gateway_body(plain(p), max_completion_tokens=300))
        s = r.summary()
        off.append({"prompt": p, **{k: s[k] for k in ("status", "content", "reasoning", "think_markup",
                                                       "finish_reason", "usage", "t_first_token")}})
    on = []
    for p in THINK_PROMPTS[:3] + THINK_PROMPTS[10:13]:
        body = L.gateway_body(plain(p), max_completion_tokens=1500)
        body["chat_template_kwargs"] = {"enable_thinking": True}
        r = L.stream_chat(body)
        s = r.summary()
        on.append({"prompt": p, "reasoning_chars": len(s["reasoning"]), "reasoning_head": s["reasoning"][:300],
                   **{k: s[k] for k in ("status", "content", "think_markup", "finish_reason", "usage",
                                        "t_first_token", "t_first_content")}})
    # No chat_template_kwargs at all: only --chat-template-args keeps thinking off.
    noargs = []
    for p in (THINK_PROMPTS[0], THINK_PROMPTS[10]):
        body = L.gateway_body(plain(p), max_completion_tokens=300)
        body.pop("chat_template_kwargs")
        s = L.stream_chat(body).summary()
        noargs.append({"prompt": p, **{k: s[k] for k in ("status", "content", "reasoning", "think_markup")}})
    return {
        "off": off, "on": on, "no_kwargs": noargs,
        "off_ok": sum(1 for o in off if o["status"] == 200 and not o["reasoning"] and not o["think_markup"]),
        "off_n": len(off),
        "on_reasoning": sum(1 for o in on if o["reasoning_chars"] > 0), "on_n": len(on),
    }


# ---------------------------------------------------------------- item 2


def item2(srv: L.Server) -> dict:
    out: dict = {}
    base = plain("Name three colours.")
    # include_usage
    r = L.stream_chat(L.gateway_body(base), keep_raw=True)
    out["include_usage"] = {"usage_before_done": r.usage_before_done, "usage": r.usage,
                            "raw_tail": r.raw[-3:]}
    b = L.gateway_body(base)
    b.pop("stream_options")
    r = L.stream_chat(b, keep_raw=True)
    out["stream_options_omitted"] = {"status": r.status, "usage": r.usage, "t_done": r.t_done,
                                     "exception": r.exception, "raw_tail": r.raw[-2:]}
    b = L.gateway_body(base)
    b["stream_options"] = {}
    off = srv.log_offset()
    r = L.stream_chat(b, keep_raw=True)
    time.sleep(0.3)
    out["stream_options_without_include_usage"] = {
        "status": r.status, "usage": r.usage, "t_done": r.t_done, "finish_reason": r.finish_reason,
        "exception": r.exception, "raw_tail": r.raw[-3:], "server_log": srv.log_since(off)[-1500:]}
    # max_completion_tokens / max_tokens
    r = L.stream_chat(L.gateway_body(plain("Count from 1 to 100."), max_completion_tokens=5))
    out["max_completion_tokens_5"] = {"completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason,
                                      "content": r.content}
    b = L.gateway_body(plain("Count from 1 to 100."))
    b.pop("max_completion_tokens")
    b["max_tokens"] = 7
    r = L.stream_chat(b)
    out["max_tokens_7_legacy"] = {"completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason}
    b = L.gateway_body(plain("Count from 1 to 100."))
    b.pop("max_completion_tokens")
    r = L.stream_chat(b, timeout=120)
    out["no_max_tokens_server_default"] = {"completion_tokens": r.completion_tokens,
                                           "finish_reason": r.finish_reason}
    # temperature / top_p: determinism probes on an open-ended prompt.
    story = plain("Write one sentence about a cat, starting with an unusual adjective.")

    def texts(sampling: dict, n: int = 4) -> list[str]:
        return [L.stream_chat(L.gateway_body(story, sampling=sampling, max_completion_tokens=40)).content
                for _ in range(n)]

    for name, sampling in {
        "temp0": {"temperature": 0.0},
        "temp0.7_top_p0.8": {"temperature": 0.7, "top_p": 0.8},
        "temp1.5_top_p1.0": {"temperature": 1.5, "top_p": 1.0},
        "temp1.5_top_p1e-6": {"temperature": 1.5, "top_p": 1e-6},
        "omitted_uses_server_flags": {},
    }.items():
        t = texts(sampling)
        out[f"sampling_{name}"] = {"distinct": len(set(t)), "n": len(t), "texts": t}
    for bad in ({"top_p": 1.5}, {"temperature": -1}, {"presence_penalty": "x"}, {"max_completion_tokens": 0}):
        b = L.gateway_body(base)
        b.update(bad)
        r = L.stream_chat(b)
        out[f"invalid_{json.dumps(bad)}"] = {"status": r.status, "body": r.error_body}
    # presence_penalty: a repetition-prone prompt at temperature 0, plus an ID-copy probe.
    rep = plain("Repeat the word 'hello' exactly 30 times, separated by spaces.")
    ids = plain("Read back this order number exactly, nothing else: RMA100100100.")
    for pp in (0.0, 1.5):
        r1 = L.stream_chat(L.gateway_body(rep, sampling={"temperature": 0.0}, extra={"presence_penalty": pp}))
        r2 = L.stream_chat(L.gateway_body(ids, sampling={"temperature": 0.0}, extra={"presence_penalty": pp}))
        out[f"presence_penalty_{pp}"] = {"status": r1.status, "repeat_hello": r1.content,
                                         "hello_count": r1.content.lower().count("hello"),
                                         "id_copy": r2.content}
    # model field
    snap = L.snapshot(srv.model)
    for name, model in (("omitted", None), ("exact_--model_string", snap), ("default_model", "default_model")):
        b = L.gateway_body(base)
        if model is not None:
            b["model"] = model
        off = srv.log_offset()
        r = L.stream_chat(b)
        out[f"model_{name}"] = {"status": r.status, "t_first_token": r.t_first_token, "content": r.content,
                                "error": r.error_body, "log": srv.log_since(off)[-800:]}
    for name, model in (("wrong_name", "gpt-4o-realtime-preview"), ("repo_id_not_--model", srv.model)):
        b = L.gateway_body(base)
        b["model"] = model
        off = srv.log_offset()
        t0 = time.perf_counter()
        r = L.stream_chat(b, timeout=600)
        wrong_s = time.perf_counter() - t0
        time.sleep(0.5)
        log = srv.log_since(off)
        # The next request with the field omitted: does it pay a reload?
        t0 = time.perf_counter()
        r2 = L.stream_chat(L.gateway_body(base), timeout=600)
        next_s = time.perf_counter() - t0
        time.sleep(0.5)
        out[f"model_{name}"] = {
            "model": model, "status": r.status, "error": (r.error_body or "")[:600], "content": r.content,
            "elapsed_s": wrong_s, "log": log[-2500:],
            "next_request_status": r2.status, "next_request_elapsed_s": next_s,
            "next_request_ttft": r2.t_first_token, "next_log": srv.log_since(off)[len(log):][-1500:],
        }
    return out


# ---------------------------------------------------------------- item 3

WAIT_LINE = "好的，我帮您查一下，请稍等。"
RESULT = json.dumps({"found": 1, "rma": "RMA1001", "status": "维修完成", "pickup": "原送修门店"}, ensure_ascii=False)
HINT_RESULT = json.dumps({"found": 1, "status": "维修完成",
                          "hint": "告诉用户维修已经完成，可以到原送修门店取机，然后问还有没有其它问题。"},
                         ensure_ascii=False)


def item3(srv: L.Server) -> dict:
    out: dict = {}
    tools = A["tools"]
    c1 = call("repair_status", {"rma": "RMA1001"}, "call_1")
    c2 = call("kb_search_slow", {"query": "合约机提前解约违约金"}, "call_2")
    ask = u("我的维修单号是RMA1001，帮我查一下进度。")
    raw_cases = {
        "no_user_message": [{"role": "system", "content": A["instructions"]},
                            {"role": "assistant", "content": A["announce"]}],
        "instructions_only": [{"role": "system", "content": A["instructions"]}],
        "system_not_first": aicc_msgs(u("你好"), {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                                      {"role": "system", "content": "通话即将结束，请礼貌道别。"}, u("嗯")),
        "wait_line_between_call_and_result": aicc_msgs(
            ask, {"role": "assistant", "content": "", "tool_calls": [c1]},
            {"role": "assistant", "content": WAIT_LINE},
            {"role": "tool", "tool_call_id": "call_1", "content": RESULT}),
        "tool_result_call_deleted": aicc_msgs(ask, {"role": "tool", "tool_call_id": "call_1", "content": RESULT}),
        "tool_result_call_deleted_after_wait_line": aicc_msgs(
            ask, {"role": "assistant", "content": WAIT_LINE},
            {"role": "tool", "tool_call_id": "call_1", "content": RESULT}),
        "several_results_one_request": aicc_msgs(
            u("帮我查一下RMA1001的进度，再问一下合约机提前解约的违约金。"),
            {"role": "assistant", "content": "", "tool_calls": [c1, c2]},
            {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
            {"role": "tool", "tool_call_id": "call_2",
             "content": json.dumps({"found": 1, "answer": "合约期内提前解约需支付剩余月数月租的30%作为违约金。"},
                                   ensure_ascii=False)}),
        "keypress_note_en": aicc_msgs(u("我想转人工"), {"role": "assistant", "content": "好的，请按0转接人工客服。"},
                                      u("(The caller pressed 0 on their keypad.)")),
        "keypress_note_zh": aicc_msgs(u("我想转人工"), {"role": "assistant", "content": "好的，请按0转接人工客服。"},
                                      u("(用户通过电话按键输入:0)")),
        "keypress_2_note_en": aicc_msgs(u("(The caller pressed 2 on their keypad.)")),
        "dead_air_note": aicc_msgs(u("(The caller has been silent for 8 seconds.)")),
        "system_note_unheard_result": aicc_msgs(
            ask, {"role": "assistant", "content": "", "tool_calls": [c1]},
            {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
            u("(System note: the caller has not heard the result of the lookup yet, not a word of it. In your next "
              "reply to the caller, first respond briefly, then tell them the result.)"),
            u("(The caller pressed 0 on their keypad.)")),
        "tool_result_with_hint": aicc_msgs(
            ask, {"role": "assistant", "content": "", "tool_calls": [c1]},
            {"role": "tool", "tool_call_id": "call_1", "content": HINT_RESULT}),
        "wait_line_then_hint_result": aicc_msgs(
            ask, {"role": "assistant", "content": "", "tool_calls": [c1]},
            {"role": "assistant", "content": WAIT_LINE},
            {"role": "tool", "tool_call_id": "call_1", "content": HINT_RESULT}),
    }
    reps = 3
    for name, msgs in raw_cases.items():
        for mode in ("raw", "gateway"):
            entries = []
            for _ in range(reps if mode == "gateway" else 1):
                body = L.gateway_body(msgs, tools, adapt=(mode == "gateway"), max_completion_tokens=200)
                r = L.stream_chat(body)
                entries.append({"status": r.status, "error": r.error_body, "content": r.content,
                                "tool_calls": [(c["function"]["name"], c["function"]["arguments"])
                                               for c in r.tool_calls],
                                "finish_reason": r.finish_reason, "ttft": r.t_first_token,
                                "prompt_tokens": r.prompt_tokens})
            out[f"{name}__{mode}"] = {"messages_sent": body["messages"][1:], "results": entries}
    # Cue wording for an empty conversation (greeting case); the AICC announce is not in history here.
    cues = {
        "gateway_default": "(The conversation has just started.)",
        "proposed_en": "(The call has connected. Greet the caller as instructed.)",
        "proposed_zh": "(电话已接通，请按指示问候来电者。)",
        "proposed_zh_short": "(来电已接通)",
    }
    cue_out = {}
    for name, cue in cues.items():
        res = []
        for _ in range(5):
            body = L.gateway_body([{"role": "system", "content": A["instructions"]}], A["tools"], adapt=False,
                                  max_completion_tokens=200)
            body["messages"] = L.gateway_messages([{"role": "system", "content": A["instructions"]}], cue=cue)
            r = L.stream_chat(body)
            res.append({"content": r.content, "tool_calls": [c["function"]["name"] for c in r.tool_calls],
                        "ttft": r.t_first_token, "status": r.status})
        cue_out[name] = {"cue": cue, "results": res}
    out["cue_wording"] = cue_out
    # Later system items folded into user notes: the wording of the marker.
    notes = {
        "gateway_default": "(System note: {text})",
        "proposed_zh": "(系统提示：{text})",
    }
    note_out = {}
    for name, fmt in notes.items():
        for text_name, text in (("closing", "通话时间即将用完，请礼貌地结束通话。"),
                                ("info", "来电号码已验证为机主本人。")):
            items = aicc_msgs(u("你好，我想问一下手机怎么截图。"),
                              {"role": "assistant", "content": "同时按住电源键和音量下键就可以截图。"},
                              {"role": "system", "content": text}, u("好的，那怎么录屏呢？"))
            res = []
            for _ in range(3):
                body = L.gateway_body(L.gateway_messages(items, system_note=fmt), A["tools"],
                                      max_completion_tokens=200)
                r = L.stream_chat(body)
                res.append({"content": r.content, "tool_calls": [c["function"]["name"] for c in r.tool_calls],
                            "status": r.status})
            note_out[f"{name}__{text_name}"] = {"format": fmt, "text": text, "results": res}
    out["system_note_wording"] = note_out
    return out


# ---------------------------------------------------------------- item 4

TOOL_CASES = [
    # (id, lang, should_call, expected tool, user text, arg check)
    ("zh_keypress0", "zh", True, "transfer_to_agent", "(用户通过电话按键输入:0)", None),
    ("zh_rma", "zh", True, "repair_status", "我的维修单号是RMA2048，帮我查一下进度。", ("rma", "RMA2048")),
    ("zh_contract", "zh", True, "kb_search_slow", "我想问一下合约机提前解约要交多少违约金？", ("query", None)),
    ("zh_roaming", "zh", True, "kb_search_too_slow", "我下个月去日本，国际漫游怎么收费？", ("query", None)),
    ("zh_bye", "zh", True, "hangup", "好的，没有其他问题了，再见。", None),
    ("en_human", "en", True, "transfer_to_agent", "I want to talk to a real person, please.", None),
    ("en_rma", "en", True, "repair_status", "Can you check my repair? The order number is RMA1001.", ("rma", "RMA1001")),
    ("en_roaming", "en", True, "kb_search_too_slow", "How much does international roaming cost when I travel to Europe?",
     ("query", None)),
    ("en_callback", "en", True, "take_message",
     "Please have someone call me back at 555-0134 about my cracked screen.", ("message", None)),
    ("en_contract", "en", True, "kb_search_slow", "What's the penalty if I end my phone contract early?",
     ("query", None)),
    ("zh_screenshot", "zh", False, None, "手机怎么截图？", None),
    ("zh_slow_charge", "zh", False, None, "我的手机充电很慢怎么办？", None),
    ("zh_repair_no_number", "zh", False, None, "你好，我想查一下维修进度。", None),
    ("zh_weather", "zh", False, None, "今天天气怎么样？", None),
    ("zh_incomplete_rma", "zh", False, None, "单号是RMA，呃，后面的我忘了。", None),
    ("en_power_off", "en", False, None, "How do I turn off my phone?", None),
    ("en_storage", "en", False, None, "My phone storage is full, what can I do?", None),
    ("en_repair_no_number", "en", False, None, "I want to check my repair status.", None),
    ("en_football", "en", False, None, "Who won the football game last night?", None),
    ("en_hello", "en", False, None, "Hello? Is anyone there?", None),
]


def judge(case, r: L.Result) -> tuple[bool, str]:
    cid, lang, should, tool, text, argcheck = case
    names = [c["function"]["name"] for c in r.tool_calls]
    if r.finish_reason == "tool_calls" and not r.tool_calls:
        return False, "silent drop"
    if not should:
        if names:
            return False, f"unexpected call {names}"
        return True, "no call"
    if tool not in names:
        return False, f"expected {tool}, got {names or 'no call'}"
    if argcheck:
        c = next(c for c in r.tool_calls if c["function"]["name"] == tool)
        args = json.loads(c["function"]["arguments"])
        k, v = argcheck
        if k not in args or (v is not None and args[k] != v) or (v is None and not str(args[k]).strip()):
            return False, f"bad args {args}"
    return True, "ok"


ITEM4_VARIANTS = {
    "gateway": (None, None),  # temperature 0.7, top_p 0.8 per request; top_k 20 from the server flag
    "presence_penalty_1.5": (None, {"presence_penalty": 1.5}),
    "temp0": ({"temperature": 0.0, "top_p": 1.0}, None),
    "temp0.3": ({"temperature": 0.3, "top_p": 0.8}, None),
}
VARIANTS = ["gateway", "presence_penalty_1.5"]
EXTRAS = True


def item4(srv: L.Server, reps: int = 3) -> dict:
    out: dict = {"cases": [], "runs": {}}
    for variant in VARIANTS:
        sampling, extra = ITEM4_VARIANTS[variant]
        per_case = []
        for case in TOOL_CASES:
            runs = []
            for _ in range(reps):
                r = L.stream_chat(L.gateway_body(aicc_msgs(u(case[4])), A["tools"], extra=extra, sampling=sampling))
                ok, why = judge(case, r)
                runs.append({"ok": ok, "why": why, "finish_reason": r.finish_reason, "content": r.content,
                             "tool_calls": r.tool_calls, "tool_call_chunks": r.tool_call_chunks,
                             "t_first_tool_call": r.t_first_tool_call, "t_done": r.t_done,
                             "completion_tokens": r.completion_tokens, "leak_tool_markup": "<tool_call>" in r.content
                             or "<function=" in r.content})
            per_case.append({"id": case[0], "lang": case[1], "should_call": case[2], "expected": case[3],
                             "text": case[4], "runs": runs, "ok_runs": sum(x["ok"] for x in runs)})
        scores = [sum(pc["runs"][i]["ok"] for pc in per_case) for i in range(reps)]
        drops = sum(1 for pc in per_case for x in pc["runs"] if x["why"] == "silent drop")
        multi_chunk = sum(1 for pc in per_case for x in pc["runs"] if x["tool_calls"] and
                          x["tool_call_chunks"] != len(x["tool_calls"]))
        out["runs"][variant] = {"per_case": per_case, "score_per_rep_out_of_20": scores, "silent_drops": drops,
                                "calls_split_over_chunks": multi_chunk, "sampling": sampling, "extra": extra}
    if not EXTRAS:
        return out
    # Multi-turn round: call -> result -> spoken answer.
    msgs = aicc_msgs(u("我的维修单号是RMA1001，帮我查一下进度。"))
    r1 = L.stream_chat(L.gateway_body(msgs, A["tools"]))
    rounds = {"first": r1.summary()}
    if r1.tool_calls:
        tc = r1.tool_calls[0]
        msgs2 = msgs + [{"role": "assistant", "content": r1.content, "tool_calls": [
            {"id": tc["id"], "type": "function", "function": tc["function"]}]},
            {"role": "tool", "tool_call_id": tc["id"], "content": RESULT}]
        rounds["second"] = [L.stream_chat(L.gateway_body(msgs2, A["tools"])).summary() for _ in range(3)]
    out["multi_turn"] = rounds
    # tool_choice is ignored by the server.
    tc_out = {}
    for name, choice, text in (("none_on_call_prompt", "none", "我的维修单号是RMA2048，帮我查一下进度。"),
                               ("required_on_chat_prompt", "required", "手机怎么截图？"),
                               ("forced_fn_on_chat_prompt", {"type": "function", "function": {"name": "hangup"}},
                                "手机怎么截图？")):
        res = []
        for _ in range(3):
            body = L.gateway_body(aicc_msgs(u(text)), A["tools"])
            body["tool_choice"] = choice
            r = L.stream_chat(body)
            res.append({"status": r.status, "error": r.error_body, "content": r.content,
                        "tool_calls": [c["function"]["name"] for c in r.tool_calls]})
        tc_out[name] = res
    out["tool_choice"] = tc_out
    # No tools at all (gateway enforcement of tool_choice none) on prompts that would call.
    nt = []
    for case in [c for c in TOOL_CASES if c[2]]:
        r = L.stream_chat(L.gateway_body(aicc_msgs(u(case[4])), A["tools"], tool_choice="none"))
        nt.append({"id": case[0], "status": r.status, "content": r.content, "tool_calls": r.tool_calls,
                   "finish_reason": r.finish_reason,
                   "leak_tool_markup": "<tool_call>" in r.content or "<function=" in r.content})
    out["no_tools_sent"] = nt
    return out


# ---------------------------------------------------------------- item 5

CONV_A = ["你好，我想问一下手机怎么截图。", "那录屏呢？", "好的，谢谢。没别的问题了。"]
CONV_B = ["喂，我手机最近充电特别慢。", "我用的是原装充电器。", "那我是不是要送修？"]


def item5(srv: L.Server) -> dict:
    tag = uuid.uuid4().hex[:6]
    instr = f"(会话编号 {tag})\n" + A["instructions"]
    hist = {"A": aicc_msgs(instructions=instr), "B": aicc_msgs(instructions=instr)}
    turns = {"A": CONV_A, "B": CONV_B}
    log = []
    for i in range(3):
        for conv in ("A", "B"):
            hist[conv].append(u(turns[conv][i]))
            r = L.stream_chat(L.gateway_body(hist[conv], A["tools"]))
            hist[conv].append({"role": "assistant", "content": r.content})
            log.append({"conv": conv, "turn": i + 1, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
                        "uncached": (r.prompt_tokens or 0) - (r.cached_tokens or 0), "ttft": r.t_first_token,
                        "content": r.content, "tool_calls": [c["function"]["name"] for c in r.tool_calls]})
    return {"interleaved": log, "tag": tag}


# ---------------------------------------------------------------- item 6

LONG_GEN = plain("Write a detailed 600-word story about a lighthouse keeper. Do not stop early.",
                 system="You are a storyteller.")
PROBE = plain("Say 'ok' and then count from one to twenty in words.")


def long_prompt(tokens_target: int) -> list[dict]:
    para = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly budget "
            "and the engineers discuss latency, throughput and the cost of memory bandwidth. ")
    # ~30 tokens per paragraph
    n = tokens_target // 30
    text = "".join(f"[{i}] {para}" for i in range(n))
    return [{"role": "system", "content": f"Run {uuid.uuid4().hex}. Read the document."},
            {"role": "user", "content": text + "\n\nIn one sentence: what is the document about?"}]


def probe(n: int = 1) -> list[dict]:
    res = []
    for _ in range(n):
        r = L.stream_chat(L.gateway_body(PROBE, max_completion_tokens=60, sampling={"temperature": 0.0}))
        res.append({"ttft": r.t_first_token, "decode_tps": r.decode_tps, "t_done": r.t_done,
                    "completion_tokens": r.completion_tokens, "cached": r.cached_tokens})
    return res


def item6(srv: L.Server) -> dict:
    out: dict = {}
    L.stream_chat(L.gateway_body(PROBE, max_completion_tokens=60))  # cache the probe prompt
    out["idle_probe"] = probe(5)
    # Reference: a full long generation and its decode speed.
    ref = L.stream_chat(L.gateway_body(LONG_GEN, max_completion_tokens=800))
    out["long_gen_reference"] = {"completion_tokens": ref.completion_tokens, "t_done": ref.t_done,
                                 "decode_tps": ref.decode_tps}
    # (a) close after the first content delta, then probe immediately.
    a = []
    for _ in range(5):
        off = srv.log_offset()
        r = L.stream_chat(L.gateway_body(LONG_GEN, max_completion_tokens=800), close_after="first_content")
        p = probe(1)[0]
        time.sleep(1.0)
        a.append({"closed_at": r.t_first_content, "probe": p, "log": srv.log_since(off)[-1200:]})
    out["close_after_first_content"] = a
    # (b) close during a ~16k prefill (after the first keepalive = first 2048-token step).
    lp_ref = L.stream_chat(L.gateway_body(long_prompt(16000), max_completion_tokens=30), timeout=600)
    out["prefill16k_reference"] = {"prompt_tokens": lp_ref.prompt_tokens, "ttft": lp_ref.t_first_token,
                                   "keepalives": lp_ref.n_keepalive}
    b = []
    for _ in range(3):
        off = srv.log_offset()
        r = L.stream_chat(L.gateway_body(long_prompt(16000), max_completion_tokens=30), close_after="first_keepalive",
                          keep_raw=True)
        t_close = time.perf_counter()
        p = probe(1)[0]
        time.sleep(1.0)
        log = srv.log_since(off)
        b.append({"closed_after_s": r.raw[0].split()[0] if r.raw else None, "probe": p,
                  "progress_lines_after": [ln for ln in log.splitlines() if "Prompt processing progress" in ln][-12:],
                  "log_tail": log[-800:]})
    out["close_during_16k_prefill"] = b
    # (c) close inside a tool-call span (no bytes are written during the span).
    tool_hist = aicc_msgs(
        u("我手机屏幕碎了，上周送修的，单号RMA3344，一直没消息。"),
        {"role": "assistant", "content": "好的，我查一下。"},
        u("算了你们太慢了，我要投诉，给我转人工，把我说的情况都告诉他们：屏幕碎了上周送修，单号RMA3344，"
          "一直没有消息，我打了三次电话都没人处理，我很不满意，希望今天给我答复。"))
    ref_tool = [L.stream_chat(L.gateway_body(tool_hist, A["tools"])) for _ in range(3)]
    out["tool_span_reference"] = [{"t_headers": r.t_headers, "t_first_tool_call": r.t_first_tool_call,
                                   "completion_tokens": r.completion_tokens, "content": r.content,
                                   "tool_calls": r.tool_calls} for r in ref_tool]
    c = []
    for _ in range(5):
        off = srv.log_offset()
        r = L.stream_chat(L.gateway_body(tool_hist, A["tools"]), close_after="first_keepalive", keep_raw=True)
        p = probe(1)[0]
        time.sleep(1.5)
        c.append({"closed_after_s": r.raw[0].split()[0] if r.raw else None, "probe": p,
                  "log_tail": srv.log_since(off)[-1500:]})
    out["close_inside_tool_span"] = c
    return out


# ---------------------------------------------------------------- item 7

CONC_PROMPTS = ["请详细介绍一下手机电池保养的方法，至少说五点。", "请详细说说手机进水后应该怎么处理，至少五个步骤。",
                "Explain in detail how to back up a phone before a repair, with at least five steps.",
                "Explain in detail how to speed up a slow phone, with at least five tips."]


def item7(srv: L.Server, rounds: int = 5) -> dict:
    out: dict = {}
    for k in (1, 2, 4):
        rows = []
        for rd in range(rounds):
            results: list = [None] * k
            barrier = threading.Barrier(k)

            def run(i: int) -> None:
                body = L.gateway_body([{"role": "system", "content": SYS_PLAIN.replace(
                    "Answer in one or two short spoken sentences, in the user's language.", "Answer fully.")},
                    u(CONC_PROMPTS[i % 4] + f"（第{rd}轮）")], max_completion_tokens=256,
                    sampling={"temperature": 0.0})
                barrier.wait()
                results[i] = L.stream_chat(body)

            ts = [threading.Thread(target=run, args=(i,)) for i in range(k)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            for i, r in enumerate(results):
                rows.append({"round": rd, "stream": i, "ttft": r.t_first_token, "decode_tps": r.decode_tps,
                             "t_done": r.t_done, "completion_tokens": r.completion_tokens,
                             "prompt_tokens": r.prompt_tokens, "cached": r.cached_tokens})
        out[f"streams_{k}"] = {
            "rows": rows, "ttft": L.pq([x["ttft"] for x in rows]),
            "decode_tps_per_stream": L.pq([x["decode_tps"] for x in rows]),
            "aggregate_tps_p50": round((L.pq([x["decode_tps"] for x in rows])["p50"] or 0) * k, 1),
        }
    return out


ITEMS = {"1": item1, "2": item2, "3": item3, "4": item4, "5": item5, "6": item6, "7": item7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6", choices=list(L.MODELS))
    ap.add_argument("--items", default="1,2,3,4,5,6,7")
    ap.add_argument("--external", action="store_true", help="use an already running server on :8101")
    ap.add_argument("--external-log", default=None, help="that server's log file")
    ap.add_argument("--item4-variants", default="gateway,presence_penalty_1.5")
    ap.add_argument("--item4-only-scores", action="store_true", help="skip the multi-turn/tool_choice extras")
    ap.add_argument("--suffix", default="", help="appended to the result file name")
    args = ap.parse_args()
    model = L.MODELS[args.model]
    global VARIANTS, EXTRAS
    VARIANTS = args.item4_variants.split(",")
    EXTRAS = not args.item4_only_scores
    srv = L.Server(model, tag=f"s1_{args.model}")
    if args.external:
        srv.log_path = L.Path(args.external_log) if args.external_log else None
    else:
        srv.start()
    try:
        for item in args.items.split(","):
            la0 = L.loadavg()
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] item {item} ...", flush=True)
            data = ITEMS[item](srv)
            res = {"spike": f"S1.{item}", "conditions": {**srv.conditions(), "concurrency": 1 if item != "7" else
                                                         "1/2/4", "loadavg_1m_before": la0,
                                                         "loadavg_1m_after": L.loadavg(),
                                                         "wall_s": round(time.time() - t0, 1),
                                                         "env": L.env_info() if item == "1" else None,
                                                         "footprint_gb": srv.footprint_gb()},
                   "data": data}
            p = L.write_json(f"s1_{item}{args.suffix}_{args.model}.json", res)
            print(f"  -> {p} ({res['conditions']['wall_s']} s)", flush=True)
    finally:
        if not args.external:
            srv.stop()


if __name__ == "__main__":
    main()

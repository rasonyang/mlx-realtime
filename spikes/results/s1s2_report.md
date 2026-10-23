# S1 / S2: LLM behaviour and performance (B1, 2026-10-23)

## Conditions

- Server: stock `mlx_lm.server` 0.32.2 (mlx 0.32.4), :8101, `HF_HUB_OFFLINE=1`, `--model <snapshot dir>`.
- Models: Qwen3.6-35B-A3B-4bit `38740b84` and Qwen3.5 `1e20fd8d`, 20.8 GB each.
- Requests are gateway-shaped (chat_wire.py): temperature 0.7, top_p 0.8, `max_completion_tokens` 512.
- AICC prompt: `line_session_zh.json`, about 1570 tokens.
- **Power:** AC `powermode 0` (Automatic), not 2 as briefed.
- **No GPU step-down:**
  - prefill held 920–1140 tok/s through the 12-minute grid;
  - 3 minutes of sustained decode went 76 → 81 tok/s (×1.06);
  - an earlier sustained run (×0.90, dips to 40–47 tok/s) coincided with external CPU load.
- **External CPU load:** someone else's Go stress tests (`test-stress.sh`) pushed the load to 12–68 from about 11:38 to 12:00 and again from about 12:15. Each file records the load before and after.

## S1: behaviour

**1. Thinking.**

- 20/20 zh/en replies had no `delta.reasoning` and no `<think>`, on both models. The server flag alone was 2/2.
- Per-request `enable_thinking: true` turns thinking back on (6/6). The content then starts with `"\n\n"`.

**2. Fields.**

- **`include_usage`:** the usage chunk (`choices: []`, with `cached_tokens`) comes before `[DONE]`.
- **`stream_options: {}`:** the stream ends after the finish chunk, with no usage and no `[DONE]`.
- **Honoured:** `max_completion_tokens`, `max_tokens`, `temperature` (at 0 the output is identical 4/4) and `top_p`.
- **Out-of-range values → 400** (for example "top_p must be at most 1").
- **`presence_penalty`:** accepted, but there is no server flag and it only covers the last 20 tokens.
- **`model`:**
  - omitted, `default_model` or the exact `--model` string: no reload;
  - a wrong name: 404 (offline HF error) **after unloading the model**, so the next request reloads it (1.4 s with a warm page cache);
  - the repo id instead of the path: two full reloads.

**3. Template edge cases.**

- 404 cases:
  - no user message: `No user query found in messages.`;
  - a system message that is not first: `System message must be at the beginning.`
- Gateway adaptation fixes both, and the model follows a folded system note.
- AICC history shapes, all 200 with correct replies:
  - a wait line between `tool_calls` and the result;
  - a result whose call was deleted;
  - two results, then one request;
  - keypress notes (key 0 → transfer, 8/8);
  - dead air;
  - "(System note …)";
  - a `"hint"` result (followed 8/8).
- Cue wording:
  - "(The call has connected. Greet the caller as instructed.)" gave the full greeting 5/5;
  - the current "(The conversation has just started.)" dropped the line name once in 5;
  - keep "(System note: {text})" as the folding marker.

**4. Tool calls** (6 tools, 20 prompts, 10 of them should call):

- **Delta shape.** Each call arrives complete in one delta: `{id: uuid4, type, index, function: {name, arguments: JSON string}}`. Arguments are typed from the schema.

| Score /20 per repetition | Qwen3.6 | Qwen3.5 |
|---|---|---|
| temp 0.7 (gateway) | 19, 16, 19, 18, 17, 19 | 20, 20, 17 |
| + presence_penalty 1.5 | 17, 17, 19 | – |
| temp 0.3 | 19, 18, 18 | – |
| temp 0 | **19, 19, 19** | **19, 19, 19** |

- **Constant miss on both models:** a bare "(用户通过电话按键输入:0)" is answered with "正在为您转接人工客服" and no call.
- **Qwen3.6 silent drops (hard defect).**
  - On the realistic turn "转人工，把情况说清楚", `finish_reason: tool_calls` came with no call in 17/150 runs (11%). In the warm shape it was 16/90 (18%). Qwen3.5: 0/80.
  - The DEBUG token log shows the cause: the model writes `<tool_call>\n<transfer_to_agent>\n<parameter=…`, without `function=`, and the qwen3_coder parser drops it.
  - Rewriting `<NAME>` → `<function=NAME>` (NAME a declared tool) recovers 7/7 offline.
- **Qwen3.5 silent non-calls.** It said "好的，我查一下。" and stopped with no call on 6/40 RMA lookups (Qwen3.6: 0/40). It also put that preamble before calls 7 times. The gateway cannot detect a non-call.
- **Other checks:**
  - multi-turn (call → result → answer) is correct;
  - `tool_choice` is ignored;
  - with no `tools` sent, the model never calls, but it may claim to act, and once it put a JSON pseudo-call into the content.

**5. Prompt cache** (A/B interleaved):

| Turn | A1 | B1 | A2 | B2 | A3 | B3 |
|---|---|---|---|---|---|---|
| cached / prompt | 0/1579 | 1530/1580 | 1575/1609 | 1576/1627 | 1605/1644 | 1623/1684 |
| TTFT (s) | 1.39 | 0.13 | 0.10 | 0.13 | 0.49 (tool) | 0.14 |

- The system prompt and tools are reused across calls.
- Checkpoints are saved at the system end, at the last user turn and at the full sequence, and only when the last message is `user`.
- The system checkpoint includes the next role's header token. So `[system, cue]` does not seed a history that starts with the assistant announce.
- `--prompt-cache-size` defaults to 10 entries, about 3 requests' worth. All runs used the default.

**6. Cancellation** (probe TTFT; idle is 0.027 s):

| Cut point | Probe TTFT after the cut | What happens |
|---|---|---|
| After the first delta | 0.47–0.55 s | The server stops on the next write, but the generation loop's 0.5 s `TimeBudget` delays every new request |
| During a 16k prefill | 5.4–6.8 s | Prefill runs 1–2 more 2048-token steps |
| Inside a tool-call span | 0.55–0.75 s, probe decode batched | Not stopped: no writes happen during the span |

**7. Concurrency.** Requests are batched.

| Streams | TTFT p50 (3.6 / 3.5) | tok/s per stream | Aggregate tok/s |
|---|---|---|---|
| 1 | 0.11 / 0.08 s | 79 / 89 | 79 / 89 |
| 2 | 0.17 / 0.13 s | 58 / 70 | 116 / 141 |
| 4 | 0.37 / 0.32 s | 39 / 47 | 155 / 189 |

## S2: performance (n=20 unless stated; p50 / p95 in s)

**Grid** (cold = unique prefix, warm = repeated):

| Prompt | 3.6 cold | 3.6 warm | 3.6 tok/s | 3.5 cold (n=10) | 3.5 tok/s |
|---|---|---|---|---|---|
| 270 | 0.47 / 0.60 | 0.03 / 0.06 | 78 | 0.44 / 0.47 | 84 |
| 940 | 1.02 / 1.10 | 0.03 / 0.03 | 78 | 1.02 / 1.13 | 84 |
| 3.8k | 3.32 / 3.60 | 0.04 / 0.05 | 78 | 3.44 / 3.59 | 79 |
| 15.4k | 16.7 / 17.8 | 0.07 / 0.10 | 66 | 16.7 / 17.4 | 67 |

**AICC prompt, spoken replies** (TTFT; first `，。！？,.!?`):

| Case | 3.6 TTFT | 3.6 boundary | 3.5 TTFT | 3.5 boundary |
|---|---|---|---|---|
| 1. cold | 1.49 / 1.63 | 1.57 / 1.72 | 1.44 / 1.60 | 1.53 / 1.75 |
| 1b. new call, cached instructions (n=19) | 0.13 / 0.23 | – | 0.14 / 0.16 | – |
| 2. warm, one turn appended (n=33/38) | 0.14 / 0.18 | 0.22 / 0.33 | 0.15 / 0.21 | 0.24 / 0.33 |
| 3. instructions change, unseen (n=10) | 1.50 / 1.64 | 1.57 / 1.84 | 1.77 / 1.99 | 1.85 / 2.07 |
| 4. per-response override (n=15/12) | 0.24 / 1.15 | 0.31 / 1.22 | 0.26 / 1.08 | 0.34 / 1.13 |
| 4 → the next normal response | 0.17 / 0.24 | 0.24 / 0.38 | 0.19 / 0.23 | 0.25 / 0.37 |

- In case 3, half of the replies were `hangup` calls.
- In case 4 the override replaces the system message (about 1000 tokens). It costs 1.1 s when new, about 0.25 s when already seen.

**5. Warming after an unseen change** (3.6, spoken TTFT):

| Variant | TTFT |
|---|---|
| None (n=9) | 1.79 / 1.89 |
| **W2** (new system + history up to the last user turn; request sent after its `[DONE]`) | **0.15 / 0.21** (n=9) |
| W2 sent concurrently | 2.93 / 3.27 |
| W2 + 0.3 s | 1.40 / 1.48 (n=5) |
| W2 + 0.8 s | 0.84 / 0.90 (n=5) |
| W3 (system + cue) | 1.58 / 1.71 (shape mismatch) |

The W2 warm-up itself takes 1.68 s. Warming only pays off when it finishes first; the gateway must wait for it rather than race it.

**Tool-only `t_done`:**

| Call | 3.6 warm | 3.6 after change | 3.5 warm | 3.5 after change |
|---|---|---|---|---|
| repair_status | 0.55 / 0.88 | 1.94 / 2.08 | 0.66 / 0.72 | 2.27 / 2.40 |
| transfer_to_agent (about 100 tokens; n=60 / 40) | 1.41 / 1.68 | **2.65 / 3.12** (max 3.69) | 1.49 / 1.88 | **2.75 / 3.22** |

**Idle 15 minutes** (3.6, n=1 each):

- First request after the idle gap: TTFT 0.21 s by default, 0.056 s with `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000`. The warm TTFT is 0.03 s.
- This comparison is confounded: the load was about 25 during the default run and about 4 during the residency run.

## G2

- **Model.** `mlx-community/Qwen3.6-35B-A3B-4bit` @ `38740b847e4cb78f352aba30aa41c76e08e6eb46`, **on condition that** `backends/llm` adds a lenient qwen3_coder parser. Without it, use Qwen3.5 @ `1e20fd8d42056f870933bf98ca6211024744f7ec`.
- **The bar:**
  - thinking fully off: yes;
  - ≥ 19/20 tool calls: only at temperature 0;
  - a hard defect: 11% of transfers are silently dropped by the stock parser.
- **Why Qwen3.6 is still preferred:** its failure is detectable and fixable, while Qwen3.5's non-calls are invisible. TTFT is equal; Qwen3.5 decodes 7–12% faster.
- **Flags:**

  ```
  --chat-template-args '{"enable_thinking":false}' --max-tokens 2048 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0
  ```

- **Per request:** temperature 0.0 and top_p 1.0 when tools are attached; no presence_penalty.
  - presence 1.5 and repetition 1.0 from the card have no server flag.
- **Gateway changes:**
  - retry once on a dropped call;
  - use the new cue;
  - filter JSON pseudo-calls when no tools are sent.

## G4: the LLM part

- **3 s to first audio = ASR 0.3–0.5 s + first LLM boundary + TTS first chunk 0.12–0.19 s.**
  - Warm: boundary 0.33 s p95. Safe.
  - Cold, an instructions change or a new override: boundary 1.72–1.84 s p95 (2.07 s on 3.5), so about 2.6 s in total before S5 contention. **At risk.**
  - W2 warming fixes it when the caller's turn arrives at least 1.7 s after the change.
  - The 0.5 s `TimeBudget` adds up to 0.5 s whenever another stream is decoding.
- **3 s for a tool-only response:**
  - `repair_status` is safe;
  - **`transfer_to_agent` after a change misses it** (3.12 s p95), and a retry cannot fit;
  - mitigations: W2 warming and a shorter summary.

## Reproduce

```
export SPIKE_LOG_DIR=<scratch>; R="uv run --project spikes/llm python spikes/llm"
$R/s1_behaviour.py --model qwen3.6 --items 1,3,4,5,6,7,2
$R/s1_behaviour.py --model qwen3.6 --items 4 --item4-variants gateway,temp0,temp0.3 --item4-only-scores --suffix _sampling
$R/s1_behaviour.py --model qwen3.5 --items 1,4,7 --item4-variants gateway,temp0
$R/s2_perf.py --model qwen3.6 --parts first,aicc,warming,grid,sustained,transfer   # transfer runs renamed _runN
$R/s2_perf.py --model qwen3.5 --parts first,aicc,transfer; $R/s2_perf.py --model qwen3.5 --parts grid --n 10
$R/s2_perf.py --model qwen3.6 --parts idle --idle-min 15 [--residency-ms 1000]
$R/s1_toolparse_probe.py --model qwen3.6 --n 30; $R/s1_toolparse_lenient.py; $R/summarize.py qwen3.6
```

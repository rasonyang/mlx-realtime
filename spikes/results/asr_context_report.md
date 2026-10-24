# ASR context biasing (issue #1)

2026-10-24. Scripts: `spikes/asr_context/` (`corpus.py`, `synth.py`, `degrade.py`, `harness.py`, `analyze.py`).
Raw results: `spikes/results/asr_context_patched.json` (full run), `asr_context_noex.json` (the no-example context),
`asr_context_latency.json`, `asr_context_huge.json`, and the `asr_context_sanity_*.json` files (10-clip check on the stock
server and on the patched one). Audio and clips are in the git-ignored `spikes/asr_context/data/`.

Backends: stock `mlx_audio.server` (mlx-audio 0.6.1) for the baseline; `python -m mlxrt_audio.serve` (printed
`mlxrt_audio: ASR context field -> system_prompt (mlx-audio 0.6.1)`) for everything else. Model
`mlx-community/Qwen3-ASR-1.7B-8bit`, `language` omitted (auto), as the gateway sends it. Transcripts are deterministic:
`none` re-run gave the identical 68.7 % / 5.6 % twice.

## What ran

- **Corpus:** 240 synthetic Chinese utterances with English / alphanumeric terms (brand 60, product 70, order 50, codeswitch 60;
  252 term occurrences), 4 Qwen3-TTS CustomVoice speakers, 5 degradations (48 clips each: clean, g711, noise, onset, echo);
  plus 84 ASCEND human rows (42 recordings, wideband and g711) with no listed term. Total 324 clips.
- **Degradations:** see the docstring of `degrade.py`. g711 is the AICC to gateway path (soxr 8 kHz, mu-law, AICC per-frame
  upsampler, `InputDecoder` 24 to 16 kHz); noise is g711 plus real babble / room noise at 10-20 dB SNR; onset removes 60-140 ms
  after the speech onset; echo runs the caller over the tail of a bot answer through the gateway's own echo stage (Speex ASR path).
- **Modes:** `none` (no field), `context` (311 chars: description sentence + term list + order-number format examples),
  `terms` (bare comma-separated list, 126 chars), `empty` (`context=""`), `en_prompt` (English description, no term), `long1k`
  (967 chars: `context` padded with unused glossary lines), `context_noex` (`context` without the format examples).
- **Metrics:** term hit, CER and non-term CER as defined in `common.py` (case, spaces, punctuation ignored; Chinese digit
  characters mapped to 0-9; no other number verbalization). Differences are paired over clips, 95 % bootstrap CI (2000 resamples).

## Headline (synthetic set, 240 clips, 252 terms)

| mode | term hit | CER | non-term CER | ASCEND CER (84 clips) |
|---|---|---|---|---|
| none | 68.7 % (173/252) | 5.6 % | 0.9 % | 9.2 % |
| context | 82.1 % (207) | 4.2 % | 0.8 % | 9.5 % |
| context_noex | **85.3 % (215)** | **2.9 %** | **0.6 %** | 9.7 % |
| terms | 84.5 % (213) | 6.1 % (one runaway, below) | 1.2 % | 9.9 % |
| long1k | 83.7 % (211) | 3.9 % | 0.8 % | 12.0 % |
| en_prompt | 69.8 % (176) | 5.4 % | 0.6 % | 10.6 % |
| empty | 68.7 % (173) | 5.6 % | 0.9 % | 9.2 % |

Paired difference against `none` (95 % CI):

| mode | term hit | CER | non-term CER | ASCEND CER |
|---|---|---|---|---|
| context | +13.5 pts [+8.1, +19.1] | -1.45 pts [-2.9, +0.2] | -0.09 pts [-0.5, +0.4] | +0.33 pts [-0.8, +1.4] |
| context_noex | +16.7 pts [+11.8, +22.1] | -2.73 pts [-3.6, -1.9] | -0.30 pts [-0.6, -0.04] | +0.52 pts [0.0, +1.2] |
| terms | +15.9 pts [+11.4, +20.9] | +0.47 pts [-3.0, +7.1] | +0.26 pts [-0.4, +1.3] | +0.72 pts [-0.05, +1.5] |
| long1k | +15.1 pts [+9.9, +20.8] | -1.74 pts [-3.1, -0.05] | -0.09 pts [-0.5, +0.4] | +2.77 pts [-0.2, +6.9] |
| en_prompt | +1.2 pts [-2.0, +4.7] | -0.23 pts [-0.7, +0.3] | -0.26 pts [-0.5, 0.0] | +1.38 pts [-0.9, +5.1] |
| empty | 0 (identical on 324/324 clips) | 0 | 0 | 0 |

## By category and degradation (term hit rate; n per row in `asr_context_patched.json`)

| category (n terms) | none | context | context_noex | terms | long1k | en_prompt |
|---|---|---|---|---|---|---|
| brand (60) | 55.0 % | 88.3 % | 88.3 % | 88.3 % | 91.7 % | 55.0 % |
| product (70) | 64.3 % | 81.4 % | 82.9 % | 80.0 % | 82.9 % | 67.1 % |
| codeswitch (72) | 68.1 % | 83.3 % | 84.7 % | 83.3 % | 83.3 % | 73.6 % |
| order (50) | 92.0 % | **74.0 %** | 86.0 % | 88.0 % | **76.0 %** | 86.0 % |

(Rates are pooled over term occurrences; the codeswitch row includes the 12 two-term sentences.)

| degradation | none | context | context_noex | terms | long1k |
|---|---|---|---|---|---|
| clean | 61.2 % | 77.6 % | 81.6 % | 79.6 % | 83.7 % |
| g711 | 67.3 % | 84.6 % | 86.5 % | 88.5 % | 84.6 % |
| noise | 68.6 % | 84.3 % | 88.2 % | 88.2 % | 88.2 % |
| echo | 74.0 % | 90.0 % | 92.0 % | 90.0 % | 92.0 % |
| onset | 72.0 % | 74.0 % | 78.0 % | 76.0 % | 70.0 % |

Context helps in every degradation; on `onset` (first syllable cut) the gain is small for `context` (+2 pts) and larger without the examples (+6).

Non-term CER by degradation (none / context / terms / long1k): clean 0.22 / 0.22 / 2.44* / 0.22, g711 0.0 / 1.1 / 0.0 / 1.3,
noise 1.28 / 1.28 / 1.49 / 1.06, echo 0.22 / 0.22 / 0.22 / 0.22, onset 2.75 / 1.27 / 1.69 / 1.27 (percent).
*= the `terms` runaway on one clip (u123).

## What context fixed (45 term occurrences with `context`, 46 with `context_noex`; 11 and 4 broken)

By term (`context`): Veltra 9, Orbix 6, 5G 尊享 299 5, NovaNet 5, NovaNet Max 4, Ping 值 3, eSIM 2, VoLTE 2, Zentra 2, and one each
of Wi-Fi 6, Wi-Fi 7, PIN 码, 5G 畅享 199, SkyWave, NovaCloud, Mesh 路由器. Examples:

| ref | none | context |
|---|---|---|
| 我说，Orbix 是做什么的？ (g711) | 我说，Orbex是做什么的？ | 我说，Orbix 是做什么的？ |
| 请问 NovaNet Max 怎么开通？ (clean) | 请问 NoWayNet Max 怎么开通？ | 请问 NovaNet Max 怎么开通？ |
| Wi-Fi 6 覆盖我们小区吗？ (clean) | 微发六覆盖我们小区吗？ | Wi-Fi 6 覆盖我们小区吗？ |
| Ping 值是不是要重新弄一下？ (noise) | 品值是不是要重新弄一下？ | Ping 值是不是要重新弄一下？ |
| NovaCloud 覆盖我们小区吗？ (onset) | Overcloud覆盖我们小区吗？ | NovaCloud 覆盖我们小区吗？ |
| NovaNet 的客服电话是多少？ (echo) | Novinet的客服电话是多少？ | NovaNet 的客服电话是多少？ |
| Veltra 是做什么的？ (g711) | Velcro 是做什么的？ | Veltra 是做什么的？ |

The two real questions, as synthesized (the real failure was a human on a phone; these were already mostly right without a
context, see below): u218 "NovaNet 是做什么的？" onset: none "Nova Net是做什么的？" (counts as a hit), context "NovaNet 是做什么的？";
u159 "我说，NovaNet 是做什么的？" echo: none "Novanet", context "NovaNet". No clip reproduces "那未来他" or "Louisette".

## What context broke

1. **Order numbers (the one real regression).** With `context` order-number hit falls 92 % to 74 % (9 of the 11 broken terms; `long1k`
   76 %). In 8 of those the model wrote the format example from the context instead of what it heard:
   `NV20241018` heard, written `NV20240101`; `TK-20328` written `TK-20240101`; on u221 the whole example list
   `NV20240101、TK-10001、SO-100-200、RMA-0001` was echoed as the transcript. Removing the examples (`context_noex`) removes all of
   these, leaving 3 order breaks (digits turned into words: `安徽二零二四…`, `NVR 零二四…`, `NN905EX`), order hit 86 % vs 92 % for
   `none`. Even then `none` is better on order numbers: the baseline already writes them as Chinese digits that the metric
   accepts, so there was nothing to gain, only noise to add. Do not put concrete codes in the context.
2. **Two 5G 畅享 199 onset/echo clips:** `唱享` / `声享` for `畅享` with `context` (1 left with `context_noex`).
3. **Bare list regurgitation.** `terms`: on u123 (clean, "Veltra 的 5G 畅享 199 我昨天刚装好，还是不行。") the transcript continued with the whole
   term list (167 characters). It is a single clip of 240, but it shows a bare list can be copied out. The descriptive sentence
   (`context`, `long1k`) did not do this.
4. **ASCEND (no listed term).** No listed term was hallucinated into any of the 84 clips (checked: none of the context vocabulary
   appears in a hypothesis where it was not in `none`). CER moves by +0.3 to +0.7 pts for the 300-char contexts (CIs include 0, the
   `context_noex` lower bound is 0.0), +1.4 for `en_prompt`, +2.8 for `long1k`; clips made worse / better: context 8 / 3, terms 9 / 2,
   en_prompt 8 / 4, long1k 10 / 2. Most differences are surface form only ("budget" to "budget", spacing, "Self-driving"
   capitalisation, which the normalization removes). The real changes are a few translation or substitution errors, e.g. mix_027_g711
   "做 optimization" to "Do model optimization," (7 edits), mix_021_g711 "我并不是要做" to "我并不知道做" (5 edits), and a few
   fixes (mix_005_g711 "哈维尔" to "Hardware", 8 edits). Short human clips are the least certain evidence here.
5. **Context in the wrong language** (`en_prompt`, no terms): no effect on terms (+1.2 pts, CI includes 0), non-term CER -0.26 pts, but
   ASCEND CER +1.4 pts; 167/324 transcripts are identical to `none`, so it changes output without helping. It is harmless on the synthetic set
   and slightly worse on human speech.

## Latency (client-side per turn, ms; 60 turns per cell, 2 rounds; stable on this machine, no other ASR load)

| sessions | mode | p50 | p95 | max |
|---|---|---|---|---|
| 1 | none | 169 | 237 | 289 |
| 1 | context (311 chars) | 231 | 291 | 304 |
| 1 | long1k (967 chars) | 339 | 417 | 423 |
| 2 | none | 320 | 397 | 436 |
| 2 | context | 483 | 563 | 592 |
| 2 | long1k | 695 | 808 | 843 |
| 3 | none | 460 | 699 | 740 |
| 3 | context | 693 | 947 | 997 |
| 3 | long1k | 1034 | 1432 | 1468 |

The context is prefilled on every request, so it costs time per turn: about +60 ms (300 chars) and +170 ms (about 1000 chars) for one
session, and +230 / +570 ms at three sessions on p50, because the three requests queue on one GPU. A context of about 300 characters
keeps three-session p95 under 1 s; 1000 characters pushes it to 1.4 s. The gateway's `asr.timeouts.total_ms` is 10 s, so nothing
times out, but turn latency matters for the voice loop. Keep the context short. No request failed (0 errors in 540 requests). The stock
server (sanity run) had the same latency with and without the field, as it ignores it.

## Failure paths

| issue's case | observed |
|---|---|
| No prompt | Unchanged from today by definition (the `none` mode is the stock behavior); on the 10 sanity clips `none` scored the same on the stock and the patched backend (75.0 % hit, 1.4 % CER). |
| Empty prompt | `context=""` is byte-identical to `none` on 324/324 clips (the patched backend ignores it). |
| Different-language prompt | No term effect, non-term CER -0.26 pts [-0.5, 0.0], ASCEND CER +1.4 pts [-0.9, +5.1] (n.s.). No crash. |
| ~1000-char prompt | Hit +15.1 pts, but order numbers 76 %, ASCEND CER +2.8 pts (n.s., largest of any mode), and +170 ms per turn. |
| 10 KiB prompt sent straight to the backend | HTTP 200, no rejection, no truncation visible; the transcripts were right (the 3 clips tried), but the turn took about 1.3 s (1323 / 1319 / 1422 ms vs about 150-200 ms) and the server was healthy after (`/v1/models` 200, a plain request 153 ms). The backend does not enforce a limit; the gateway's 1024-character cap is the only guard, and it matters for latency, not only for safety. |
| Prompt changes mid-session | Not tested here (this harness measures the backend only, not session state). |

## Verdict against "Done when"

- **Term hit rate rises clearly: yes.** 68.7 % to 82-85 % with a 300-char context (+13.5 pts [+8.1, +19.1] for `context`, +16.7 pts [+11.8, +22.1]
  without the format examples). Brand names 55 % to 88 %; the gain holds across voices and degradations except onset-clipped clips (+2 to +6 pts there).
- **Non-term CER does not regress: yes on the synthetic set** (-0.09 pts [-0.5, +0.4] for `context`, -0.30 pts [-0.6, -0.04] for `context_noex`;
  overall CER improves too). **Not shown on human speech:** ASCEND CER is +0.3 to +0.7 pts for the short contexts (CI includes 0 or touches 0.0),
  +2.8 pts for the 1000-char one. This is a small, unproven cost of about half a point on 84 short clips; watch it.
- **Failure paths: partly.** Empty equals none and the wrong-language prompt does no harm to terms. The oversized-prompt rejection or truncation is not the
  backend's job today (it accepts 10 KiB and is 8x slower), so it depends on the gateway's 1024-character cap. The order-number regression
  is a context-content problem, not a failure path, but it is the main way a deployment could make things worse.

Recommended context: the descriptive sentence plus the term list (about 300 characters), with no concrete order-number examples (describe them in words, or
leave them out). The bare list works about as well for hits but copied itself into one transcript.

## Caveats, said plainly

- **Synthetic-voice ceiling.** All 240 term utterances are TTS. Qwen3-TTS says "NovaNet" clearly and consistently, so the baseline
  is already 68.7 % and both of the issue's own questions came out nearly right without a context (hit by the metric's rule). The real
  failures ("那未来他", "Louisette") came from a human on a phone line; no clip here fails that way. The gain is therefore measured on
  mishearings like Orbex / Velcro / NoWayNet / 品值, which are the same kind of error but may not be the same size. Expect the effect on real
  callers to differ in either direction.
- **No human recordings with terms.** The ASCEND clips carry no listed term, so they test regression only. A real check needs recorded
  calls (or recordings of people reading the corpus) over the phone path.
- **Noise.** There was no dedicated noise corpus in the datasets. The noise clips use babble from real recordings and room-noise frames from
  ASCEND recordings, so they are real but not the usual street / cafe set.
- **Metric limits.** A digit string heard as Chinese numerals counts as a hit (the baseline's habit for orders); "5G 畅享 199" heard
  as "五G畅享一九九" does not (the digits-as-words mapping covers only single-digit characters). Case-insensitive: "Novanet" is a hit.
  Samples are small per cell (11-14 clips, 10-20 terms), so slice rates carry about +-10 pts of noise; read the pooled rows.
- **Echo clips** use the gateway's real echo stage on a simulated AICC chain (G.168 paths), not recorded calls.

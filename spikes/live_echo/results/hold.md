# Hold measurement (issue #5)

Measured on the simulated AICC call (tests/phone.py), in real time, with the real Qwen3-ASR-1.7B-8bit and Qwen3-TTS (bot voice `vivian`), Silero and Smart Turn as in config.yaml, the canceller as configured (in its no-echo bypass at the start of 31/36 default-model and 15/18 mild-model baseline bursts). Two gateways in one process, baseline (`confirm_while_playing: false`) and fix (true), alternating chunk by chunk. Everything below is a simulation: the residual-echo bursts follow a model I chose (see Setup), not a measured phone AEC, and the result for A depends on it.

## Findings

1. **Real barge-ins are safe and cheap.** 40 of 40 real barge-ins (B 30 on a line without echo, Be 10 over steady echo) were released as `caller` by the first or second check; no real turn was dropped. The hold adds P50 155 ms / P95 348 ms (`held_ms`, B); paired against the baseline, onset -> speech_started is +150 / +342 ms (P50 / P95). Committed-turn transcripts: identical in 23/30 (B) and 5/10 (Be); the rest differ in a word at the edges, because the turn's start moves with the VAD from run to run (the baseline's own variation); no systematic loss.
2. **Echo bursts: the fix helps, but only as far as the ASR recognises the leak.** Self-interruptions per burst, baseline -> fix: default (strongly processed) model 24/36 = 67 % -> 17/36 = 47 %; mild model 14/18 = 78 % -> 6/18 = 33 %. Mid-answer bursts (A3) are not helped at all under the default model (9/12 -> 10/12). Under the default model the ASR hears the first 0.7-1 s of a leak as unrelated words ("对于", "病情重的", "桥市路"): 18 of 23 held bursts were released as `caller`, 10 of them on the very first check. With a milder leak the ASR text matches the played text (median share of units found 0.67-1.0) and the hold drops 5 of 14 as `echo`, but still releases 6.
3. **Short caller utterances are delayed to the end of the turn.** Of 22 short utterances (11 words x 2), 14 never fired the VAD gate in either condition (speech under 400 ms during playback: existing rule). Of the 8 that did, "等一下" (3 units) was released early at ~0.6 s, but "喂", "好的", "hello" (1 to 2 units, or 2 units of which one occurs in the played text) were released only at the turn end: onset -> speech_started 3.1-4.1 s against ~0.45 s in the baseline, while AICC keeps playing. The turn end of a one-word utterance came 3.0 s after the speech (Smart Turn's 3000 ms silence fallback). N=1 would have released all 8 at ~0.62 s.
4. **Noise.** None of 30 synthesised sounds (cup, keys, clicks; peaks -3, -12, -24 dBFS) fired the gate in either condition, so the hold had nothing to decide: no data on noise, and no regression.
5. **Cost.** Check latency (request -> result) P50 154 ms / P95 187 ms. ASR checks per held turn: 1.1 for real barge-ins, 2.3 for default-model bursts, 5.9 for mild bursts and 10.8 for short utterances (the checks go on every 250 ms through the silence until the turn ends: up to 15 requests of 4 s windows for a 0.6 s "喂"). Ordinary turns over silence are unchanged (speech_stopped -> transcript P50 225 vs 216 ms, a difference within run noise).

## Observations that look like defects or weaknesses of the fix

- **Release at `playback_ended` of an echo turn (5 of 5 cases).** A1 bursts and Am bursts whose echo outlasts the playback were released with `reason: playback_ended` although every check matched the played text; the turn was committed as an echo turn right after. A `speech_started` therefore reaches the client for pure echo, which is what issue #5 wanted to prevent whenever AICC still has audio queued that the playout model does not know about. Example (Am00, fix): `hold_check ... "text": "给您一个明确的答复。", "unmatched": 0` at t=19704.5, then `{"t": 19837.8, "kind": "hold_release", "reason": "playback_ended", "held_ms": 1317}` and `speech_started ... "barge_in": false, "held_ms": 1317`, then `echo_turn` at 20470.8. Suggestion: at playback end keep holding while the checks still match, and let the final check drop it.
- **Checks continue through the post-speech silence.** C18 ("喂", 0.58 s): `hold` at 137616.5, the speech ends at 137568 (`hold_end audio_end_ms 137568` only arrives at 140596.7), and in between 12 identical `hold_check "text": "喂。", "units": 1, "unmatched": 1, "verdict": "hold"` at 250 ms steps, then the final check and `hold_release reason turn_end held_ms 3079`. A check whose window has no new voiced audio, or whose text has not changed, is wasted ASR load (the ASR is shared by 4 sessions).
- **A 1-unit utterance can never pass N=2.** `unmatched >= confirm_min_new_units` cannot hold for a one-word transcript, so "喂" waits for the turn end by construction.
- **The unmatched count is a subsequence match.** "好的" had `unmatched: 1` against a played text that contains 的 and 好 in other places, so a 2-unit utterance counts as 1 new unit. Short transcripts are matched by chance against 20-40 characters of Chinese text.
- The early check is unreliable on degraded audio: 10 of the 18 false releases (default model) were decided on the first check, which covers about 0.7 s of audio.


## Setup

- **models**: {"asr": "mlx-community/Qwen3-ASR-1.7B-8bit", "tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "bot voice": "vivian", "vad": "Silero + Smart Turn (config.yaml turn:)"}

- **pmset -g** (at the start of the run):

```
System-wide power settings:
 SleepDisabled		0
Currently in use:
 standby              1
 Sleep On Power Button 1
 hibernatefile        /var/vm/sleepimage
 powernap             1
 networkoversleep     0
 disksleep            10
 sleep                1 (sleep prevented by caffeinate, caffeinate, caffeinate)
 hibernatemode        3
 ttyskeepawake        1
 displaysleep         10
 tcpkeepalive         1
 powermode            0
 womp                 1

```

- **git**: HEAD 7920a38 plus the uncommitted fix (9 changed paths under src/ and config.yaml)

- **config**: {"barge_in (fix)": {"echo_guard_ms": 2000, "min_speech_ms_while_playing": 400, "echo_turn_suppression": true, "echo_match_threshold": 0.7, "echo_match_min_units": 4, "echo_match_window_ms": 3000, "confirm_while_playing": true, "confirm_min_new_units": 2, "confirm_step_ms": 250, "confirm_window_ms": 4000, "confirm_timeout_ms": 2000}, "aec": {"enabled": true, "engine": "webrtc_aec3", "asr_input": "speex"}, "output.max_lead_ms": 10000}

- **seed**: 20261029

- **burst_model**: {"band_hz": [300.0, 2400.0], "clip_rel": 0.7, "gain_sigma_db": 5.0, "gain_rho": 0.5, "dropout_p": 0.1, "dropout_db": -25.0, "ir_taps": 48, "delays_ms": [316, 506], "levels_db": [-5, -20, -35], "length_s": "1.5-2.5"}

- **started**: 2026-10-29 16:40:55 +0800

## A. Residual-echo bursts, default (strongly processed) leak model (no steady echo)

Bursts run: 72 (baseline A1 12, baseline A2 12, baseline A3 12, fix A1 12, fix A2 12, fix A3 12); trials with an error: 0.

Canceller bypass at the start of each burst's window: {('baseline', True): 31, ('baseline', False): 5, ('fix', True): 32, ('fix', False): 4}.

Self-interruption = AICC flushed unplayed frames after a speech_started that no caller spoke (the harness's own cleanup flushes are excluded). Gate fired = the gateway's VAD rules let the burst start a turn (baseline: speech_started; fix: hold). Wilson 95 % intervals in brackets.

### By placement

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 | 9/12 = 75 % [47, 91] | 9/12 = 75 % [47, 91] | 9/12 = 75 % [47, 91] | 6/12 = 50 % [25, 75] |
| A2 | 6/12 = 50 % [25, 75] | 6/12 = 50 % [25, 75] | 3/12 = 25 % [9, 53] | 1/12 = 8 % [1, 35] |
| A3 | 9/12 = 75 % [47, 91] | 9/12 = 75 % [47, 91] | 11/12 = 92 % [65, 99] | 10/12 = 83 % [55, 95] |
| all | 24/36 = 67 % [50, 80] | 24/36 = 67 % [50, 80] | 23/36 = 64 % [48, 78] | 17/36 = 47 % [32, 63] |

### By placement and delay

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 316 ms | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 3/6 = 50 % [19, 81] |
| A1 506 ms | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 3/6 = 50 % [19, 81] |
| A2 316 ms | 2/6 = 33 % [10, 70] | 2/6 = 33 % [10, 70] | 2/6 = 33 % [10, 70] | 1/6 = 17 % [3, 56] |
| A2 506 ms | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 1/6 = 17 % [3, 56] | 0/6 = 0 % [0, 39] |
| A3 316 ms | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 6/6 = 100 % [61, 100] | 6/6 = 100 % [61, 100] |
| A3 506 ms | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 4/6 = 67 % [30, 90] |

### By placement and level (dB under the played audio)

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 -20 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] |
| A1 -35 dB | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] | 0/4 = 0 % [0, 49] |
| A1 -5 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 2/4 = 50 % [15, 85] |
| A2 -20 dB | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] | 0/4 = 0 % [0, 49] |
| A2 -35 dB | 2/4 = 50 % [15, 85] | 2/4 = 50 % [15, 85] | 1/4 = 25 % [5, 70] | 0/4 = 0 % [0, 49] |
| A2 -5 dB | 3/4 = 75 % [30, 95] | 3/4 = 75 % [30, 95] | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] |
| A3 -20 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 3/4 = 75 % [30, 95] |
| A3 -35 dB | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] | 3/4 = 75 % [30, 95] | 3/4 = 75 % [30, 95] |
| A3 -5 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] |

### By level

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| -20 dB | 9/12 = 75 % [47, 91] | 9/12 = 75 % [47, 91] | 9/12 = 75 % [47, 91] | 7/12 = 58 % [32, 81] |
| -35 dB | 4/12 = 33 % [14, 61] | 4/12 = 33 % [14, 61] | 5/12 = 42 % [19, 68] | 3/12 = 25 % [9, 53] |
| -5 dB | 11/12 = 92 % [65, 99] | 11/12 = 92 % [65, 99] | 9/12 = 75 % [47, 91] | 7/12 = 58 % [32, 81] |

### What the hold did with the bursts that fired the gate (fix)

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 0 | 2 | 1 | 3 |
| no_gate | 3 | 9 | 1 | 13 |
| released:caller | 7 | 1 | 10 | 18 |
| released:playback_ended | 2 | 0 | 0 | 2 |

### Released bursts (fix): why

| trial | kind | delay | level | reason | held ms | unmatched | ASR text | played text (end) | self-interruption |
|---|---|---|---|---|---|---|---|---|---|
| A00 | A3 | 316 | -35 | caller | 402 | 2 | '你好，注册是。' | 息。请您准备好注册时使用的手机号码、身份证后四位，以及最近一次充值的大概金额。' | yes |
| A01 | A3 | 506 | -5 | caller | 130 | 2 | '对于。' | '请您准备好注册时使用的手机号码、身份证后四位，以及最近一次充值的大概金额。' | yes |
| A03 | A3 | 316 | -35 | caller | 639 | 2 | '上门之前会出现。' | 之前会提前一个小时给您打电话，安装过程包括布线、调试路由器和测试网速三个步骤。' | yes |
| A04 | A3 | 506 | -20 | caller | 157 | 3 | '病情重的。' | 改约到后天或者选择放到附近的自提柜，取件码会通过短信发送到您预留的手机号码上。' | yes |
| A06 | A1 | 316 | -35 | playback_ended | 993 | 0 | '给您一个明确的答复。' | '我们会尽快核实情况，并在今天下班之前给您一个明确的答复。' | no |
| A07 | A3 | 316 | -5 | caller | 898 | 2 | '嗯，好。' | 改约到后天或者选择放到附近的自提柜，取件码会通过短信发送到您预留的手机号码上。' | yes |
| A12 | A3 | 316 | -20 | caller | 132 | 2 | '桥市路。' | 之前会提前一个小时给您打电话，安装过程包括布线、调试路由器和测试网速三个步骤。' | yes |
| A13 | A1 | 506 | -20 | caller | 151 | 2 | '领导。' | '好的，我已经为您记录下来了，稍后会有专人跟进，请您保持电话畅通，谢谢。' | yes |
| A16 | A2 | 316 | -5 | caller | 163 | 2 | '如果您想办理新号养码。' | 要满足几个条件：号码已经使用满一百二十天，账户没有欠费，也没有正在履行的合约。' | yes |
| A19 | A1 | 316 | -5 | playback_ended | 1255 | 0 | '电话畅通，谢谢。' | '好的，我已经为您记录下来了，稍后会有专人跟进，请您保持电话畅通，谢谢。' | no |
| A21 | A1 | 316 | -20 | caller | 372 | 4 | '是的，是的。' | '师傅会在明天上午九点到十点之间上门，请您提前准备好相关证件。' | yes |
| A23 | A3 | 316 | -5 | caller | 147 | 3 | '这里面。' | 室内墙体遮挡或者手机本身的设置。建议您先重启手机，再检查一下飞行模式是否关闭。' | yes |
| A27 | A1 | 506 | -20 | caller | 174 | 3 | '在会议中。' | '新的套餐从下个月一号开始生效，本月的费用不会受到任何影响。' | yes |
| A28 | A3 | 506 | -5 | caller | 121 | 3 | '我看一下。' | 您的耐心等待。经过系统查询，您的订单目前处于配送途中，预计明天下午五点前送达。' | yes |
| A29 | A3 | 506 | -35 | caller | 416 | 5 | '不能只单指。' | '请您准备好注册时使用的手机号码、身份证后四位，以及最近一次充值的大概金额。' | yes |
| A30 | A1 | 506 | -5 | caller | 651 | 2 | '提前准备好香烟。' | '师傅会在明天上午九点到十点之间上门，请您提前准备好相关证件。' | no |
| A31 | A1 | 316 | -20 | caller | 168 | 2 | '嗯嗯。' | '我们会尽快核实情况，并在今天下班之前给您一个明确的答复。' | yes |
| A32 | A3 | 316 | -20 | caller | 129 | 2 | '准备，准备好。' | 息。请您准备好注册时使用的手机号码、身份证后四位，以及最近一次充值的大概金额。' | yes |
| A33 | A1 | 316 | -5 | caller | 380 | 2 | '首先。' | '我正在为您查询账户信息，大概需要十几秒钟的时间，感谢您的耐心。' | yes |
| A35 | A1 | 506 | -5 | caller | 363 | 2 | 'कमी करने' | '我们会尽快核实情况，并在今天下班之前给您一个明确的答复。' | yes |

### How the ASR hears a held burst (fix): first check and last check before the decision

| level | held bursts | median share of ASR units found in the played text, first check | same, last check | audio of the first check, ms (P50 / P95) |
|---|---|---|---|---|
| -35 dB | 5 | 0.71 | 0.71 | 720 / 727 (n=5) |
| -20 dB | 9 | 0.25 | 0.25 | 734 / 944 (n=9) |
| -5 dB | 9 | 0.25 | 0.00 | 736 / 1599 (n=9) |

Audio covered by the check that released a burst (false release): 984 / 1819 (n=20) ms.

False releases by the number of checks run before the decision (1 = decided on the very first check): {1: 10, 2: 5, 3: 2, 4: 1}.

### Baseline: what the burst turns became

Baseline bursts with a gate: 24; speech_started at the client while AICC had audio: 24; barge-in outcomes of those turns: {'real': 7, 'echo': 7, None: 10}; mean audio cut by a self-interruption: 5159 ms.

Realised burst level against the played audio (rms over the burst, dB): P50 -19.2, min -36.1, max -0.5.

A3 bursts whose source ended less than 3 s before the end of the answer (not mid-answer; answers whose audio was cut before the end count as mid): 0.

## Am. Residual-echo bursts, mild (cleaner, nearly linear) leak model (no steady echo)

Bursts run: 36 (baseline A1 6, baseline A2 6, baseline A3 6, fix A1 6, fix A2 6, fix A3 6); trials with an error: 0.

Canceller bypass at the start of each burst's window: {('baseline', True): 15, ('baseline', False): 3, ('fix', True): 17, ('fix', False): 1}.

Self-interruption = AICC flushed unplayed frames after a speech_started that no caller spoke (the harness's own cleanup flushes are excluded). Gate fired = the gateway's VAD rules let the burst start a turn (baseline: speech_started; fix: hold). Wilson 95 % intervals in brackets.

### By placement

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 2/6 = 33 % [10, 70] |
| A2 | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 4/6 = 67 % [30, 90] | 1/6 = 17 % [3, 56] |
| A3 | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 5/6 = 83 % [44, 97] | 3/6 = 50 % [19, 81] |
| all | 14/18 = 78 % [55, 91] | 14/18 = 78 % [55, 91] | 14/18 = 78 % [55, 91] | 6/18 = 33 % [16, 56] |

### By placement and delay

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 316 ms | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] |
| A1 506 ms | 3/3 = 100 % [44, 100] | 3/3 = 100 % [44, 100] | 3/3 = 100 % [44, 100] | 1/3 = 33 % [6, 79] |
| A2 316 ms | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 0/3 = 0 % [0, 56] |
| A2 506 ms | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] |
| A3 316 ms | 3/3 = 100 % [44, 100] | 3/3 = 100 % [44, 100] | 3/3 = 100 % [44, 100] | 2/3 = 67 % [21, 94] |
| A3 506 ms | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] |

### By placement and level (dB under the played audio)

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| A1 -20 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] |
| A1 -35 dB | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] | 0/2 = 0 % [0, 66] |
| A1 -5 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 0/2 = 0 % [0, 66] |
| A2 -20 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] |
| A2 -35 dB | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] |
| A2 -5 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 0/2 = 0 % [0, 66] |
| A3 -20 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] |
| A3 -35 dB | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] |
| A3 -5 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] |

### By level

| cell | base: gate fired | base: self-interruption | fix: gate fired | fix: self-interruption |
|---|---|---|---|---|
| -20 dB | 6/6 = 100 % [61, 100] | 6/6 = 100 % [61, 100] | 6/6 = 100 % [61, 100] | 4/6 = 67 % [30, 90] |
| -35 dB | 2/6 = 33 % [10, 70] | 2/6 = 33 % [10, 70] | 2/6 = 33 % [10, 70] | 1/6 = 17 % [3, 56] |
| -5 dB | 6/6 = 100 % [61, 100] | 6/6 = 100 % [61, 100] | 6/6 = 100 % [61, 100] | 1/6 = 17 % [3, 56] |

### What the hold did with the bursts that fired the gate (fix)

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 0 | 3 | 2 | 5 |
| no_gate | 1 | 2 | 1 | 4 |
| released:caller | 2 | 1 | 3 | 6 |
| released:playback_ended | 3 | 0 | 0 | 3 |

### Released bursts (fix): why

| trial | kind | delay | level | reason | held ms | unmatched | ASR text | played text (end) | self-interruption |
|---|---|---|---|---|---|---|---|---|---|
| Am00 | A1 | 316 | -5 | playback_ended | 1317 | 0 | '给您一个明确的答复。' | '我们会尽快核实情况，并在今天下班之前给您一个明确的答复。' | no |
| Am04 | A3 | 316 | -35 | caller | 161 | 2 | '典型的合约。' | 带上身份证到新的营业厅办理就可以，全程大约十分钟，原来的号码和余额都不会改变。' | yes |
| Am05 | A3 | 316 | -20 | caller | 1191 | 3 | '背诵途中，遇。' | 您的耐心等待。经过系统查询，您的订单目前处于配送途中，预计明天下午五点前送达。' | yes |
| Am07 | A2 | 506 | -20 | caller | 150 | 4 | '影响办理协后转。' | 要满足几个条件：号码已经使用满一百二十天，账户没有欠费，也没有正在履行的合约。' | yes |
| Am08 | A3 | 506 | -5 | caller | 152 | 2 | '现在最。' | 卡可以添加最多四张副卡，所有成员共用流量和通话时长，每个月只需要一百五十九元。' | yes |
| Am12 | A1 | 506 | -20 | caller | 1190 | 3 | '本月的费用，黄教授。' | '已经为您办理完成，新的套餐从下个月一号开始生效，本月的费用不会受到任何影响。' | yes |
| Am13 | A1 | 316 | -20 | caller | 145 | 2 | '首先。' | '我正在为您查询账户信息，大概需要十几秒钟的时间，感谢您的耐心。' | yes |
| Am14 | A1 | 506 | -5 | playback_ended | 1637 | 0 | '保持电话畅通，谢。' | '好的，我已经为您记录下来了，稍后会有专人跟进，请您保持电话畅通，谢谢。' | no |
| Am16 | A1 | 506 | -35 | playback_ended | 974 | 0 | '不会受到任何影响。' | '新的套餐从下个月一号开始生效，本月的费用不会受到任何影响。' | no |

### How the ASR hears a held burst (fix): first check and last check before the decision

| level | held bursts | median share of ASR units found in the played text, first check | same, last check | audio of the first check, ms (P50 / P95) |
|---|---|---|---|---|
| -35 dB | 2 | 0.80 | 0.80 | 725 / 730 (n=2) |
| -20 dB | 6 | 0.67 | 0.53 | 735 / 1640 (n=6) |
| -5 dB | 6 | 1.00 | 1.00 | 735 / 2206 (n=6) |

Audio covered by the check that released a burst (false release): 1670 / 1916 (n=9) ms.

False releases by the number of checks run before the decision (1 = decided on the very first check): {1: 4, 5: 2}.

### Baseline: what the burst turns became

Baseline bursts with a gate: 14; speech_started at the client while AICC had audio: 14; barge-in outcomes of those turns: {None: 5, 'echo': 7, 'real': 2}; mean audio cut by a self-interruption: 5283 ms.

Realised burst level against the played audio (rms over the burst, dB): P50 -20.6, min -36.0, max -3.5.

A3 bursts whose source ended less than 3 s before the end of the answer (not mid-answer; answers whose audio was cut before the end count as mid): 0.

## B. Real barge-ins over playback, line without echo

Added delay of the hold itself (fix, `speech_started.held_ms`, hold -> release): 155 / 348 (n=30). The onset -> speech_started columns also contain the VAD's own delay, which varies from run to run (the bot's audio under the caller differs), so compare the held_ms.

| condition | barge-ins | speech_started at the client | onset -> speech_started ms (P50 / P95) | onset -> AICC flush ms (P50 / P95) | speech_stopped -> transcript ms (P50 / P95) | turns without speech_started |
|---|---|---|---|---|---|---|
| baseline | 30 | 30/30 = 100 % [89, 100] | 490 / 1249 (n=30) | 490 / 1249 (n=30) | 229 / 305 (n=30) | 0 |
| fix | 30 | 30/30 = 100 % [89, 100] | 648 / 1565 (n=30) | 648 / 1565 (n=30) | 227 / 306 (n=30) | 0 |

Audio covered by the check that released a real caller (ms, P50 / P95): 730 / 972 (n=30); ASR checks per held turn: 1 / 2 (n=30).

VAD gate delay (onset in the injected audio -> the gate fired in the capture, input ms; P50 / P95): baseline 501 / 1260 (n=30), fix 499 / 1415 (n=30). Same code in both; it varies with the bot audio under the caller.

Paired by trial (same clip and offset), fix minus baseline speech_started latency: P50 +150 ms, P95 +342 ms, mean +176 ms (n=30).

| set | condition | n | speech_started | onset -> speech_started ms | onset -> flush ms |
|---|---|---|---|---|---|
| zh | baseline | 10 | 10 | 487 / 1359 (n=10) | 488 / 1359 (n=10) |
| zh | fix | 10 | 10 | 636 / 1525 (n=10) | 636 / 1525 (n=10) |
| en | baseline | 10 | 10 | 498 / 1110 (n=10) | 498 / 1111 (n=10) |
| en | fix | 10 | 10 | 692 / 1437 (n=10) | 692 / 1437 (n=10) |
| mix | baseline | 10 | 10 | 511 / 1031 (n=10) | 511 / 1031 (n=10) |
| mix | fix | 10 | 10 | 665 / 1193 (n=10) | 666 / 1193 (n=10) |

Fix outcomes (how each turn was released or dropped):

| outcome | count |
|---|---|
| released:caller | 30 |

Baseline: {'speech_started': 30}.

Committed-turn transcripts (baseline | fix), same clip:

Identical transcripts when both conditions committed the turn: 23/30. Differences:

| trial | clip | baseline | fix | fix outcome |
|---|---|---|---|---|
| B02 | en_004 | 'The find also grabs insight into the evolution of feathers and birds.' | 'The find also grants insight into the evolution of feathers and birds.' | released:caller |
| B05 | mix_034 | '对啊，大家最近都是觉得AI有一天能够replace人类。' | '对啊，大家最近都是觉得A I有一天能够replace人类。' | released:caller |
| B08 | mix_042 | '也就是我现在是个 adviser 呢，就跟我说，哎，那你过来跟我做，我就给你一些 robotics 相关的题目啊，这样子做，这样，那我就想说，哎，那其实。' | '也就是我现在是个 adviser 呢，就跟我说，哎，那你过来跟我说，我就给你一些 robotics 相关的题目啊，这样子做，这样，那我就想说，哎，那其实。' | released:caller |
| B10 | mix_041 | '或者是其他东西分给我们吃，然后我们就边吃边听他分享，share他的一些他们公司的一些东西。' | '或者是其他的东西分给我们吃，然后我们就边吃边听他分享，share他的一些他们公司的一些东西。' | released:caller |
| B17 | mix_038 | '就是会有很多呃，比如说跟Luminite聚会啊，一起吃饭。' | '就是会有很多呃，比如说跟Luminize聚会啊，一起吃饭。' | released:caller |
| B20 | zh_036 | '你手上的水分层会与外层发生反应，这会形成一层类似外壳的东西，感觉十分有趣。' | '你手上的水分成会与外层发生反应，这会形成一层类似外壳的东西，感觉十分有趣。' | released:caller |
| B23 | zh_027 | '你也可以咨询除贵国以外其他政府的建议，但他们的建议是为他们的公民而制定的。' | '你也可以咨询除本国以外其他政府的建议，但他们的建议是为他们的公民而制定的。' | released:caller |

Real turns dropped in the fix (no speech_started at the client): 0. 

## Be. Real barge-ins over playback, line with steady echo (ERL 10 dB, 100 / 180 ms)

Added delay of the hold itself (fix, `speech_started.held_ms`, hold -> release): 152 / 270 (n=10). The onset -> speech_started columns also contain the VAD's own delay, which varies from run to run (the bot's audio under the caller differs), so compare the held_ms.

| condition | barge-ins | speech_started at the client | onset -> speech_started ms (P50 / P95) | onset -> AICC flush ms (P50 / P95) | speech_stopped -> transcript ms (P50 / P95) | turns without speech_started |
|---|---|---|---|---|---|---|
| baseline | 10 | 10/10 = 100 % [72, 100] | 559 / 1807 (n=10) | 560 / 1807 (n=10) | 217 / 293 (n=10) | 0 |
| fix | 10 | 10/10 = 100 % [72, 100] | 712 / 2225 (n=10) | 712 / 2225 (n=10) | 219 / 276 (n=10) | 0 |

Audio covered by the check that released a real caller (ms, P50 / P95): 726 / 869 (n=10); ASR checks per held turn: 1 / 2 (n=10).

VAD gate delay (onset in the injected audio -> the gate fired in the capture, input ms; P50 / P95): baseline 570 / 1820 (n=10), fix 570 / 1971 (n=10). Same code in both; it varies with the bot audio under the caller.

Paired by trial (same clip and offset), fix minus baseline speech_started latency: P50 +122 ms, P95 +1061 ms, mean +213 ms (n=10).

| set | condition | n | speech_started | onset -> speech_started ms | onset -> flush ms |
|---|---|---|---|---|---|
| zh | baseline | 3 | 3 | 542 / 938 (n=3) | 542 / 938 (n=3) |
| zh | fix | 3 | 3 | 740 / 1064 (n=3) | 740 / 1064 (n=3) |
| en | baseline | 4 | 4 | 611 / 2208 (n=4) | 611 / 2208 (n=4) |
| en | fix | 4 | 4 | 1351 / 2367 (n=4) | 1351 / 2367 (n=4) |
| mix | baseline | 3 | 3 | 549 / 794 (n=3) | 549 / 794 (n=3) |
| mix | fix | 3 | 3 | 609 / 670 (n=3) | 609 / 670 (n=3) |

Fix outcomes (how each turn was released or dropped):

| outcome | count |
|---|---|
| released:caller | 10 |

Baseline: {'speech_started': 10}.

Committed-turn transcripts (baseline | fix), same clip:

Identical transcripts when both conditions committed the turn: 5/10. Differences:

| trial | clip | baseline | fix | fix outcome |
|---|---|---|---|---|
| Be00 | mix_010 | '有时候给一些 data set，然后你。' | '比如说，给一些data set，然后你。' | released:caller |
| Be01 | zh_005 | '卷心菜汁的颜色将根据化学物质的酸碱性发生变化。' | '青菜汁的颜色将根据化学物质的酸碱性发生变化。' | released:caller |
| Be02 | en_031 | 'Most riders wearing riding boots with a heel and a smooth, quite narrow sole.' | 'First, most riders wearing riding boots with a heel and a smooth, quite narrow sole.' | released:caller |
| Be04 | en_040 | 'In the north, the region is bounded by the Sahel, and in the south and west by the Atlantic Ocean.' | 'It is bounded by the Sahel, and in the south and west by the Atlantic Ocean.' | released:caller |
| Be08 | zh_026 | '不是告别，这是一个篇章的结束，也是新篇章的开始。' | '这并不是告别，这是一个篇章的结束，也是新篇章的开始。' | released:caller |

Real turns dropped in the fix (no speech_started at the client): 0. 

## E0. Ordinary turns over silence

Added delay of the hold itself (fix, `speech_started.held_ms`, hold -> release): n/a. The onset -> speech_started columns also contain the VAD's own delay, which varies from run to run (the bot's audio under the caller differs), so compare the held_ms.

| condition | barge-ins | speech_started at the client | onset -> speech_started ms (P50 / P95) | onset -> AICC flush ms (P50 / P95) | speech_stopped -> transcript ms (P50 / P95) | turns without speech_started |
|---|---|---|---|---|---|---|
| baseline | 8 | 8/8 = 100 % [68, 100] | 194 / 262 (n=8) | n/a | 216 / 264 (n=8) | 0 |
| fix | 8 | 8/8 = 100 % [68, 100] | 223 / 277 (n=8) | n/a | 225 / 262 (n=8) | 0 |

Audio covered by the check that released a real caller (ms, P50 / P95): n/a; ASR checks per held turn: n/a.

VAD gate delay (onset in the injected audio -> the gate fired in the capture, input ms; P50 / P95): baseline 208 / 276 (n=8), fix 237 / 291 (n=8). Same code in both; it varies with the bot audio under the caller.

Paired by trial (same clip and offset), fix minus baseline speech_started latency: P50 +20 ms, P95 +42 ms, mean +14 ms (n=8).

| set | condition | n | speech_started | onset -> speech_started ms | onset -> flush ms |
|---|---|---|---|---|---|
| zh | baseline | 3 | 3 | 202 / 205 (n=3) | n/a |
| zh | fix | 3 | 3 | 221 / 224 (n=3) | n/a |
| en | baseline | 2 | 2 | 205 / 222 (n=2) | n/a |
| en | fix | 2 | 2 | 181 / 186 (n=2) | n/a |
| mix | baseline | 3 | 3 | 184 / 273 (n=3) | n/a |
| mix | fix | 3 | 3 | 226 / 296 (n=3) | n/a |

Fix outcomes (how each turn was released or dropped):

| outcome | count |
|---|---|
| gate_not_held | 8 |

Baseline: {'speech_started': 8}.

Committed-turn transcripts (baseline | fix), same clip:

Identical transcripts when both conditions committed the turn: 5/8. Differences:

| trial | clip | baseline | fix | fix outcome |
|---|---|---|---|---|
| E000 | mix_023 | '那图像识别，其实上学上个semester我们。' | '那图像识别其实上学上个semester我们。' | gate_not_held |
| E003 | mix_035 | '对，我的我undergraduate的时候学的major也是。' | '对我的我undergraduate的时候学的major也是。' | gate_not_held |
| E007 | mix_005 | '也是那个hardware。' | '也是那个哈维。' | gate_not_held |

Real turns dropped in the fix (no speech_started at the client): 0. 

## C. Short caller utterances over playback

Utterances synthesised with the gateway's Qwen3-TTS: Chinese in `uncle_fu`, English in `aiden` (the bot speaks `vivian`), trimmed, scaled to -26 dBov active level, played from a clean 16 kHz clip through the simulated line.

| utterance | base: barge-ins | base: onset -> speech_started ms | fix: barge-ins | fix: onset -> speech_started ms | fix outcomes |
|---|---|---|---|---|---|
| 喂 | 2/2 | 468 / 486 (n=2) | 2/2 | 3596 / 3607 (n=2) | released:turn_end 2 |
| 嗯 | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| 对 | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| 好的 | 2/2 | 445 / 447 (n=2) | 2/2 | 3806 / 3807 (n=2) | released:turn_end 2 |
| 等一下 | 2/2 | 448 / 448 (n=2) | 2/2 | 592 / 624 (n=2) | released:caller 2 |
| 不是 | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| 可以 | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| OK | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| wait | 0/2 | n/a | 0/2 | n/a | no_gate 2 |
| hello | 2/2 | 475 / 481 (n=2) | 2/2 | 4105 / 4122 (n=2) | released:turn_end 2 |
| hold on | 0/2 | n/a | 0/2 | n/a | no_gate 2 |

Totals by fix outcome (latency onset -> speech_started at the client, ms P50 / P95):

| outcome | count | latency |
|---|---|---|
| no_gate | 14 | n/a |
| released:caller | 2 | 592 / 624 (n=2) |
| released:turn_end | 6 | 3806 / 4114 (n=6) |

Baseline: 8/22 barge-ins, latency 448 / 486 (n=8).

## D. Non-speech noise over playback

| sound | condition | n | gate fired | barge-in (speech_started) | fix outcomes |
|---|---|---|---|---|---|
| cup | baseline | 5 | 0 | 0 | - |
| cup | fix | 5 | 0 | 0 | no_gate 5 |
| keys | baseline | 5 | 0 | 0 | - |
| keys | fix | 5 | 0 | 0 | no_gate 5 |
| clicks | baseline | 5 | 0 | 0 | - |
| clicks | fix | 5 | 0 | 0 | no_gate 5 |

Noise is synthesised (damped resonances, key clicks with a low thump, a 8 Hz click train) at a peak of -3, -12 or -24 dBFS; hold_lib.synth_noise.

## E. Cost of the checks

Held turns in the fix runs: 85; ASR checks per held turn: mean 3.1, P50 1.0, P95 14.0, max 15; failed checks: 0.

Check latency (request -> result), estimated as the capture time of the result less the end of the audio the check covered (non-final checks only; ms, P50 / P95): 154 / 187 (n=251).

Checks per held turn by scenario: A: 53/23 = 2.3, Am: 82/14 = 5.9, B: 33/30 = 1.1, Be: 11/10 = 1.1, C: 86/8 = 10.8.

### Commit -> transcript latency (speech_stopped at the client -> transcription.completed, ms P50 / P95)

| scenario | baseline | fix |
|---|---|---|
| E0 | 216 / 264 (n=8) | 225 / 262 (n=8) |
| B | 229 / 305 (n=30) | 227 / 306 (n=30) |
| Be | 217 / 293 (n=10) | 219 / 276 (n=10) |
| C | 123 / 149 (n=8) | 0 / 102 (n=8) |

E0 are ordinary turns over silence (no hold possible): any difference there is run-to-run variation. B, Be and C turns were held, and the fix reuses the final check's transcript, so their commit -> transcript time can be shorter.

## Offline what-ifs: confirm_min_new_units = 1 and 3

Replayed from the fix runs' `hold_check` events (each check's ASR text and the assistant text played around it; units and the longest in-order match are the gateway's own `text.similarity`). Limits: the recorded sequence of a turn stops at the check where N=2 decided. N=1 can only decide earlier or at the same check, so its decision and latency are exact. N=3 agrees with N=2 only if that check already had at least 3 unmatched units; otherwise the later checks, which would have shown whether 3 is reached, do not exist (censored: N=3 releases at or after the recorded time, or never). Latencies are capture time of the deciding check less the onset in the injected audio; they exclude the delivery of the event.

### A (bursts): false releases

| N | held burst turns | released as caller | censored (N=3 unknown) | false-release rate per burst |
|---|---|---|---|---|
| 1 | 23 | 21 | 0 | 21/36 bursts = 58 % |
| 2 | 23 | 18 | 0 | 18/36 bursts = 50 % |
| 3 | 23 | 6 | 12 | 6/36 bursts = 17 % (upper bound 50 %) |

### Am (bursts): false releases

| N | held burst turns | released as caller | censored (N=3 unknown) | false-release rate per burst |
|---|---|---|---|---|
| 1 | 14 | 11 | 0 | 11/18 bursts = 61 % |
| 2 | 14 | 6 | 0 | 6/18 bursts = 33 % |
| 3 | 14 | 3 | 3 | 3/18 bursts = 17 % (upper bound 33 %) |

### B (real barge-ins, no echo): release by the checks

| N | released early (partial check) | released at the final check | dropped / other | censored | release latency from onset, ms (P50 / P95) |
|---|---|---|---|---|---|
| 1 | 30 | 0 | 0 | 0 | 646 / 1578 (n=30) |
| 2 | 30 | 0 | 0 | 0 | 662 / 1578 (n=30) |
| 3 | 15 | 0 | 0 | 15 | 740 / 1734 (n=15) |

### Be (real barge-ins, steady echo): release by the checks

| N | released early (partial check) | released at the final check | dropped / other | censored | release latency from onset, ms (P50 / P95) |
|---|---|---|---|---|---|
| 1 | 10 | 0 | 0 | 0 | 727 / 2105 (n=10) |
| 2 | 10 | 0 | 0 | 0 | 727 / 2238 (n=10) |
| 3 | 4 | 0 | 0 | 6 | 933 / 1848 (n=4) |

### C (short utterances): release by the checks

| N | released early (partial check) | released at the final check | dropped / other | censored | release latency from onset, ms (P50 / P95) |
|---|---|---|---|---|---|
| 1 | 8 | 0 | 0 | 0 | 624 / 656 (n=8) |
| 2 | 2 | 6 | 0 | 0 | 3720 / 4124 (n=8) |
| 3 | 0 | 6 | 0 | 2 | 3819 / 4127 (n=6) |

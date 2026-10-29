# Hold measurement, v3 (issue #5)

Same harness, seeds, burst models and trials as `hold.md` (v1). Baseline rows are the v1 run's (the same code path, `confirm_while_playing: false`), re-using them to stay inside the time budget; v1 fix rows are kept for comparison; v3 (v2 + correlation veto, all defaults) was run on the same trials (A 36, Am 18, B 30, Be 10, C 22 bursts or turns), plus F (24 double-talk trials per condition, baseline and v3). D was skipped. Alternation of conditions chunk by chunk applied to F only. Gate-fired counts differ between runs (A: baseline 24/36, v3 18/36) because the sessions differ (TTS output, canceller state); compare self-interruptions, not gate counts.

The run used `confirm_echo_psr` 10 and `confirm_echo_drop_psr` 15. After it the gateway's default `confirm_echo_psr` became 15, from the replay under Thresholds (raising the threshold replays exactly: it only releases earlier, at a check that is in the recorded sequence).

An incident: trial F10 of the first F job took 523 s of wall clock. The machine went into idle sleep at 18:57:20 (`pmset -g log`: "Entering Sleep state due to 'Idle Sleep'", maintenance sleep, wake at 19:06:32, 505 s) in the middle of that trial; the asyncio clock (monotonic, stopped during sleep) shows F10 as 22 s, as its neighbours, and the capture is normal. It is not the gateway, the backends or the harness. For the rest of the run I wrapped the run in `caffeinate -i -w` (a process assertion, no system setting changed).

## Findings

1. **Self-interruptions per burst, baseline -> v1 -> v3**: default model 24/36 (67 %) -> 17/36 (47 %) -> 3/36 (8 %); mild model 14/18 (78 %) -> 6/18 (33 %) -> 2/18 (11 %). A3 (mid-answer): 9/12 -> 10/12 -> 2/12. Most of the gain comes from the v2 rules (10 default-model bursts dropped as echo, against 3 in v1). The veto itself decided little in these runs: 7 of the 8 default-model releases happened while the canceller was not in bypass (`psr_why: no_bypass`, no correlation available), and one Am burst was dropped by `echo_signal`.
2. **Real barge-ins**: 30/30 (B) and 10/10 (Be) released, none dropped. Paired added latency (onset -> speech_started) v3 minus baseline P50 +162 ms / P95 +422 ms (B), +148 / +956 ms (Be); v1 was +150 / +342 and +122 / +1061. `held_ms` P50 153 / P95 406 (B).
3. **Short utterances**: of the 8 that reach the gate, "喂", "好的", "hello" now release at the pause (P50 1.17-1.58 s after onset, was 3.6-4.1 s in v1; baseline ~0.45 s); "等一下" at 0.6 s. None dropped. The 14 that never reach the gate (under 400 ms of speech) are unchanged.
4. **F, double talk with a leak (24 trials)**: all 24 callers released (none dropped), but late when the leak is loud: onset -> speech_started P50 / P95 for F1 -5 dB 1913 / 2939 ms, F2 -5 dB 1315 / 1618 ms, F1 -20 dB 860 / 1180 ms, F2 -20 dB 714 / 1227 ms, against 360-520 ms (P50) in the baseline. The veto holds a caller while the PSR of [onset, check end] is at least 10 (max PSR per trial up to 62.8 at -5 dB); the caller is released once the growing window dilutes it. Hold time of released callers P50 378 / P95 1403 ms.
5. **Cost**: checks per held turn: B 1.3, Be 1.1, C 3.4 (was 10.8), F 2.8, A 6.3 and Am 6.2 (more than v1's 2.3: bursts are held longer, so they are checked longer). Partial-check latency P50 151 / P95 186 ms. speech_stopped -> transcript unchanged (B 227 / 305 ms).

## Thresholds (from the what-if table below)

- `confirm_echo_psr`: the replay shows no loss on A/Am for 10, 12, 15 (8 / 36 and 3 / 18 releases by a check at each), and the delay of F callers falls with the threshold (F P50 release time 1157 ms at 10, 812 at 12, 656 at 15; B 689 -> 681). 8 gains one A release (7/36) but would hold 9 of 30 B callers and 11 of 24 F callers beyond the recorded sequence (censored), later by an unknown amount. I would take 15 rather than 10, with the caveat that the evidence for any psr threshold is thin: only one default-model release (A07, psr 8.07) and one Am drop depend on it.
- `confirm_echo_drop_psr`: no real caller was dropped at any value in {10, 15, 20, 25, null} (B, Be, F, C: 0 drops), and 10 or 15 drops one more Am burst than 20 or higher. Keep 15.
- The larger issue is not the threshold: with the canceller out of bypass the veto has no signal, and that is where the default-model releases still happen.


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
 sleep                1 (sleep prevented by caffeinate, powerd)
 hibernatemode        3
 ttyskeepawake        1
 displaysleep         10
 tcpkeepalive         1
 powermode            0
 womp                 1

```

- **git**: HEAD 7920a38 plus the uncommitted fix (11 changed paths under src/ and config.yaml)

- **config**: {"barge_in (fix)": {"echo_guard_ms": 2000, "min_speech_ms_while_playing": 400, "echo_turn_suppression": true, "echo_match_threshold": 0.7, "echo_match_min_units": 4, "echo_match_window_ms": 3000, "confirm_while_playing": true, "confirm_min_new_units": 2, "confirm_step_ms": 250, "confirm_window_ms": 4000, "confirm_timeout_ms": 2000, "confirm_echo_psr": 10, "confirm_echo_drop_psr": 15}, "aec": {"enabled": true, "engine": "webrtc_aec3", "asr_input": "speex"}, "output.max_lead_ms": 10000}

- **seed**: 20261029

- **burst_model**: {"profiles": {"default": {"band_hz": [300.0, 2400.0], "clip_rel": 0.7, "gain_sigma_db": 5.0, "dropout_p": 0.1, "dropout_db": -25.0}, "mild": {"band_hz": [200.0, 3400.0], "clip_rel": 1.5, "gain_sigma_db": 2.5, "dropout_p": 0.03, "dropout_db": -15.0}}, "gain_rho": 0.5, "ir_taps": 48, "delays_ms": [316, 506], "levels_db": [-5, -20, -35], "length_s": "1.5-2.5"}

- **started**: 2026-10-29 18:50:13 +0800

## A. Bursts, default leak model

Self-interruption = AICC flushed unplayed frames after a speech_started with no caller. Baseline and v1 are the v1 run; v3 is the current fix. [Wilson 95 %]

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 | 9/12 = 75 % [47, 91] | 6/12 = 50 % [25, 75] | 0/12 = 0 % [0, 24] |
| A2 | 6/12 = 50 % [25, 75] | 1/12 = 8 % [1, 35] | 1/12 = 8 % [1, 35] |
| A3 | 9/12 = 75 % [47, 91] | 10/12 = 83 % [55, 95] | 2/12 = 17 % [5, 45] |
| all | 24/36 = 67 % [50, 80] | 17/36 = 47 % [32, 63] | 3/36 = 8 % [3, 22] |

By placement and level:

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 -20 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 0/4 = 0 % [0, 49] |
| A1 -35 dB | 1/4 = 25 % [5, 70] | 0/4 = 0 % [0, 49] | 0/4 = 0 % [0, 49] |
| A1 -5 dB | 4/4 = 100 % [51, 100] | 2/4 = 50 % [15, 85] | 0/4 = 0 % [0, 49] |
| A2 -20 dB | 1/4 = 25 % [5, 70] | 0/4 = 0 % [0, 49] | 0/4 = 0 % [0, 49] |
| A2 -35 dB | 2/4 = 50 % [15, 85] | 0/4 = 0 % [0, 49] | 0/4 = 0 % [0, 49] |
| A2 -5 dB | 3/4 = 75 % [30, 95] | 1/4 = 25 % [5, 70] | 1/4 = 25 % [5, 70] |
| A3 -20 dB | 4/4 = 100 % [51, 100] | 3/4 = 75 % [30, 95] | 0/4 = 0 % [0, 49] |
| A3 -35 dB | 1/4 = 25 % [5, 70] | 3/4 = 75 % [30, 95] | 0/4 = 0 % [0, 49] |
| A3 -5 dB | 4/4 = 100 % [51, 100] | 4/4 = 100 % [51, 100] | 2/4 = 50 % [15, 85] |

By placement and delay:

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 316 ms | 5/6 = 83 % [44, 97] | 3/6 = 50 % [19, 81] | 0/6 = 0 % [0, 39] |
| A1 506 ms | 4/6 = 67 % [30, 90] | 3/6 = 50 % [19, 81] | 0/6 = 0 % [0, 39] |
| A2 316 ms | 2/6 = 33 % [10, 70] | 1/6 = 17 % [3, 56] | 1/6 = 17 % [3, 56] |
| A2 506 ms | 4/6 = 67 % [30, 90] | 0/6 = 0 % [0, 39] | 0/6 = 0 % [0, 39] |
| A3 316 ms | 4/6 = 67 % [30, 90] | 6/6 = 100 % [61, 100] | 1/6 = 17 % [3, 56] |
| A3 506 ms | 5/6 = 83 % [44, 97] | 4/6 = 67 % [30, 90] | 1/6 = 17 % [3, 56] |

Gate fired (the bursts that could have interrupted): baseline 24/36 = 67 % [50, 80], v1 23/36 = 64 % [48, 78], v3 18/36 = 50 % [34, 66]

v1: what the hold did

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 0 | 2 | 1 | 3 |
| no_gate | 3 | 9 | 1 | 13 |
| released:caller | 7 | 1 | 10 | 18 |
| released:playback_ended | 2 | 0 | 0 | 2 |

v3: what the hold did

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 2 | 2 | 6 | 10 |
| no_gate | 5 | 9 | 4 | 18 |
| released:caller | 5 | 1 | 1 | 7 |
| released:pause | 0 | 0 | 1 | 1 |

v3 releases other than at playback end:

| trial | kind | delay | level | reason | held ms | ASR text | unmatched | psr | psr_why | self-interruption |
|---|---|---|---|---|---|---|---|---|---|---|
| A07 | A3 | 316 | -5 | pause | 728 | '不是。' | 1 | 8.07 | None | yes |
| A16 | A2 | 316 | -5 | caller | 667 | '如果您想办理福泰电网，' | 3 | None | no_bypass | yes |
| A21 | A1 | 316 | -20 | caller | 921 | '办好相关证件。' | 1 | None | no_bypass | no |
| A27 | A1 | 506 | -20 | caller | 1735 | '业务的费用不会是没有任何。' | 5 | None | no_bypass | no |
| A28 | A3 | 506 | -5 | caller | 1734 | '配送途中，一名中。' | 3 | None | no_bypass | yes |
| A30 | A1 | 506 | -5 | caller | 1474 | '跟准备好相关证件。' | 1 | None | no_bypass | no |
| A33 | A1 | 316 | -5 | caller | 1176 | '感谢您的爱心。' | 1 | None | no_bypass | no |
| A35 | A1 | 506 | -5 | caller | 1702 | '咱之前给您一个明确的答案。' | 2 | None | no_bypass | no |

v3 release reasons: {'pause': 1, 'caller': 7}; drops: {'echo': 10}.

psr_why over all v3 checks: {None: 76, 'no_bypass': 37}.

## Am. Bursts, mild leak model

Self-interruption = AICC flushed unplayed frames after a speech_started with no caller. Baseline and v1 are the v1 run; v3 is the current fix. [Wilson 95 %]

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 | 5/6 = 83 % [44, 97] | 2/6 = 33 % [10, 70] | 0/6 = 0 % [0, 39] |
| A2 | 4/6 = 67 % [30, 90] | 1/6 = 17 % [3, 56] | 1/6 = 17 % [3, 56] |
| A3 | 5/6 = 83 % [44, 97] | 3/6 = 50 % [19, 81] | 1/6 = 17 % [3, 56] |
| all | 14/18 = 78 % [55, 91] | 6/18 = 33 % [16, 56] | 2/18 = 11 % [3, 33] |

By placement and level:

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 -20 dB | 2/2 = 100 % [34, 100] | 2/2 = 100 % [34, 100] | 0/2 = 0 % [0, 66] |
| A1 -35 dB | 1/2 = 50 % [9, 91] | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] |
| A1 -5 dB | 2/2 = 100 % [34, 100] | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] |
| A2 -20 dB | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] | 0/2 = 0 % [0, 66] |
| A2 -35 dB | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] | 1/2 = 50 % [9, 91] |
| A2 -5 dB | 2/2 = 100 % [34, 100] | 0/2 = 0 % [0, 66] | 0/2 = 0 % [0, 66] |
| A3 -20 dB | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] | 0/2 = 0 % [0, 66] |
| A3 -35 dB | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] | 0/2 = 0 % [0, 66] |
| A3 -5 dB | 2/2 = 100 % [34, 100] | 1/2 = 50 % [9, 91] | 1/2 = 50 % [9, 91] |

By placement and delay:

| cell | baseline | v1 | v3 |
|---|---|---|---|
| A1 316 ms | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] | 0/3 = 0 % [0, 56] |
| A1 506 ms | 3/3 = 100 % [44, 100] | 1/3 = 33 % [6, 79] | 0/3 = 0 % [0, 56] |
| A2 316 ms | 2/3 = 67 % [21, 94] | 0/3 = 0 % [0, 56] | 1/3 = 33 % [6, 79] |
| A2 506 ms | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] | 0/3 = 0 % [0, 56] |
| A3 316 ms | 3/3 = 100 % [44, 100] | 2/3 = 67 % [21, 94] | 0/3 = 0 % [0, 56] |
| A3 506 ms | 2/3 = 67 % [21, 94] | 1/3 = 33 % [6, 79] | 1/3 = 33 % [6, 79] |

Gate fired (the bursts that could have interrupted): baseline 14/18 = 78 % [55, 91], v1 14/18 = 78 % [55, 91], v3 14/18 = 78 % [55, 91]

v1: what the hold did

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 0 | 3 | 2 | 5 |
| no_gate | 1 | 2 | 1 | 4 |
| released:caller | 2 | 1 | 3 | 6 |
| released:playback_ended | 3 | 0 | 0 | 3 |

v3: what the hold did

| outcome | A1 | A2 | A3 | all |
|---|---|---|---|---|
| dropped_echo | 4 | 3 | 2 | 9 |
| dropped_echo_signal | 0 | 0 | 1 | 1 |
| dropped_noise | 0 | 1 | 0 | 1 |
| no_gate | 1 | 1 | 2 | 4 |
| released:caller | 1 | 1 | 1 | 3 |

v3 releases other than at playback end:

| trial | kind | delay | level | reason | held ms | ASR text | unmatched | psr | psr_why | self-interruption |
|---|---|---|---|---|---|---|---|---|---|---|
| Am06 | A2 | 316 | -35 | caller | 666 | '您刚才提到的上班问题。' | 2 | None | no_bypass | yes |
| Am08 | A3 | 506 | -5 | caller | 1724 | '但最多四张副卡，还有。' | 2 | None | no_bypass | yes |
| Am12 | A1 | 506 | -20 | caller | 1707 | '的费用，不会出现任何问题。' | 4 | None | no_bypass | no |

v3 release reasons: {'caller': 3}; drops: {'echo': 10, 'noise': 1, 'echo_signal': 1}.

psr_why over all v3 checks: {None: 58, 'no_bypass': 35}.

## B. Real barge-ins, no echo

| condition | barge-ins | speech_started | onset -> speech_started ms (P50 / P95) | onset -> flush ms | held_ms | speech_stopped -> transcript ms | dropped (no speech_started) |
|---|---|---|---|---|---|---|---|
| baseline | 30 | 30/30 = 100 % [89, 100] | 490 / 1249 (n=30) | 490 / 1249 (n=30) | n/a | 229 / 305 (n=30) | 0 |
| v1 | 30 | 30/30 = 100 % [89, 100] | 648 / 1565 (n=30) | 648 / 1565 (n=30) | 155 / 348 (n=30) | 227 / 306 (n=30) | 0 |
| v3 | 30 | 30/30 = 100 % [89, 100] | 676 / 1457 (n=30) | 676 / 1457 (n=30) | 153 / 406 (n=30) | 227 / 305 (n=30) | 0 |

Paired by trial, v1 minus baseline speech_started latency: P50 +150 ms, P95 +342 ms, mean +176 (n=30).

Paired by trial, v3 minus baseline speech_started latency: P50 +162 ms, P95 +422 ms, mean +200 (n=30).

v1 outcomes: {'released:caller': 30}

v1 real turns dropped: 0 

v3 outcomes: {'released:caller': 30}

v3 real turns dropped: 0 

Identical committed transcripts, baseline vs v3: 22/30.

## Be. Real barge-ins, steady echo

| condition | barge-ins | speech_started | onset -> speech_started ms (P50 / P95) | onset -> flush ms | held_ms | speech_stopped -> transcript ms | dropped (no speech_started) |
|---|---|---|---|---|---|---|---|
| baseline | 10 | 10/10 = 100 % [72, 100] | 559 / 1807 (n=10) | 560 / 1807 (n=10) | n/a | 217 / 293 (n=10) | 0 |
| v1 | 10 | 10/10 = 100 % [72, 100] | 712 / 2225 (n=10) | 712 / 2225 (n=10) | 152 / 270 (n=10) | 219 / 276 (n=10) | 0 |
| v3 | 10 | 10/10 = 100 % [72, 100] | 733 / 2172 (n=10) | 733 / 2172 (n=10) | 150 / 299 (n=10) | 212 / 290 (n=10) | 0 |

Paired by trial, v1 minus baseline speech_started latency: P50 +122 ms, P95 +1061 ms, mean +213 (n=10).

Paired by trial, v3 minus baseline speech_started latency: P50 +148 ms, P95 +956 ms, mean +226 (n=10).

v1 outcomes: {'released:caller': 10}

v1 real turns dropped: 0 

v3 outcomes: {'released:caller': 10}

v3 real turns dropped: 0 

Identical committed transcripts, baseline vs v3: 4/10.

## C. Short utterances over playback

| utterance | baseline: barge-ins; onset -> speech_started ms | v1 | v3 |
|---|---|---|---|
| wait | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| 好的 | 2/2; 445 / 447 (n=2) | 2/2; 3806 / 3807 (n=2); released:turn_end 2 | 2/2; 1267 / 1267 (n=2); released:pause 1, released:turn_end 1 |
| 嗯 | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| OK | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| 对 | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| hold on | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| 不是 | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| 可以 | 0/2; n/a | 0/2; n/a; no_gate 2 | 0/2; n/a; no_gate 2 |
| 等一下 | 2/2; 448 / 448 (n=2) | 2/2; 592 / 624 (n=2); released:caller 2 | 2/2; 615 / 618 (n=2); released:caller 2 |
| hello | 2/2; 475 / 481 (n=2) | 2/2; 4105 / 4122 (n=2); released:turn_end 2 | 2/2; 1581 / 1583 (n=2); released:pause 2 |
| 喂 | 2/2; 468 / 486 (n=2) | 2/2; 3596 / 3607 (n=2); released:turn_end 2 | 2/2; 1164 / 1186 (n=2); released:pause 2 |

v1 by outcome: released:caller: n=2, 592 / 624 (n=2); released:turn_end: n=6, 3806 / 4114 (n=6); dropped real: 0

v3 by outcome: released:caller: n=2, 615 / 618 (n=2); released:pause: n=5, 1267 / 1583 (n=5); released:turn_end: n=1, 1267 / 1267 (n=1); dropped real: 0

## F. Double talk: a caller speaks while a leak burst is on the line

F1: the caller's speech starts with the burst; F2: 0.5-1 s into it. Caller clips as in B. Leak -5 / -20 dB, 316 / 506 ms. A caller 'dropped' = the turn never reached the client as speech_started.

| condition | kind | leak dB | n | caller reached the client | onset -> speech_started ms | v3 outcomes | max psr per trial (P50 / max) |
|---|---|---|---|---|---|---|---|
| baseline | F1 | -5 | 6 | 6/6 = 100 % [61, 100] | 454 / 497 (n=6) | - | n/a |
| baseline | F1 | -20 | 6 | 6/6 = 100 % [61, 100] | 486 / 992 (n=6) | - | n/a |
| baseline | F2 | -5 | 6 | 6/6 = 100 % [61, 100] | 356 / 948 (n=6) | - | n/a |
| baseline | F2 | -20 | 6 | 6/6 = 100 % [61, 100] | 517 / 1004 (n=6) | - | n/a |
| v3 | F1 | -5 | 6 | 6/6 = 100 % [61, 100] | 1913 / 2939 (n=6) | released:caller 6 | 28.9 / 62.8 |
| v3 | F1 | -20 | 6 | 6/6 = 100 % [61, 100] | 860 / 1180 (n=6) | released:caller 6 | 10.4 / 14.2 |
| v3 | F2 | -5 | 6 | 6/6 = 100 % [61, 100] | 1315 / 1618 (n=6) | released:caller 6 | 18.6 / 26.5 |
| v3 | F2 | -20 | 6 | 6/6 = 100 % [61, 100] | 714 / 1227 (n=6) | released:caller 6 | 8.3 / 19.5 |

baseline overall: reached 24/24 = 100 % [86, 100]; dropped 0.

v3 overall: reached 24/24 = 100 % [86, 100]; dropped 0.

v3 hold time of released callers (held_ms): 378 / 1403 (n=24).

Per trial (v3):

| trial | kind | leak dB | outcome | held/drop ms | max psr | checks | dropped text |
|---|---|---|---|---|---|---|---|
| F00 | F2 | -20 | released:caller | 1163 | 19.46 | 5 | - |
| F01 | F1 | -20 | released:caller | 674 | 14.01 | 3 | - |
| F02 | F1 | -20 | released:caller | 125 | 8.25 | 1 | - |
| F03 | F1 | -5 | released:caller | 662 | 16.2 | 3 | - |
| F04 | F2 | -5 | released:caller | 156 | None | 1 | - |
| F05 | F2 | -20 | released:caller | 661 | 11.45 | 3 | - |
| F06 | F1 | -5 | released:caller | 121 | 8.07 | 1 | - |
| F07 | F1 | -5 | released:caller | 127 | None | 1 | - |
| F08 | F2 | -5 | released:caller | 1186 | 18.56 | 5 | - |
| F09 | F2 | -5 | released:caller | 122 | 9.29 | 1 | - |
| F10 | F2 | -5 | released:caller | 344 | 8.43 | 2 | - |
| F11 | F1 | -5 | released:caller | 1701 | 31.31 | 7 | - |
| F12 | F1 | -5 | released:caller | 1439 | 62.83 | 6 | - |
| F13 | F2 | -20 | released:caller | 155 | 7.67 | 1 | - |
| F14 | F2 | -20 | released:caller | 398 | 9.02 | 2 | - |
| F15 | F1 | -20 | released:caller | 665 | 12.49 | 3 | - |
| F16 | F2 | -20 | released:caller | 112 | 6.48 | 1 | - |
| F17 | F1 | -20 | released:caller | 351 | 14.17 | 2 | - |
| F18 | F1 | -20 | released:caller | 117 | 8.25 | 1 | - |
| F19 | F1 | -20 | released:caller | 152 | 8.22 | 1 | - |
| F20 | F1 | -5 | released:caller | 1202 | 28.95 | 5 | - |
| F21 | F2 | -5 | released:caller | 959 | 21.6 | 4 | - |
| F22 | F2 | -5 | released:caller | 1179 | 26.46 | 5 | - |
| F23 | F2 | -20 | released:caller | 358 | 7.65 | 2 | - |

Callers dropped by v3: []

## Cost of the checks

| version | scenario | held turns | checks per held turn (mean) | max |
|---|---|---|---|---|
| v1 | A | 23 | 2.3 | 5 |
| v1 | Am | 14 | 5.9 | 15 |
| v1 | B | 30 | 1.1 | 2 |
| v1 | Be | 10 | 1.1 | 2 |
| v1 | C | 8 | 10.8 | 15 |
| v3 | A | 18 | 6.3 | 11 |
| v3 | Am | 15 | 6.2 | 9 |
| v3 | B | 30 | 1.3 | 3 |
| v3 | Be | 10 | 1.1 | 2 |
| v3 | C | 8 | 3.4 | 5 |
| v3 | F | 24 | 2.8 | 7 |

v3 checks by kind: {'partial': 317, 'final': 23, 'utterance': 8}; psr_why: {None: 257, 'no_bypass': 91}. Partial-check latency (result less end of its audio, ms P50 / P95): 151 / 186 (n=317).

speech_stopped -> transcript ms (P50 / P95):

| scenario | condition | latency |
|---|---|---|
| B | baseline | 229 / 305 (n=30) |
| B | v1 | 227 / 306 (n=30) |
| B | v3 | 227 / 305 (n=30) |
| Be | baseline | 217 / 293 (n=10) |
| Be | v1 | 219 / 276 (n=10) |
| Be | v3 | 212 / 290 (n=10) |

## Offline what-ifs: confirm_echo_psr and confirm_echo_drop_psr

Replayed from the v3 captures' per-check `psr` and verdicts (a would-be release is a check with verdict `caller`, `echo_signal` or `echo_signal_drop`). Limits: the recorded sequence of a turn stops at the decision made with 10 / 15. A lower `confirm_echo_psr` than 10 would hold some releases that were recorded as released, and the later checks that would decide them do not exist: those are counted as censored (unknown, between 'still held' and a later release). Higher thresholds and every `confirm_echo_drop_psr` are exact. A/Am: 'self-interruptions' = releases by a check (not at the playback's end) over the fix bursts; B, Be, F, C: drops of real callers and the release delay (check time less the onset in the injected audio, ms).

| psr | drop psr | A self-int. | Am self-int. | B | Be | C | F |
|---|---|---|---|---|---|---|---|
| 8 | 10 | 7/36 (+1 cens.) | 3/18 (+0 cens.) | 0 drop, 9 cens.; 863 / 1290 (n=21) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 2 cens.; 1279 / 1596 (n=6) | 0 drop, 11 cens.; 1418 / 2600 (n=13) |
| 8 | 15 | 7/36 (+1 cens.) | 3/18 (+0 cens.) | 0 drop, 9 cens.; 863 / 1290 (n=21) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 2 cens.; 1279 / 1596 (n=6) | 0 drop, 11 cens.; 1418 / 2600 (n=13) |
| 8 | 20 | 7/36 (+1 cens.) | 4/18 (+0 cens.) | 0 drop, 9 cens.; 863 / 1290 (n=21) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 2 cens.; 1279 / 1596 (n=6) | 0 drop, 11 cens.; 1418 / 2600 (n=13) |
| 8 | 25 | 7/36 (+1 cens.) | 4/18 (+0 cens.) | 0 drop, 9 cens.; 863 / 1290 (n=21) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 2 cens.; 1279 / 1596 (n=6) | 0 drop, 11 cens.; 1418 / 2600 (n=13) |
| 8 | null | 7/36 (+1 cens.) | 4/18 (+0 cens.) | 0 drop, 9 cens.; 863 / 1290 (n=21) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 2 cens.; 1279 / 1596 (n=6) | 0 drop, 11 cens.; 1418 / 2600 (n=13) |
| 10 | 10 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 1157 / 2161 (n=24) |
| 10 | 15 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 1157 / 2161 (n=24) |
| 10 | 20 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 1157 / 2161 (n=24) |
| 10 | 25 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 1157 / 2161 (n=24) |
| 10 | null | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 1157 / 2161 (n=24) |
| 12 | 10 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 812 / 2161 (n=24) |
| 12 | 15 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 812 / 2161 (n=24) |
| 12 | 20 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 812 / 2161 (n=24) |
| 12 | 25 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 812 / 2161 (n=24) |
| 12 | null | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 689 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 812 / 2161 (n=24) |
| 15 | 10 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 681 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 656 / 2161 (n=24) |
| 15 | 15 | 8/36 (+0 cens.) | 3/18 (+0 cens.) | 0 drop, 0 cens.; 681 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 656 / 2161 (n=24) |
| 15 | 20 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 681 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 656 / 2161 (n=24) |
| 15 | 25 | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 681 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 656 / 2161 (n=24) |
| 15 | null | 8/36 (+0 cens.) | 4/18 (+0 cens.) | 0 drop, 0 cens.; 681 / 1471 (n=30) | 0 drop, 0 cens.; 745 / 2185 (n=10) | 0 drop, 0 cens.; 1240 / 1595 (n=8) | 0 drop, 0 cens.; 656 / 2161 (n=24) |

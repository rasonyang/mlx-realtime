# Live verification of #8, #9 and #10, 2026-10-31: the paraphrase works, but five garbled echoes reached AICC

Issue #5. The full verification call, on a line with strong echo: Zoiper on speakerphone, Zoiper's AEC
off, volume at max. The gateway is main e50b528 (#9, #8, #10), the same process as the first C6 repeat
(`live_c6_20261031.md`). That call had no audible echo and exercised nothing; this one has plenty.

| | |
|---|---|
| Capture | `sess_gw87l4skamgxr7o3`, 267.08 s of input (wall 267.13 s: 06:18:17.063Z to 06:22:44.193Z); 18 holds, 141 `hold_check`s |
| AICC | aiccCallId `01a1909b-e03f-7729-addf-9550afcc3e23`, main a3ccde1 |
| Line | `analyze.py` class **weak**; ERL 0.6 dB (the echo is as loud as our reference); the owner's speech is 8-10 dB louder than the echo |
| Owner (authoritative) | said "为什么天是蓝的？" "during the greeting" and the bot answered; about 3 minutes of silence from about 06:18:45Z to 06:21:45Z; then 7 or more barge-ins; noticed the bot stopping by itself |
| Probes before it | `sess_u7f8xtwlag7dch4k` (AEC on, class none, no hold); `sess_kbinl121cwejjqpc` (AEC off, class weak, ERL 42.1 dB, one leak held and dropped) |

Input time `t` is seconds from the session start; the AICC clock is `t + 06:18:17.063Z` and every floor-take
below matches a gateway `speech_started` within 50 ms. Quotes are short transcript excerpts. The
audio (capture and AICC recording) holds the owner's voice and was read in place only. The ASR on 8102
was not used; the hold checks are the recorded ones. `analyze.py` (#11) rewrote the git-ignored
`report.md` and `report.json` of the capture.

## Verdicts

| criterion | verdict | evidence |
|---|---|---|
| #9 (a) the paraphrase is released and answered, contiguity < 0.7 | **PASS** | "为什么天是蓝的？" (t = 26.2-28.8 s): contiguity 0.00; released at its first check with the owner's words; answered. But it was said over the 19-28 s reprompt, not the greeting (below), and the first audio of the answer came 3.7 s after the question ended |
| #9 (b) every leak in silence held and dropped, 0 `speech_started` from echo | **FAIL** | 4 leaks held and dropped as echo (9, 17, 21 and 10 units, all ours). **6 leaks released or unheld, 5 of them in the silent part**: t = 102.56, 122.91, 132.03, 172.76, 206.76; plus 4.89 in the greeting. The final transcripts of 122.91, 132.03, 172.76 and 206.76 were judged callers and answered |
| #8 a barge-in whose VAD turn began on echo | **NOT EXERCISED on a real merge; FAIL on a false one** | The one real case (the paraphrase, a hold that began on echo) ended with no echo in the final transcript (`heard` null). `heard` ≠ `text` happened once, at 122.91, and `text` was a garble of the bot's own tail, not the owner's words; it was answered |
| #10 (a) a cut-edge mishearing on a matched hold is held | **PARTIAL** | 3 such checks were held (t = 36.94, 99.23, 122.68). 36.94 ended in a drop. 99.23 and 122.68 did not: the holds were released as callers at 102.56 and 122.91 by the next garble |
| #10 (b) a real barge-in over a matched hold is released within one interval | **PASS (1 case)** | the paraphrase: matched at check 0 ("现在的锂电池", 6 units, all ours); released at 27.62 s, the first check that contains the owner's words ("为什么？"): 0 ms delay. The 7 scripted barge-ins were fresh holds, not matched |
| #10 (c) no self-interruption from a partial-check release | **FAIL** | 3 releases of matched holds on echo garbles: t = 102.56, 122.91, 206.76 |

Self-interruptions overall: 6 (4.89, 102.56, 122.91, 132.03, 172.76, 206.76). Real caller turns
dropped: 0. Dead air after a dropped turn: 2 (18 s and 9 s of reprompt wait, below).

## The call, turn by turn

`playedMs` is AICC's. "Rule" is what released the turn.

| t (s) | AICC (UTC) | what | gateway | rule / class |
|---|---|---|---|---|
| 4.89 | 06:18:21.966 floor, playedMs 4720 | greeting cut after 4.7 s | fresh hold at 4.75; one check "买点，我是。" (4 units, 2 not ours, PSR 7.06); released after 138 ms; the final transcript "我是一位科普讲解员，您可以随便。" is the greeting (14 units, echo score 1.0), dropped as echo | **leak, fresh hold** |
| 10.4-18.4 | 06:18:35.451 dead air; 06:18:36.055 latency 8823 | no answer to the dropped echo turn until AICC's no-input cue; reprompt "您还在听吗？" at 19.0 s | | the 8823 ms is false: a timer from the echo turn's speech_stopped (10.16 s) to the cue's reply |
| 20.4-25.3 | none | echo of the reprompt, 8 checks all ours | hold, dropped as echo at 25.3 ("刚才您似乎有些犹豫。", 9 units) | **held and dropped** |
| 26.2-28.8 | 06:18:44.691 floor, playedMs 8620 | **the paraphrase**, over the reprompt | matched hold (checks 0 and 1: "现在的锂电池。", all ours); released at 27.62 on "为什么？" (3 units, 3 not ours, contiguity 0.00); final "为什么天是蓝的？" (7 units, score 0.0) | real, answered |
| 33.6-41.4 | none | echo of the answer | hold, 18 checks, dropped as echo (17 units) | **held and dropped** |
| 77.9-84.3 | none | echo | hold, 22 checks, dropped as echo (21 units) | **held and dropped** |
| 93.4-102.6 | 06:19:59.631 floor, playedMs 10980 | **leak** after 8.5 s of hold; the reprompt was cut | matched hold; release check [98.36-102.36] "充电罢了。现在的锂电池其实并没有所谓的记忆。" (20 units, 2 not ours, contiguity 0.80); final transcript the reprompt (45 units, score 1.0), dropped | **leak, #10 inside-the-edges branch** |
| 103.7-113 | 06:20:09.083 dead air; 06:20:09.759 latency 9004 | no answer until the cue | | false latency 9004 ms (timer from speech_stopped at 103.68 s) |
| 114.4-122.9 | 06:20:19.985 floor, playedMs 10220 | **leak** after 7.8 s; reprompt cut again | matched hold; release check "用的手机充电了，现在的锂电池其实并不难。" (18 units, 2 not ours, contiguity 0.83); the check before ("…并不足。", 2 not ours) was held; final `text` "不容易记忆效应。" (7 units, score 0.57), `heard` = the bot's full text | **leak, #10 persistence**; then **#8 remainder** called the tail a caller |
| | 06:20:22.063 latency 868 | the phantom turn was answered ("您刚才提到的记忆效应，确实是个常见的误区...") | | answered a turn nobody spoke |
| 131.2-132.0 | 06:20:29.105 floor, playedMs 7040 | **leak**, the answer cut at 7 s | fresh hold; one check "军事。" (2 units, 2 not ours); final "军事平台。" (4 units, score 0.0); the bot was saying "…以前的镍镉电池…" at that point: 镍镉电池 heard as 军事平台 | **leak, fresh hold on a garbled echo** |
| | 06:20:30.223 latency 626 | answered with a 29 s monologue about "军事平台" | | |
| 172.5-172.9 | 06:21:11.368 latency 794 (no floor-take) | **leak**, 0.3 s after our playback ended at 172.24 s | no hold (the bot was not playing); "研究成果。" (4 units, score 0.0) is the garbled tail of "…高精尖成果。" | **leak, unheld, contiguity 0.0 on 4 units** |
| 204.8-206.8 | 06:21:43.833 floor, playedMs 32480 | **leak**, a 32 s answer cut | matched hold ("的探索。", 3 units, all ours, 6 checks); release check "的探索，如今去珠峰基。" (9 units, 4 not ours, contiguity 0.56); final "的探索，如今去珠峰及整个地。" (12 units, score 0.42) | **leak, #10 inside-the-edges branch; #9 contiguity under 0.7** |
| | 06:21:45.545 latency 701 | phantom answered ("您提到的珠峰探索...") | | |
| 211.0 | 06:21:48.632 floor, playedMs 3080 | barge-in 1, "我想问问天气。" | fresh hold, released at 122 ms | real |
| 215.3 | 06:21:55.368 latency 631 | "我想问问大海。", said after the bot had finished (no floor-take needed) | unheld | real |
| 220.9-257.3 | seven floor-takes, 06:21:57.951 to 06:22:34.387 | barge-ins 2-8, see below | fresh holds | real |

The AICC pattern lists:

- **Floor-takes with no reply until the dead-air cue (AICC #404).** 06:18:21.966 and 06:19:59.631 are
  both leaks whose final turn was dropped as echo: the gateway had sent `speech_started`, AICC took
  the floor, and nothing answered. 06:18:21.966 is not the paraphrase. The greeting was cut at 4.7 s,
  before the sentence "比如天为什么是蓝色的" (about 13 s in) could be said.
- **Floor-takes and turns in the silent part.** All five are leaks: 06:19:59.631, 06:20:19.985,
  06:20:29.105, 06:21:11.368 and 06:21:43.833.

## Why they are leaks, not the owner

The owner was silent (their account), and the capture agrees:

| window | input level |
|---|---|
| the echo of the bot, held and dropped | -20 to -23 dBFS |
| the owner's speech (the paraphrase, the 7 barge-ins) | -12.4 to -13.4 dBFS |
| the leaks at 102.4, 122.9, 131.2, 205.5 | -20 to -22 dBFS: echo level, 8-10 dB below the owner |
| the leak at 172.5 | -27 dBFS (the decaying tail) |
| the leak at 4.3 | -40 dBFS |

The texts are garbles of the bot's own audio, matched by position: 军事平台 for 镍镉电池 at 5.2-5.9 s into
the answer, 研究成果 for 高精尖成果 at its end, 去珠峰基 for 却支撑起, 罢了 for 吧. The owner's scripted
sentences are all in the transcripts and are 8-10 dB louder.

## Which rule released each leak, and the narrowest fix

Contiguity of the **release check's** text against the played text (`similarity.contiguity`, recomputed
here from the recorded `hold_check` events), against the real barge-ins:

| release | check text | units / not ours | contiguity of the check |
|---|---|---|---|
| 102.56 matched hold, leak | 充电罢了。现在的锂电池其实并没有所谓的记忆。 | 20 / 2 | **0.80** |
| 122.91 matched hold, leak | 用的手机充电了，现在的锂电池其实并不难。 | 18 / 2 | **0.83** |
| 206.76 matched hold, leak | 的探索，如今去珠峰基。 | 9 / 4 | **0.56** |
| 4.89 fresh, leak | 买点，我是。 | 4 / 2 | 0.00 |
| 132.03 fresh, leak | 军事。 | 2 / 2 | 0.00 |
| 27.62 matched hold, **real** (paraphrase) | 为什么？ | 3 / 3 | 0.00 |
| the 7 scripted barge-ins, fresh, real | 等一下。 / 我想问问欧洲。 / ... | 3-9 / 3-9 | 0.00 |

Held checks with 2 not-ours units on a matched hold also read as echo by contiguity: 0.85 (36.94),
0.90 (99.23), 0.84 (122.68).

1. **Matched holds (102.56, 122.91, 206.76): the partial-check release counts units on the in-order
   match and does not look at contiguity.** #9 changed the complete-utterance checks and the unheld
   rule, and left "partial-check releases still count unmatched units on the in-order match" (PR #18).
   A matched hold's check text that is mostly our words, with a garble of 2-4 units in it, is released
   by the inside-the-edges branch (the garble sits mid-window, not at the cut edge) or by persistence
   (two garbles of the same stretch in a row, "并不足" then "并不难": the PR's "persistence counts
   units" assumption fails when the ASR garbles the same phrase twice).
   **Narrowest fix:** on a matched hold, a partial check is held while its own contiguity is at least
   0.5, whatever its unmatched count. The data separates cleanly: 0.56-0.83 for the three leaks and
   0.85-0.90 for the held garbles, against 0.00 for the only real release on a matched hold and for
   every real barge-in. It adds no delay for a caller over echo whose check shows the caller's
   words alone; for a caller whose words are mixed with the echo in the window, the release waits until the
   words are at least half of it (the final-turn rule of #8 still answers them). Needs a replay on
   all recorded holds (`hold_replay_v7`) before it is chosen: V1, V2 and call 2 have real releases
   on matched holds.
2. **Fresh holds (4.89 and 132.03) and the unheld 172.76.** The echo is a 2-4 unit garble that shares
   no 3-unit run with our text, so no text rule sees it. What separates them on this call is level:
   the input at the leak is 8-10 dB below the owner's speech at the same volume (and 27 dB below at
   4.3 s; on the first C6 repeat the 70.4 s burst was 28-30 dB below the callers). I propose no
   change yet: a level gate on a fresh hold's release (the window's input level against the callers'
   recent level) needs the V1, V2 and call 2 levels to be sure it does not delay a quiet real caller.
   The cost to bound is a real barge-in held until the next check (250 ms). A cheaper step for the
   unheld case: `echo_match_min_units` is 4 and "研究成果" has exactly 4; the garble shares 成果 (2 units)
   with the played tail, so it is 4/4 = 0.0 contiguity, not 0.5. No simple threshold change catches it.
3. **#8's remainder (122.91).** `after_echo` cut the bot's tail "…并没有所谓的" and called the rest
   ("不容易记忆效应", 7 units) a caller: `confirm_min_new_units` and `echo_match_min_units` were met, and
   the remainder's contiguity 0.57 is below 0.7 because three units are garbled. Same fix family: a
   remainder that is itself 0.5 contiguous (4 of 7 units: 记忆效应) with no unit unmatched at the
   start of the remainder is echo. Do this together with 1, not separately.

## Real barge-ins

The owner's seven scripted sentences ("等一下，我想问问X", one a plain "我想问问天气"). Onset is the
VAD's onset (`hold.onset_ms`); `analyze.py`'s caller onset is not usable here because it takes the
echo's start (up to 2 s earlier) for the caller's. All are fresh holds with the owner's words in the
first check: no matched hold, no #10 delay.

| # | transcript (`text`) | hold onset (s) | release (s) | after onset | AICC floor-take (UTC) | `playedMs` | bot stopped |
|---|---|---|---|---|---|---|---|
| 1 | 我想问问天气。 | 210.98 | 211.56 | 122 ms hold | 06:21:48.632 | 3080 | yes |
| 2 | 等一下，我想问问人生。 | 220.32 | 220.88 | 121 ms | 06:21:57.951 | 2580 | yes |
| 3 | 我想问问美国。 | 227.90 | 228.84 | 130 ms | 06:22:05.921 | 2600 | yes |
| 4 | 等一下，我想问问欧洲。 | 233.95 | 235.87 | 157 ms | 06:22:12.947 | 2620 | yes |
| 5 | 我想问中国。 | 239.39 | 239.98 | 128 ms | 06:22:17.059 | 2860 | yes |
| 6 | 等一下，我想问问日本。 | 245.06 | 246.69 | 156 ms | 06:22:23.767 | 2580 | yes |
| 7 | 等一下，我想问问韩国。 | 248.93 | 250.61 | 156 ms | 06:22:27.687 | 2620 | yes |
| 8 | 等一下，我想问问台湾。 | 255.49 | 257.31 | 155 ms | 06:22:34.387 | 2620 | yes |

The seven floor-takes at 06:21:57-06:22:34 (AICC pattern 3) are barge-ins 2-8: each transcript is the
scripted sentence, the input is -12.4 to -13.4 dBFS, and each `playedMs` of 2.6 s is the 2 s after
the bot's start plus the 0.5 s playout delay. They are not the bot's own echo. The transcript of each
reached the LLM whole and was answered (AICC latencies 515-754 ms, from the speech_stopped).

Barge-in 1 was answered "请问有什么可以帮您？" (a generic reply to "我想问问天气。": the LLM, not the
transport).

## Slow answers: Smart Turn waited 3 s on 6 of 17 turns

The gateway's `turn timeline` shows `turn_confirmed` at 2.98-3.04 s after `speech_end` on six turns, against
0.59-0.61 s on the rest: the echo turn at 4.3 s, the paraphrase "为什么天是蓝的？", and the barge-ins
"等一下，我想问问人生" (2), "我想问问美国" (3), "我想问中国" (5) and "等一下，我想问问韩国" (7). So the
paraphrase was answered promptly in the AICC log (latency 668 ms, 06:18:49.600Z) but the first audio
started 4.2 s after the question ended at 28.3 s (`first_audio_sent` 3713 ms after `speech_end`). The owner heard
the answer, as they say, but 3.7 s late. AICC's latency starts at speech_stopped and hides the 3 s.
The same four barge-ins have a 3.7 s answer. The first C6 repeat (10-31) has no such wait
(its turns confirmed in about 0.6 s). Not examined further: the cause (the end-of-turn model's "unfinished" decision timing out)
needs the model's output; flag for a separate look.

## G7's numbers, against the two earlier C6 calls

| | C6 10-30 | C6 10-31 (first repeat) | this call |
|---|---|---|---|
| Bulk delay (reference file to input) | 540-600 ms, above the [5, 450] range | 571.5 ms | 73.8 ms; 373.8 ms after the playout model; filter tap 95.7 ms: inside the range |
| ERL | | 45.4 dB | **0.6 dB** (filter estimate 2.8): the echo equals our reference |
| AEC3 ERLE | 32 dB (`meta`) | `meta` 22.3 dB, no echo to cancel | `meta` 18.6 dB; steady-state median 41.6 dB (23 windows, converged 38.8 s after the first answer); but over the leak windows the AEC3 output was only 3-6 dB below the input |
| Speex ASR path | 19 realigns, ERLE 2.6 dB | 0 realigns, 1.9 dB | 29 realigns, ERLE 3.3 dB (`meta`; analyzer 11.3 dB steady-state); aligned at 22.0 s; reference 20.0 ms ahead of the echo (target 20) |
| Bypass | engaged once, released once | engaged at 3.0 s, released at 9.5 s, engaged again to the end | engaged at 3.5 s, released at 22.0 s ("echo on the line"), then on; 7 AEC3 realigns; not flapping |
| Uplink samples vs wall | +0.02 % | -0.04 % | **-0.02 %** (267.08 s of 267.13 s) |
| AICC uplink clock | received 9422, sent 9650, filled 236, backlog 8 | received 10874, sent 11046, filled 193, backlog 21 | received 13083 (261.7 s, -2.0 % of wall), sent 13356 (267.1 s), filled 286, dropped backlog 13 |

The line behaves as G7 assumed here (delay in range, aligned), and the leaks still happen: the
ASR path (Speex, 3.3 dB ERLE) hears the loud echo almost word for word, and nothing but the text
rules stands between it and AICC. The correlation veto was blind: `psr_why no_bypass` on every check after
22.0 s, when AEC3 left the bypass (PSR is only measured in bypass).

## Other

- **Gateway log.** No error or warning in the session. 29 Speex and 7 AEC3 realign lines, one
  "no echo ... bypassed" and one "echo on the line" line. "a turn matched our own speech: treated as
  noise" twice, "a barge-in turned out empty or echo" twice, "a held turn was dropped" four times.
- **AICC log.** No barge-guard, provider floor-take or removed-turn line in this call. The 06:18:17.237 WARN
  "first bot audio before media" (187 ms) is expected at call start.
- **Dead air.** Three 8 s no-input cues (06:18:35.451, 06:19:47.967 and 06:20:09.083),
  each followed by the reprompt. Two (06:18:35 and 06:20:09) were caused by a dropped
  echo turn after a leak: 18 s and 9 s without an answer the owner had no reason to wait for.
- **False latencies.** 8823 ms (06:18:36.055Z) and 9004 ms (06:20:09.759Z): the timer of an echo turn
  that AICC saw as speech and the gateway dropped.
- **`analyze.py`.** Its "self-interruptions: 2" (at 4.89 and 102.56) are the two leaks whose final turn was an
  echo turn. The other four self-interruptions were judged `real` by the gateway and are not found by it.
  Its onset column is the echo's, not the caller's, on this loud line.

## Open

- Needs the owner: whether the greeting played at the start for 4.7 s only (it was cut by the leak); that
  the sentence "为什么天是蓝的？" was said at 26.2 s, over "您还在听吗？", and not during the greeting.
- Needs the AICC recording (left caller, right bot): listening to 06:19:59.6Z, 06:20:19.9Z, 06:20:29.1Z,
  06:21:11.4Z and 06:21:43.8Z would confirm the five silent-part leaks by ear; the level and text
  evidence above is conclusive for the owner's account but is not listening.
- Needs a replay before the proposed contiguity gate on matched holds is chosen (`hold_replay_v7` over
  all recorded holds), and the V1, V2 and call 2 input levels before any level gate is.

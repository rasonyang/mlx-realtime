# Hold v7 replay: a mishearing at a partial check's cut edge does not release a held echo

Issue #10. On the live C6 call a held turn began at input 23.2 s on the echo of the end of the
greeting ("...也不会反问你"). The first partial check heard "也不会。" (0 units not ours), the second
"也不会否认。": "否认" for "反问" made 2 units not ours (`confirm_min_new_units`), and the hold was
released after 403 ms (`speech_started` at 04:00:49.79). The turn's final transcript was the echo.

## The rule

A hold is *matched* once a check of it was all ours: `MIN_RUN` (3) units or more, none outside the
in-order match. (`echo_match_min_units`, 4, would not have caught C6: the first check had 3 units.
`MIN_RUN` is the length of run that `contiguity` already calls echo, #9; no new config key.)

On a matched hold a partial check that reaches the release count (`confirm_min_new_units` while
the client plays, 1 after) releases only if

- `confirm_min_new_units` of its units that are not ours lie inside the transcript
  (`text.similarity.unmatched_inside`): without its last unit, and, once the 4 s cap slid the
  window's start past the hold's start, without its first unit too, the units a partial check's
  cut truncates into other words (the first version excluded the last unit only; the fresh checks
  below showed the slid window's first unit mishearing the same way), or
- the previous partial check also reached the count and was held (*persist*).

Persist is the count, not the same words: a caller's words reappear as the ASR hears them, with
other homophones, and a mishearing at an edge does not recur once the audio grows past it (the
syllable is whole in the next check). Two edges mishearing in a row is the residual risk. A fresh
hold keeps the rule of #5. `hold.last_new`, which the early release at the playback's end reads, is
0 for a check that was held or is not firm, so the early release and the rule agree: a matched hold
is not released at the playback's end on a unit the rule has not accepted.

## Method

Offline, no models: `spikes/live_echo/hold_replay_v7.py` walks the recorded `hold_check` sequences
of every live capture and of the synthetic v3 captures (v6's verdicts for complete-utterance
checks, v7 for partial ones, the correlation veto as in v5). A recorded partial check with verdict
`caller` or `echo_signal` reached the count. The sequences stop at the recording build's own
decision, so a release that v7 delays to a check that was never recorded cannot be followed: the
table at the end lists these as "needs ASR".

    uv run --locked python spikes/live_echo/hold_replay_v7.py > spikes/live_echo/results/hold_v7_replay.md

## Must hold

| requirement | result |
|---|---|
| C6 04:00:49.79 (item 56h0, input 22.90-23.91 s) is not released | the check ("也不会否认。", 2 not ours, 1 inside the cut edges) is held, and the hold is matched (check 0: "也不会。"). It disappears if the next check does not reach the count: the next partial check was never recorded (the build released here); the turn's final transcript was the echo "也不会反问你。" (6 units, overlap 0.83). Fresh ASR check pending |
| C6 04:02:40 (item ggyf, input 130.56-134.56 s) survives, delayed by at most one interval | the check ("...关于市场。", 2 not ours, 1 inside the cut edges) is held on a matched hold (check 0: 4 units, 0 not ours); the release is at the next check (about 250 ms) if it reaches the count too: it was never recorded. Fresh ASR check pending |
| V1, V2, call 1, call 2 real barge-ins: delay at most one interval | 0 ms for all of them (the 19 holds of these four captures). Only call 2's first hold (zg04) was matched (check 0: "可以随便问。", all ours); its release ("可以随便问我一个简短的问题。等一下。", 3 not ours, 2 inside the cut edges) is firm, so it is at the same check. The others never had an all-ours check before the caller's words |
| synthetic F (a caller over a leak) and B (callers) | 0 of 24 and 0 of 30 holds delayed or changed in the recorded sequences: none had a matched check before its release. The latency cost on a synthetic hold that matches first is not in these captures |

## v7 against v6

## Holds v7 decides at another check, from the recorded checks

| capture | item | recorded | v6 | v7 | delay ms | v6 deciding check | unmatched | inside the cut edges |
|---|---|---|---|---|---|---|---|---|
| none | | | | | | | | |

## Holds whose release v7 defers to a check that was never recorded

| capture | item | recorded | v6 | v7 | delay ms | v6 deciding check | unmatched | inside the cut edges |
|---|---|---|---|---|---|---|---|---|
| live:C6 (10-30) | 56h0 | released caller | released @1 | undecided (needs ASR) | - | 也不会否认。 | 2 | 1 |
| live:C6 (10-30) | ggyf | released caller | released @15 | undecided (needs ASR) | - | 只是暂时没想好问什么，那我先给您讲个关于市场。 | 2 | 1 |

## Outcomes per capture group

| group | held turns | v6 released | v6 dropped | v7 released | v7 dropped | v7 undecided | outcome changed | delayed | max delay ms |
|---|---|---|---|---|---|---|---|---|---|
| live:call 1 (10-29) | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| live:call 2 (10-29) | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| live:V1 (10-30) | 5 | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| live:V2 (10-30) | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| live:C6 (10-30) | 10 | 5 | 5 | 3 | 5 | 2 | 2 | 0 | 0 |
| v3:F | 24 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| v3:A | 18 | 8 | 10 | 8 | 10 | 0 | 0 | 0 | 0 |
| v3:Am | 15 | 3 | 12 | 3 | 12 | 0 | 0 | 0 | 0 |
| v3:B | 30 | 30 | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| v3:Be | 10 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 0 |
| v3:C | 8 | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |

## Fresh ASR checks on C6 (`hold_next_check.py`)

The recorded sequences of 56h0 and ggyf end at the recording build's release.
`hold_next_check.py` re-slices the capture's `asr.wav` (the ASR path's track) as a check does,
transcribes the last recorded check again, five further partial checks (`confirm_step_ms` 250
each) and the whole turn, and walks them through `walk7`. The played text of a fresh check is the
recorded check's. Re-slicing reproduces both recorded transcripts exactly.

    uv run --locked python spikes/live_echo/hold_next_check.py \
        spikes/live_echo/captures/live/20261030T040025725Z_sess_bnblfvhsgir6mt70 --items 56h0,ggyf --next 5

56h0 (recorded check interval 262 ms):

| check | input ms | recorded | fresh | not ours | inside the cut edges |
|---|---|---|---|---|---|
| last recorded, again | 22900-23910 | 也不会否认。 | 也不会否认。 | 2 | 1 |
| next 1-5 | 22900-24160...25160 | - | 也不会访问您。 (all five) | 1 | 1 |
| final | 22900-24032 | - | 也不会反问你。 | 1 | |

Pass. v7 holds at "也不会否认。", none of the five next checks reaches the count, and the final
check calls the turn echo: dropped as echo, no release.

ggyf (recorded check interval 268 ms):

| check | input ms | recorded | fresh | not ours | inside the cut edges |
|---|---|---|---|---|---|
| last recorded, again | 130560-134560 | 只是暂时没想好问什么，那我先给您讲个关于市场。 | same | 2 | 1 |
| next 1 | 130810-134810 | - | 暂时没想好问什么，那我先给您讲个关于日常生活。 | 0 | 0 |
| next 2 | 131060-135060 | - | 准备想好问什么？那我先给您讲个关于日常生活。 | 2 | 1 |
| next 3 | 131310-135310 | - | 想好问什么？那我先给您讲个关于日常生活。等一。 | 2 | 1 |
| next 4 | 131560-135560 | - | 问什么？那我先给您讲个关于日常生活。等一下。 | 3 | 2 |
| next 5 | 131810-135810 | - | 干什么？那我先给您讲个关于日常生活。等一下。 | 4 | 2 |
| final | 129876-137312 | - | 您可能只是暂时没想好问什么，那我先给您讲个关于日常生活。等一下，我想问问天气。 | 9 | |

Pass. Every window here is slid (starts after 129.876 s), so the first unit is left out too. The
mishearing of next 2 ("准备", 1 inside) is held and marks the hold suspect. The caller's own words
first appear at next 3 (window end 135310 ms, "等一"); it is the check that persists, so the hold
is released there: 0 ms after the words appear (804 ms after the recorded release, 3 intervals,
which is when the caller spoke, not a delay). Without the suspect from next 2, next 3 would be
held (1 inside) and next 4 (end 135560 ms, "等一下", 2 inside) would release: one interval
(250 ms) after the words appear. The recorded 04:02:40 release was a mishearing about 1 s before
the caller.

## Slid window replay

The recorded-data replay is unchanged by the leading-edge exclusion: no hold of any live or
synthetic capture changes outcome or moves its release (the table above), and the V1, V2, call 1
and call 2 barge-ins release at the same check. F and B do not change.

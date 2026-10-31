# Hold v5 replay: contiguity replaces the plain overlap, #7's low-PSR release is removed

Issue #9. On the live C6 call real echo correlated as weakly as real callers (PSR 6-9 for the first
~7 s of a leak; callers 5.7-9.7), so #7's rule (a transcript echo whose PSR is below
`confirm_echo_psr` is released) could release a real leak. v5 drops that rule and scores the
transcript against our played text by contiguity: the share of the transcript's units that lie in
runs of 3 or more units that also follow each other in the played text. The echo-turn rule uses it
in place of the in-order overlap, against the unchanged `echo_match_threshold` (0.7) and
`echo_match_min_units` (4). The unmatched-unit count that releases a partial check is unchanged.

## Method

Offline, no gateway and no models: `spikes/live_echo/hold_replay_v5.py` reads the `hold_check` events
of every live capture (2026-10-29 calls 1 and 2; 2026-10-30 V1, V2, C6) and of the synthetic v3
captures (`captures/hold_v3/fix`: leaks A and Am, real callers B, Be, C, F). Each event has the ASR
text, the assistant text played around the turn, the correlation and the recorded verdict. A turn is
walked check by check under three rules:

- v3: overlap rule, no correlation override (the build before #7);
- v4: overlap rule, and a complete-utterance check with a computed PSR below 15 releases a transcript
  echo (#7);
- v5: contiguity rule, no override.

Limits: a turn's recorded checks stop where the recording build decided, so a rule that decides later
than that build sees no further checks (`undecided`; none occurred). Partial checks are taken as
recorded. A turn released by a partial check has no complete-utterance check and no row in the first
table: the other live turns (call 1, V1 and V2 barge-ins; C6's four scripted barge-ins) are of that
kind and are unchanged by construction. The synthetic simulated captures (`captures/simulated`) hold
no hold events.

## Scores

| turn | overlap | contiguity |
|---|---|---|
| call 2 "为什么天是蓝的" over "比如天为什么是蓝色的" (a caller) | 0.86 | 0.43 |
| every C6 echo drop (5 leaks, 4 to 72 units) | 0.99-1.00 | 0.99-1.00 |
| lowest echo drop of the synthetic leaks A and Am | 0.91 | 0.91 |

Threshold: 0.7, the existing `echo_match_threshold`, now applied to the contiguity. It sits in the
middle of the gap between the paraphrase (0.43) and the lowest echo (0.91), so no new field. The
overlap could not separate them: the paraphrase scored 0.86, above 0.7.

In place of the overlap, not alongside it: a run of 3 or more units that is contiguous in both texts
is also an in-order match, so requiring both adds nothing (no complete check of 4 or more units in
any capture had an overlap below 0.7 and a contiguity at or above it).

## Must hold

| requirement | result |
|---|---|
| call 2 "为什么天是蓝的" is kept | pass: contiguity 0.43; v3 dropped it, v4 and v5 release it at the same check (index 6, a pause) |
| every C6 echo drop is still an echo drop | pass: 5 of 5, contiguity 0.99-1.00, same deciding check as v3 |
| no new release of a leak | pass: 0 differences from v3 on A (18 held turns), Am (15) and C6 (10) |
| no change for the synthetic callers | pass: B, Be, C, F (72 held turns): 0 differences from v3 and v4 |

v4 had released one synthetic leak (A22 "三个部分。", 4 units, PSR 13.45); v5 drops it again.

Giving up #7's low-PSR release means a caller who repeats our words in long runs (an echo by the
transcript rule) is dropped as in v3; only a paraphrase (short fragments) is kept.

## Correlation veto

`confirm_echo_psr` and `confirm_echo_drop_psr` are unchanged. The last table counts how often the veto
(15) decided anything under v5: a check it turned from a release into `echo_signal` or
`echo_signal_drop`, and the turns whose outcome that changed. It is counted on the recorded
sequences, which stop at the recording build's own decision.

- Live lines: it decided on 1 of 70 checks that had a PSR (V1 "我有个问题", PSR 22.6, the false veto
  of the issue; the turn was held a check longer). On C6 it flagged 8 of 43 checks and changed none:
  each was on a turn the transcript rule already held or dropped.
- Synthetic captures: it decided on 55 checks and changed 20 turns (leaks A 9, Am 5; F callers 6,
  whose release it delayed), where leaks reached PSR 10 or more.

So on the live line it decides almost nothing and was wrong once; on the synthetic leak model it does
hold leaks that the ASR heard as a caller. The owner decides whether to keep, raise or drop it.

## Turns of the live calls (complete-utterance check, or the last one)

| capture | item | transcript | units | psr | overlap | contiguity | recorded | v3 | v4 | v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| call 2 (10-29) | 3km6 | 为什么天是蓝的？ | 7 | 8.39 | 0.86 | 0.43 | dropped echo | dropped @7 | released @6 | released @6 |
| call 2 (10-29) | b0wt | 喂。 | 1 | 6.37 | 0.00 | 0.00 | released pause | released @3 | released @3 | released @3 |
| V2 (10-30) | 1ro6 | 嗯。 | 1 | 9.12 | 0.00 | 0.00 | released turn_end | released @2 | released @2 | released @2 |
| C6 (10-30) | y4po | 科普讲解员，您可以随便问我一个简短的问题，比如天为什 | 72 | None | 0.99 | 0.99 | dropped echo | dropped @51 | dropped @51 | dropped @51 |
| C6 (10-30) | a947 | 讲的时候，我不会打断。 | 9 | None | 1.00 | 1.00 | dropped echo | dropped @6 | dropped @6 | dropped @6 |
| C6 (10-30) | hbev | 于日常生。 | 4 | None | 1.00 | 1.00 | dropped echo | dropped @3 | dropped @3 | dropped @3 |
| C6 (10-30) | 1igy | 大脑就能恢复。 | 6 | None | 1.00 | 1.00 | dropped echo | dropped @4 | dropped @4 | dropped @4 |
| C6 (10-30) | fumh | 是暂时没想好问什么，那我先给您讲个关于日常。 | 20 | None | 1.00 | 1.00 | dropped echo | dropped @10 | dropped @10 | dropped @10 |

Outcome cells: released / dropped / undecided, `@n` the index of the deciding check.

## Turns where v5 differs from v3 or v4 (all captures)

| capture | item | transcript | units | psr | overlap | contiguity | recorded | v3 | v4 | v5 |
|---|---|---|---|---|---|---|---|---|---|---|
| call 2 (10-29) | 3km6 | 为什么天是蓝的？ | 7 | 8.39 | 0.86 | 0.43 | dropped echo | dropped @7 | released @6 | released @6 |
| A | s7nb | 三个部分。 | 4 | 13.45 | 1.00 | 1.00 | dropped echo | dropped @3 | released @3 | dropped @3 |

## Outcomes per capture group

| group | held turns | v3 released | v3 dropped | v3 undecided | v4 released | v4 dropped | v4 undecided | v5 released | v5 dropped | v5 undecided | v5 differs from v3 | v5 differs from v4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live:call 1 (10-29) | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| live:call 2 (10-29) | 8 | 7 | 1 | 0 | 8 | 0 | 0 | 8 | 0 | 0 | 1 | 0 |
| live:V1 (10-30) | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| live:V2 (10-30) | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| live:C6 (10-30) | 10 | 5 | 5 | 0 | 5 | 5 | 0 | 5 | 5 | 0 | 0 | 0 |
| v3:F | 24 | 24 | 0 | 0 | 24 | 0 | 0 | 24 | 0 | 0 | 0 | 0 |
| v3:A | 18 | 8 | 10 | 0 | 9 | 9 | 0 | 8 | 10 | 0 | 0 | 1 |
| v3:Am | 15 | 3 | 12 | 0 | 3 | 12 | 0 | 3 | 12 | 0 | 0 | 0 |
| v3:B | 30 | 30 | 0 | 0 | 30 | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| v3:Be | 10 | 10 | 0 | 0 | 10 | 0 | 0 | 10 | 0 | 0 | 0 | 0 |
| v3:C | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |

## Correlation veto (confirm_echo_psr 15, confirm_echo_drop_psr 15), under v5

| group | turns | checks with a psr | checks at or over confirm_echo_psr | checks it changed | turns it changed |
|---|---|---|---|---|---|
| live:call 1 (10-29) | 2 | 2 | 0 | 0 | 0 |
| live:call 2 (10-29) | 8 | 13 | 0 | 0 | 0 |
| live:V1 (10-30) | 5 | 6 | 1 | 1 | 1 |
| live:V2 (10-30) | 4 | 6 | 0 | 0 | 0 |
| live:C6 (10-30) | 10 | 43 | 8 | 0 | 0 |
| v3:F | 24 | 58 | 23 | 20 | 6 |
| v3:A | 18 | 76 | 65 | 27 | 9 |
| v3:Am | 15 | 58 | 49 | 8 | 5 |
| v3:B | 30 | 38 | 0 | 0 | 0 |
| v3:Be | 10 | 0 | 0 | 0 | 0 |
| v3:C | 8 | 27 | 0 | 0 | 0 |

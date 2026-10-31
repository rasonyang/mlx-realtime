# Hold v6 replay: a caller inside the echo is judged without the echo prefix

Issue #8. On the live C6 call a caller started "等一下，我想问问天气" about 4 s into the echo of the bot's
reprompt. The VAD turn began at the echo's onset (input 129.88 s, speech from 130.18 s) and ran to
137.31 s, so the transcript was the echo and the question in one: 35 units, contiguity 0.74 against
the played text. The echo-turn rule (`echo_match_threshold` 0.7) called the whole turn echo and it
got no answer. The partial check had released the hold correctly (`hold_release reason=caller`), so
this was the unheld rule's drop (`transcript echo=true`, `barge_in_outcome echo`).

v6 cuts the echo prefix off a transcript that the rule calls echo (`text.similarity.after_echo`) and
judges the remainder; if it is a caller, the remainder is the turn's transcript.

## Locating the prefix

From the transcript and the played text alone: the leading units covered by the runs of 3 or more
units that are contiguous in the played text (the runs `contiguity` counts, #9), joined across gaps of
fewer than 3 uncovered units (an ASR error inside the echo). The prefix ends after the last such run,
before a gap of 3 uncovered units or more; a transcript whose first 3 units are uncovered has no
prefix. Why this and not the alternatives:

- The hold's last check before the release exists only for held turns, and a turn released by a
  partial check (C6's) gets its transcript from the commit, so the check's text would have to be
  carried to the turn. The unheld rule is where C6 failed, and the transcript is all it has.
- An LCS alignment is order-preserving but scattered: a caller's "我想" would align with our "我"
  and "想" far apart. Units in runs of 3 cannot be scattered, and runs are what #9 already decided
  echo is made of.

## Judging the remainder

- Both: at least `echo_match_min_units` (4) units, and its own contiguity below
  `echo_match_threshold` (0.7).
- Held (a complete-utterance or final check): also `confirm_min_new_units` (2; 1 once nothing plays)
  units that are not ours, by the in-order match the partial checks use. The held turn is released at
  that check with the remainder as its transcript.
- Unheld (the commit's transcript): the remainder replaces the transcript.

The issue asked for `confirm_min_new_units` alone for held turns. That would release a held echo
whose last two words the ASR misheard, which
`test_an_echo_turn_with_two_misheard_units_is_dropped_at_the_final_decision` (#5) keeps dropped, so
held turns need the 4 units too.

## Method

Offline, no gateway and no models: `spikes/live_echo/hold_replay_v6.py` reads the events of every
live capture and of the synthetic v3 captures, as `hold_replay_v5.py` does, and mirrors the actor's
rules. Held turns are walked check by check (v5 against v6); the `transcript` event of every detected
turn is judged by the unheld rule. The replay cannot tell whether the client still played at a held
check, so it takes `confirm_min_new_units` 2 throughout.

    uv run --locked python spikes/live_echo/hold_replay_v6.py > spikes/live_echo/results/hold_v6_replay.md

## Must hold

| requirement | result |
|---|---|
| C6, input 129.88-137.31 s, becomes a caller turn with the remainder "等一下，我想问问天气" | pass: the prefix is 26 units (the reprompt), the remainder 9 units with contiguity 0.00; the transcript becomes "等一下，我想问问天气。" |
| every other C6 echo drop stays an echo drop | pass: the 5 `hold_drop reason=echo` turns (72, 9, 4, 6 and 20 units) cut to an empty remainder, same deciding checks as v5; the unheld echo turn at 22.9 s ("也不会反问你。", 6 units) has an empty remainder and stays echo |
| no new release of a leak, no change for synthetic callers | pass: 0 of 134 held turns change outcome in any capture; of 113 detected-turn transcripts, only C6's changes (the 6 echo transcripts of A and Am stay echo) |
| every echo check of every capture | 30 complete-utterance or final checks that the rule calls echo, all with an empty remainder |

The one turn whose outcome changes is the C6 turn above.

## Unheld turns that become caller turns (`transcript` events)

| capture | item | input s | heard | units | contiguity | remainder | units | contiguity |
|---|---|---|---|---|---|---|---|---|
| C6 (10-30) | ggyf | 129.88-137.31 | 您可能只是暂时没想好问什么，那我先给您讲个关于日常生活。等一下，我想问问天气。 | 35 | 0.74 | 等一下，我想问问天气。 | 9 | 0.00 |

## Held turns whose outcome changes

none

## Outcomes per capture group

| group | held turns | v5 released | v5 dropped | v5 undecided | v6 released | v6 dropped | v6 undecided | held turns whose outcome changed | transcripts of detected turns | transcripts echo (v5) | transcripts echo (v6) | transcripts made caller turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| live:call 1 (10-29) | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| live:call 2 (10-29) | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 7 | 0 | 0 | 0 |
| live:V1 (10-30) | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 0 | 6 | 0 | 0 | 0 |
| live:V2 (10-30) | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| live:C6 (10-30) | 10 | 5 | 5 | 0 | 5 | 5 | 0 | 0 | 6 | 2 | 1 | 1 |
| v3:F | 24 | 24 | 0 | 0 | 24 | 0 | 0 | 0 | 27 | 0 | 0 | 0 |
| v3:A | 18 | 8 | 10 | 0 | 8 | 10 | 0 | 0 | 8 | 4 | 4 | 0 |
| v3:Am | 15 | 3 | 12 | 0 | 3 | 12 | 0 | 0 | 3 | 2 | 2 | 0 |
| v3:B | 30 | 30 | 0 | 0 | 30 | 0 | 0 | 0 | 31 | 0 | 0 | 0 |
| v3:Be | 10 | 10 | 0 | 0 | 10 | 0 | 0 | 0 | 10 | 0 | 0 | 0 |
| v3:C | 8 | 8 | 0 | 0 | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |

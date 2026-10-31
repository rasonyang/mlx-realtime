# Hold v4 replay: the correlation overrules a transcript echo match

Superseded (issue #9): on a live line echo correlated as weakly as callers, so this rule is removed.
The transcript's contiguity replaces it: `hold_v5_replay.md`.

Rule under test (v4): on a complete-utterance check (pause or final) whose GCC-PHAT correlation was
computed (AEC3 in bypass, span long enough) and is below `confirm_echo_psr` (15), the echo-turn
match of the transcript does not stand: the turn is released (at the pause, or at its end with the
final transcript) instead of held or dropped. Trigger case: a live call where the caller asked
"why is the sky blue" over the bot saying those words (6 of 7 units matched, PSR 5.7-8.6, the
question was dropped).

## Method

Offline, no gateway run: `spikes/live_echo/hold_replay_v4.py` reads the v3 measurement rows
(`captures/hold_v3/rows/*-fix.json`: synthetic leaks A and Am, real callers B, Be, C, F, real
ASR) and their `hold_check` events. A held turn changes outcome if

- its final check was `echo` (v3 dropped it) with a computed `psr` < 15, or
- a pause (`utterance`) check with >= 4 units was kept `hold` (only the echo rule holds such a
  check) with a computed `psr` < 15.

Checks with `psr` null (`no_bypass`, `too_short`, ...) and partial checks are unchanged by design.
For a leak, `playing` is whether the capture's events place the check inside a modelled
`play_start` .. `play_end`: a release then sends `speech_started` to a client that is still
playing (AICC flushes its queue: a self-interruption).

## Counts

| group | held turns | v3 echo drops | drops with a computed psr | changed by v4 |
|---|---|---|---|---|
| leaks A | 18 | 10 | 1 (psr 13.45) | 1 |
| leaks Am | 15 | 10 (+1 noise, 1 echo_signal) | 0 | 0 |
| callers B, Be, C, F | 72 | 0 | 0 | 0 |

- Leaks: 1 of 33 held turns (A22, "三个部分。", 4 units, 0 unmatched, final PSR 13.45, just under 15)
  would be released instead of dropped; the client was still playing at that check, so it would be
  a self-interruption. The other 19 leak echo drops had no correlation (`no_bypass`), so the
  transcript rule decides as before. v3 had 3 of 36 (default model) and 2 of 18 (mild model) self-interrupted bursts; A22 would be one more
  on the default model if it was among those A rows.
- Real callers: 0 of 72 held turns change outcome. None of them was dropped in v3 or had a
  4-unit echo match to begin with, so this measurement cannot show the benefit; the live call
  that motivated the rule is the evidence for it, and the integration tests cover the mechanism.

Caveat: A22 shows the trade-off is real but small: a synthetic leak that the ASR hears as >= 4 of our
own words and whose PSR is between roughly 10 and 15 now interrupts. Lowering `confirm_echo_psr`
(10 held more leaks in v3) would narrow that window at the cost of a later release for callers
with a mid-range PSR.

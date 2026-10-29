# Hold: does a correlation of the input against our played reference separate leaks from callers?

Fix-condition captures of `results/hold.md`; ground truth from the rows (burst span for A/Am, injected clip span for B/Be/C); input-timeline ms. A1 bursts (answer already finished, nothing playing) are excluded. The leak model is a synthetic assumption (`bursts.py`); real phone leaks may differ.

Lag reference (A/Am): the PHAT peak over the whole burst (PSR >= 8); the burst's nominal delay is not a lag in the reference file (the playout model's shift differs per burst), so it cannot be used. Be: the last valid logged lag before the onset.

## 1. Counts and canceller state at the onset

| group | n | bypass on | bypass off |
|---|---|---|---|
| A leak (default model) | 24 | 20 | 4 |
| Am leak (mild model) | 12 | 11 | 1 |
| B caller, no echo line | 30 | 30 | 0 |
| Be caller, steady-echo line | 10 | 0 | 10 |
| C short caller | 22 | 22 | 0 |

## 2. Gateway's logged `gcc` windows overlapping the speech span

A logged window is 2 s ending at `t` (logged 0.3 s later); `mostly` = at least 1 s of it is speech. PSR cells: p10 / p50 / p90.

| group | windows | n | valid (PSR>=8) | PSR p10/p50/p90 | lag consistent |
|---|---|---|---|---|---|
| A leak (default model) | overlapping | 126 | 113 | 8.0 / 29.4 / 49.5 | 102/110 |
| A leak (default model) | mostly speech | 79 | 78 | 22.7 / 37.4 / 54.6 | 73/75 |
| Am leak (mild model) | overlapping | 67 | 62 | 10.8 / 36.6 / 71.0 | 57/62 |
| Am leak (mild model) | mostly speech | 36 | 36 | 26.4 / 49.5 / 80.2 | 34/36 |
| B caller, no echo line | overlapping | 97 | 32 | 6.0 / 7.3 / 9.1 | - |
| B caller, no echo line | mostly speech | 38 | 9 | 5.8 / 6.8 / 8.4 | - |
| Be caller, steady-echo line | overlapping | 34 | 34 | 28.7 / 51.0 / 74.2 | 34/34 |
| Be caller, steady-echo line | mostly speech | 16 | 16 | 25.1 / 41.8 / 58.5 | 16/16 |
| C short caller | overlapping | 115 | 38 | 5.6 / 7.2 / 9.9 | - |
| C short caller | mostly speech | 14 | 1 | 5.7 / 6.3 / 7.7 | - |

Median delay from the onset to the first logged window that is mostly speech: A leak (default model) 1253 ms; Am leak (mild model) 1273 ms; B caller, no echo line 1220 ms; Be caller, steady-echo line 1150 ms; C short caller 1350 ms (the window ends there, plus 0.3 s of reference wait).

## 3. Own short-span correlation (input vs reference, lags 0-1000 ms)

Band 200-3600 Hz. PSR = PHAT peak against the other lags (as the gateway's); NCC = best normalised cross-correlation over the lags. Cells: p10 / p50 / p90. `lag ok` = peak within 5 ms of the lag reference (A/Am vs whole-burst peak; Be vs pre-onset logged lag).

### Span onset to onset + 400 ms

| group | bypass | n | PSR | NCC x100 | lag ok |
|---|---|---|---|---|---|
| A leak (default model) | on | 20 | 7.0 / 12.3 / 27.1 | 11.6 / 55.5 / 66.9 | 14/19 |
| A leak (default model) | off | 4 | 20.9 / 27.2 / 38.4 | 42.0 / 57.5 / 71.2 | 4/4 |
| Am leak (mild model) | on | 11 | 11.6 / 26.3 / 43.3 | 43.0 / 59.6 / 78.3 | 11/11 |
| Am leak (mild model) | off | 1 | 23.3 / 23.3 / 23.3 | 71.2 / 71.2 / 71.2 | 1/1 |
| B caller, no echo line | on | 30 | 6.0 / 7.5 / 9.4 | 10.9 / 16.1 / 23.1 | - |
| Be caller, steady-echo line | off | 10 | 8.0 / 13.1 / 24.2 | 18.2 / 21.7 / 51.4 | 7/10 |
| C short caller | on | 22 | 6.6 / 7.2 / 8.4 | 9.6 / 14.7 / 24.1 | - |

### Span onset to onset + 700 ms

| group | bypass | n | PSR | NCC x100 | lag ok |
|---|---|---|---|---|---|
| A leak (default model) | on | 20 | 8.4 / 23.3 / 40.6 | 10.8 / 54.5 / 66.0 | 16/19 |
| A leak (default model) | off | 4 | 28.5 / 30.5 / 36.2 | 38.3 / 57.8 / 71.9 | 4/4 |
| Am leak (mild model) | on | 11 | 13.8 / 34.6 / 59.3 | 40.0 / 56.6 / 79.3 | 11/11 |
| Am leak (mild model) | off | 1 | 31.0 / 31.0 / 31.0 | 69.7 / 69.7 / 69.7 | 1/1 |
| B caller, no echo line | on | 30 | 5.9 / 7.3 / 9.1 | 8.5 / 12.4 / 19.5 | - |
| Be caller, steady-echo line | off | 10 | 10.0 / 16.7 / 51.1 | 14.6 / 25.9 / 37.4 | 9/10 |
| C short caller | on | 22 | 6.4 / 7.1 / 8.4 | 8.6 / 13.8 / 22.7 | - |

### Span onset to onset + 1000 ms

| group | bypass | n | PSR | NCC x100 | lag ok |
|---|---|---|---|---|---|
| A leak (default model) | on | 20 | 9.2 / 27.1 / 47.5 | 25.2 / 53.9 / 66.0 | 19/19 |
| A leak (default model) | off | 4 | 34.3 / 36.4 / 43.5 | 37.8 / 54.7 / 69.2 | 4/4 |
| Am leak (mild model) | on | 11 | 17.2 / 37.4 / 66.8 | 37.4 / 58.8 / 79.2 | 11/11 |
| Am leak (mild model) | off | 1 | 38.3 / 38.3 / 38.3 | 70.4 / 70.4 / 70.4 | 1/1 |
| B caller, no echo line | on | 30 | 6.1 / 6.8 / 8.7 | 7.5 / 8.8 / 13.0 | - |
| Be caller, steady-echo line | off | 10 | 10.0 / 22.7 / 46.4 | 13.1 / 25.6 / 35.4 | 10/10 |
| C short caller | on | 22 | 6.2 / 7.2 / 8.4 | 6.7 / 13.4 / 22.7 | - |

## 4. Threshold separation (share flagged as 'our own audio')

Flagged = metric >= threshold. Leaks should be flagged (high), callers should not. Spans clipped at the end of the speech (C is 0.4-0.5 s long). A/Am/B/C columns are bypass-on cases (the 5 bypass-off A/Am leaks are in section 5); `Be` is the steady-echo line (canceller active) where no leak was injected. The `aec3` columns correlate the canceller's output instead of the raw input.

### PSR, span 400 ms

| PSR >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 4 | 20/20 | 11/11 | 30/30 | 22/22 | 10/10 | 20/20 | 10/10 |
| 6 | 19/20 | 11/11 | 27/30 | 22/22 | 10/10 | 19/20 | 10/10 |
| 8 | 16/20 | 11/11 | 12/30 | 6/22 | 9/10 | 16/20 | 6/10 |
| 10 | 13/20 | 10/11 | 2/30 | 0/22 | 7/10 | 14/20 | 2/10 |
| 12 | 10/20 | 9/11 | 2/30 | 0/22 | 6/10 | 12/20 | 1/10 |
| 15 | 8/20 | 8/11 | 0/30 | 0/22 | 4/10 | 8/20 | 1/10 |

### PSR, span 700 ms

| PSR >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 4 | 20/20 | 11/11 | 30/30 | 22/22 | 10/10 | 20/20 | 10/10 |
| 6 | 20/20 | 11/11 | 26/30 | 22/22 | 10/10 | 19/20 | 10/10 |
| 8 | 18/20 | 11/11 | 7/30 | 5/22 | 10/10 | 17/20 | 7/10 |
| 10 | 16/20 | 10/11 | 0/30 | 0/22 | 9/10 | 15/20 | 1/10 |
| 12 | 15/20 | 10/11 | 0/30 | 0/22 | 8/10 | 15/20 | 0/10 |
| 15 | 14/20 | 9/11 | 0/30 | 0/22 | 5/10 | 14/20 | 0/10 |

### PSR, span 1000 ms

| PSR >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 4 | 20/20 | 11/11 | 30/30 | 22/22 | 10/10 | 20/20 | 10/10 |
| 6 | 20/20 | 11/11 | 28/30 | 21/22 | 10/10 | 20/20 | 10/10 |
| 8 | 18/20 | 11/11 | 9/30 | 5/22 | 10/10 | 19/20 | 4/10 |
| 10 | 17/20 | 10/11 | 0/30 | 0/22 | 9/10 | 17/20 | 0/10 |
| 12 | 17/20 | 10/11 | 0/30 | 0/22 | 8/10 | 17/20 | 0/10 |
| 15 | 16/20 | 10/11 | 0/30 | 0/22 | 6/10 | 16/20 | 0/10 |

### NCC, span 400 ms

| NCC >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 0.05 | 20/20 | 11/11 | 30/30 | 22/22 | 10/10 | 20/20 | 10/10 |
| 0.08 | 19/20 | 11/11 | 30/30 | 20/22 | 10/10 | 19/20 | 10/10 |
| 0.1 | 19/20 | 11/11 | 28/30 | 19/22 | 10/10 | 18/20 | 9/10 |
| 0.15 | 17/20 | 11/11 | 16/30 | 10/22 | 10/10 | 17/20 | 7/10 |
| 0.2 | 17/20 | 11/11 | 7/30 | 5/22 | 7/10 | 17/20 | 3/10 |
| 0.3 | 17/20 | 10/11 | 0/30 | 0/22 | 3/10 | 16/20 | 1/10 |

### NCC, span 700 ms

| NCC >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 0.05 | 20/20 | 11/11 | 30/30 | 21/22 | 10/10 | 20/20 | 10/10 |
| 0.08 | 19/20 | 11/11 | 28/30 | 20/22 | 10/10 | 19/20 | 9/10 |
| 0.1 | 18/20 | 11/11 | 24/30 | 17/22 | 10/10 | 18/20 | 8/10 |
| 0.15 | 17/20 | 10/11 | 5/30 | 7/22 | 9/10 | 17/20 | 2/10 |
| 0.2 | 17/20 | 10/11 | 3/30 | 5/22 | 6/10 | 17/20 | 0/10 |
| 0.3 | 17/20 | 10/11 | 0/30 | 0/22 | 4/10 | 17/20 | 0/10 |

### NCC, span 1000 ms

| NCC >= | A leaks | Am leaks | B callers | C callers | Be callers (input) | A leaks (aec3, bypass on) | Be callers (aec3) |
|---|---|---|---|---|---|---|---|
| 0.05 | 20/20 | 11/11 | 30/30 | 22/22 | 10/10 | 20/20 | 9/10 |
| 0.08 | 19/20 | 11/11 | 21/30 | 19/22 | 10/10 | 19/20 | 9/10 |
| 0.1 | 19/20 | 11/11 | 10/30 | 17/22 | 9/10 | 19/20 | 6/10 |
| 0.15 | 19/20 | 10/11 | 2/30 | 6/22 | 8/10 | 19/20 | 0/10 |
| 0.2 | 18/20 | 10/11 | 0/30 | 4/22 | 7/10 | 18/20 | 0/10 |
| 0.3 | 17/20 | 10/11 | 0/30 | 0/22 | 2/10 | 18/20 | 0/10 |

## 5. A/Am leaks with the canceller active (bypass off at the onset)

| scenario | span ms | n | PSR p10/p50/p90 |
|---|---|---|---|
| A | 400 | 4 | 20.9 / 27.2 / 38.4 |
| A | 700 | 4 | 28.5 / 30.5 / 36.2 |
| A | 1000 | 4 | 34.3 / 36.4 / 43.5 |
| Am | 400 | 1 | 23.3 / 23.3 / 23.3 |
| Am | 700 | 1 | 31.0 / 31.0 / 31.0 |
| Am | 1000 | 1 | 38.3 / 38.3 / 38.3 |

## 6. Reading and caveats

- On bypass lines (no steady echo) a PSR >= 10 on the 700-1000 ms span flags most leaks and no caller (B, C); at 400 ms
  the leak PSR is lower and the callers' tail rises (see section 4). The misses are bursts whose lag is not found
  (the same bursts have a stray lag in the gateway's own windows).
- On the steady-echo line (Be) the raw input correlates with the reference whatever the speaker says (the steady
  echo itself), so the raw correlation cannot separate callers; only the canceller's output (`aec3`) is a candidate
  there, and no leak was injected on such a line, so that case is untested.
- The burst model (`bursts.py`: band-limited, clipped, gain-modulated copies of the played audio at a fixed delay)
  is a synthetic assumption built from the playout reference. A real leak is a copy by construction; a real phone
  leak may be less correlated, so these separations are optimistic for leaks.
- Callers are clean recorded clips with no room echo of the bot beyond the Be steady echo. A caller talking in a
  room with strong bot echo would correlate like Be.
- Small n (11-24 leaks, 22-30 callers per cell): treat shares as indications.

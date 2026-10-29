# Validation on simulated echo

analyze.py on captures of the real gateway (aec.capture_dir) in `tests/phone.py`'s simulated AICC call: 24 kHz PCM both ways, AICC's 24 -> 8 kHz downsampler, G.711 mu-law round trip, a synthetic or ITU-T G.168 echo path after a bulk delay, AICC's 8 -> 24 kHz uplink, three answers heard in silence and a fourth cut by a caller. Regenerate with `simulate.py` and `validate.py` (README).

Pass: delay within 2.0 ms and ERL within 1.5 dB (echo lines), class `none` with the bypass engaged and not flapping (no-echo line). The first answer is reported but not judged: the playout model learns AICC's tick phase during it and moves by up to a tick, so its modelled start is not where the rest of the answer sits.

## Summary

| scenario | expected | measured | result |
|---|---|---|---|
| erl10_d100 | ERL 10.0 dB (effective on this speech: 9.3 dB), delay 100 ms (dominant tap 101.54 ms) | ERL 9.7 dB (error 0.3), delay error up to 0.61 ms over answers 2-4 | PASS |
| erl15_d180 | ERL 15.0 dB (effective on this speech: 15.7 dB), delay 180 ms (dominant tap 180.79 ms) | ERL 16.0 dB (error 0.3), delay error up to 0.27 ms over answers 2-4 | PASS |
| g168_m4 | ERL 12.0 dB (effective on this speech: 10.1 dB), delay 120 ms (dominant tap 122.79 ms) | ERL 10.5 dB (error 0.4), delay error up to 0.71 ms over answers 2-4 | PASS |
| no_echo | no echo, bypass engaged | class none, bypass 1x (first 2.48 s after the first answer starts), flapping False | PASS |

## erl10_d100

| answer | model error (true - modelled start, ms) | expected delay after model (ms) | measured (ms) | error (ms) | start / end of answer (ms) | filter's dominant tap (ms) |
|---|---|---|---|---|---|---|
| 1 | -9.7 | 91.84 | 91.37 | -0.47 | 91.89 / 91.36 | 91.88 |
| 2 | -10.9 | 90.64 | 90.37 | -0.27 | 90.24 / 90.85 | 90.38 |
| 3 | -10.6 | 90.94 | 90.99 | 0.05 | 90.72 / 90.99 | 91.25 |
| 4 | -29.2 | 72.34 | 71.73 | -0.61 | 72.23 / 71.73 | 72.25 |

- ERL: configured 10.0 dB on the downsampled audio, which is 9.3 dB when the path's ripples meet this speech's spectrum (and AICC's downsampler); measured 9.7 dB; dispersion (90 % of the filter's energy) 3.4 ms
- AEC3 ERLE: converges to 10 dB 0.47 s after the first answer starts, steady-state median 35.7 dB
- Speex ERLE: 2.97 s, steady-state median 44.6 dB; first aligned 1.97 s after the first answer starts
- the gateway's GCC-PHAT against the independent delay: median |difference| 0.24 ms
- drift flag: True; self-interruptions: 0
- caller barge-ins: [(697, 697)] (latency from the estimated caller onset, excess over the guard, ms)

## erl15_d180

| answer | model error (true - modelled start, ms) | expected delay after model (ms) | measured (ms) | error (ms) | start / end of answer (ms) | filter's dominant tap (ms) |
|---|---|---|---|---|---|---|
| 1 | -11.2 | 169.59 | 166.84 | -2.75 | 166.84 / 166.85 | 166.88 |
| 2 | -13.3 | 167.49 | 167.53 | 0.03 | 167.51 / 167.77 | 167.50 |
| 3 | -13.8 | 166.99 | 167.10 | 0.11 | 167.09 / 167.10 | 167.12 |
| 4 | 5.5 | 186.29 | 186.02 | -0.27 | 186.35 / 186.03 | 186.00 |

- ERL: configured 15.0 dB on the downsampled audio, which is 15.7 dB when the path's ripples meet this speech's spectrum (and AICC's downsampler); measured 16.0 dB; dispersion (90 % of the filter's energy) 4.2 ms
- AEC3 ERLE: converges to 10 dB 1.49 s after the first answer starts, steady-state median 29.5 dB
- Speex ERLE: 2.49 s, steady-state median 44.0 dB; first aligned 1.99 s after the first answer starts
- the gateway's GCC-PHAT against the independent delay: median |difference| 0.03 ms
- drift flag: True; self-interruptions: 0
- caller barge-ins: [(798, 798)] (latency from the estimated caller onset, excess over the guard, ms)

## g168_m4

G.168 model m4 (coefficients not committed), delay 120 ms before the path.

| answer | model error (true - modelled start, ms) | expected delay after model (ms) | measured (ms) | error (ms) | start / end of answer (ms) | filter's dominant tap (ms) |
|---|---|---|---|---|---|---|
| 1 | -17.9 | 104.89 | 102.61 | -2.28 | 102.89 / 102.61 | 102.62 |
| 2 | -20.2 | 102.59 | 102.75 | 0.16 | 102.63 / 102.75 | 102.62 |
| 3 | 0.1 | 122.89 | 122.88 | -0.02 | 122.88 / 122.88 | 122.88 |
| 4 | -13.0 | 109.79 | 110.50 | 0.71 | 111.28 / 109.72 | 109.75 |

- ERL: configured 12.0 dB on the downsampled audio, which is 10.1 dB when the path's ripples meet this speech's spectrum (and AICC's downsampler); measured 10.5 dB; dispersion (90 % of the filter's energy) 7.0 ms
- AEC3 ERLE: converges to 10 dB 1.48 s after the first answer starts, steady-state median 35.0 dB
- Speex ERLE: 2.48 s, steady-state median 47.4 dB; first aligned 1.98 s after the first answer starts
- the gateway's GCC-PHAT against the independent delay: median |difference| 0.05 ms
- drift flag: True; self-interruptions: 0
- caller barge-ins: [(755, 755)] (latency from the estimated caller onset, excess over the guard, ms)

## no_echo

- line class: none; bypass engaged 1x, first 2.48 s after the first answer starts (expected about 1.5 s plus the first estimator window); flapping: False
- self-interruptions: 0
- caller barge-ins: [(698, 698)] (latency, excess over the guard, ms)

## Residual-echo burst detector on injected leaks

The no-echo capture (canceller bypassed) with delayed copies of the reference added to its input; the detector must find these and nothing else.

| injected at (s) | duration (s) | lag (ms) | level vs reference (dB) |
|---|---|---|---|
| 15 | 2 | 350 | -12 |
| 27 | 1.5 | 500 | -30 |

| found at (s) | duration (s) | lag (ms) | level vs reference (dB) | over floor (dB) | position |
|---|---|---|---|---|---|
| 14.95 | 1.75 | 350 | -12.0 | 33.6 | start of a response |
| 26.95 | 1.00 | 500 | -29.9 | 16.8 | start of a response |

# Live echo verification (spikes/live_echo)

Does our echo cancellation work on real AICC calls? D17's design and every number behind it come
from simulated telephone echo (M0 S11, G7). This spike is the tooling to measure it on real calls:
a capture in the gateway, an analysis that does not reuse the gateway's estimators, and a
validation of that analysis against simulated echo with known ground truth
(`results/simulated.md`). The calls themselves are placed by a human.

What the calls look like: AICC speaks `audio/pcm` at 24 kHz both ways (its profile has
`AcceptsG711: false`), but its trunk leg is G.711 PCMU at 8 kHz, so the uplink carries at most
4 kHz plus mu-law noise. There is no PSTN hybrid: all echo is acoustic (softphones over VoIP), so
expect weak or nonlinear residual echo and jitter when a phone's or Zoiper's own AEC is on. AICC's
TX queue takes our audio up to 10 s ahead of real time and a 20 ms RTP pacer plays it out; the
reference follows the playout model, so a key check is whether the measured delay stays constant.

## 1. Capture

Set `aec.capture_dir` (default `null`, off; `aec.capture_max_s`, default 900, caps the input per
session). For example with an environment override:

```sh
MLXRT__AEC__CAPTURE_DIR="$PWD/spikes/live_echo/captures/live" make dev
```

Every realtime session whose cancellers run then writes
`<capture_dir>/<UTC start to the ms>_<session id>/` (it sorts by time; transcription sessions have
no canceller and write nothing):

| file | content |
|---|---|
| `input.wav` | the input as received, before cancelling |
| `ref_aec3.wav` | AEC3's modelled reference (before the bypass silences it), placed 60 ms early plus the guard's shift |
| `aec3.wav` | AEC3's output (what Silero and Smart Turn hear) |
| `ref_speex.wav` | the ASR path's reference at its own shift (zeros until Speex is aligned) |
| `asr.wav` | the ASR path's output (the input where the ASR hears it as received) |
| `events.jsonl` | `t` in input-timeline ms plus `kind`: `play_start` / `play_end` (modelled), `flush`, `truncate`, `reanchor`, `gcc` (every GCC-PHAT window: lag, PSR, valid, confirmed, the shift in force), `realign` (aec3 and speex), `bypass`, `speech_started` (with `audio_start_ms`, `barge_in`, and `held_ms` when `barge_in.confirm_while_playing` held it back, so onset to `speech_started` includes the hold), `hold` (with `playing`: held because the client plays, else only to keep order) / `hold_end` / `hold_check` (`check` partial / utterance / final, text, units, unmatched, `psr` of the raw input against our audio or null with `psr_why` (disabled, no_canceller, no_bypass, too_short, silent_reference, error), verdict incl. `echo_signal` / `echo_signal_drop`, audio span) / `hold_release` (`reason`: caller, pause, turn_end, check_failed, playback_ended, order_only) / `hold_drop` (`reason` echo, noise or echo_signal) / `hold_cancel` (a turn held back while we play), `speech_stopped` (`audio_end_ms`), `barge_in_outcome` (real / empty / echo), `echo_turn`, `transcript` (text, `audio_start_ms` / `audio_end_ms`, `units`, `echo_score`, `played_text` and `played_spans`: what the echo matcher compared against), `assistant_text` (each response's segments: text, `start_ms` and `audio_ms` on the item's audio timeline; with the item's `play_start` it places them on the input timeline) |
| `meta.json` | session id, wall-clock start (UTC, ms), first-audio time, formats, the `aec` and `barge_in` configuration, the final `echo summary` |

All WAVs are mono int16, 16 kHz, sample-aligned to each other: sample i of every file is input
sample i. The files contain the caller's voice and transcripts: dev use only, never committed
(`*.wav` is git-ignored, and so is `captures/`; real calls go to `captures/live/`, which the simulation scripts never touch, simulated ones to `captures/simulated/`).
Capture never blocks the event loop or fails a session: a writer thread does the I/O, memory is
bounded, and the first I/O error (or a writer that falls behind) logs one warning and stops the
capture.

The session start in `meta.json` is the key to AICC's call: calls are placed one at a time, and
our session start is about AICC's "ai call bridged" log line.

## 2. Analyse

```sh
cd spikes/live_echo
uv sync --locked
uv run --locked python analyze.py <capture dir>... [--aicc-recording <stereo wav>...]
```

It writes `report.md` and `report.json` into each capture directory and prints the verdicts.
`--aicc-recording` takes one AICC stereo recording per capture directory (same order). AICC's
recordings are at `http://127.0.0.1:8888/buckets/aicc-recordings/YYYY/MM/DD/<aiccCallId>.wav`
(UTC date), probably 8 kHz; left = the caller as the switch receives it from the trunk (echo
included), right = the bot as the switch sends it.

Everything is band-limited to 300-3400 Hz and handled at 8 kHz; times are input-timeline ms.

- **Line class.** `none` if fewer than 10 % of the 2 s reference-active windows show a correlation
  peak (z >= 6), `weak` if fewer than 50 % or ERL >= 25 dB, else `echo`. A weak or nonlinear
  residual is reported with its numbers (valid-window fraction, the linear fit's gain in dB) and
  never called "no echo" on a single missing window.
- **Bulk delay.** Plain cross-correlation of the pre-emphasized reference and input over -100 ...
  +700 ms in 2 s windows every 1 s (not GCC-PHAT, not the gateway's code). Reported as the lag
  against the reference file and "after the playout model" (lag + the gateway's shift - the 60 ms
  lead). Per answer: the first and last third of its windows, the drift between them, and the
  spread of the answers' medians; **drift above 3 ms is flagged** (the playout model diverges from
  AICC's pacer, or clock drift).
- **Echo path.** Per answer, at that answer's own delay (the model places every run of audio at its
  own tick phase), a frequency-domain least-squares filter over echo-only blocks (blocks the
  filter does not explain are dropped). ERL = reference energy over predicted-echo energy;
  dispersion = the shortest span holding 90 % of the filter's energy (noise floor removed); ERL
  over time in 5 s blocks.
- **Caller speech and echo-only stretches.** A frame is caller speech when the input less a
  per-second echo estimate is 12 dB over the line's noise floor and at least -9 dB of the input's
  energy, in runs of at least 300 ms, with the ASR track (Speex's output) 9 dB over its own floor
  in 30 % of the run. The onset is the run's first frame (accuracy about a few hundred ms; early
  rather than late). Echo-only = the reference played, nothing else.
- **ERLE.** Input energy over output energy per 0.5 s window of echo-only frames whose predicted
  echo is 6 dB over the noise floor, for AEC3 and Speex, in the 300-3400 Hz band. Convergence = the
  first three consecutive windows at 10 dB, counted from the first answer's modelled start. Speex's
  figure includes SpeexDSP's noise suppression.
- **After a barge-in.** AICC waits out its 800 ms guard before it clears its TX queue, so up to
  about 800 ms of our audio keeps playing (and echoing) after our `speech_started`. For 1.1 s after
  every `flush` the report treats AICC as still playing: no caller speech starts there and a
  `speech_started` there is not a self-interruption.
- **Self-interruptions.** A `speech_started` with no caller speech in the 3 s before it (and not in
  that window). Barge-in latency = `speech_started` less the caller's onset, less what is left of
  the 2000 ms onset guard.
- **AICC recording.** Left is analysed against right as a line of its own: the switch-side echo
  delay, ERL and dispersion. Our end-to-end delay (after the model) is then split into that
  switch-side path plus the remainder (AICC's TX queue, the switch-to-AICC hop and the playout
  model's error). It also checks the bot channel after each barge-in for AICC's known pacer stall
  (no RTP until the next frame arrives): a shift of the audio against our modelled playout
  (> 15 ms, not present before the first barge-in; noted when a re-anchor preceded it) or exact
  silence (> 100 ms) where our reference says audio plays. Each flag has its recording time, to
  report to AICC.
- **Residual-echo bursts (while the canceller is bypassed).** A phone's own AEC can leak echo past
  a bypassed canceller in bursts. In 0.5 s windows with an active reference inside the bypassed
  stretches, the input is correlated with the reference at 50-700 ms and compared with its
  correlation to the reference of 12 other times (a shuffled baseline: z >= 6 on the median and
  MAD, ncc >= 0.1), with energy 6 dB over the noise floor. Consecutive windows merge into a burst
  (start and end are good to about 0.25 s). The report gives the count per minute of bot speech,
  and for each burst its duration, lag, level (dBFS, over the floor, against the reference), its
  position (start of a response, i.e. within 2.5 s of its modelled start; mid-answer; tail, the
  last 1.5 s; outside) and whether it triggered `speech_started`, a barge-in and with which outcome.
- **Verdict.** Pass criteria: zero self-interruptions in echo-only stretches; every caller
  barge-in's `speech_started` within 1.5 s of onset past the 2 s guard; median AEC3 ERLE >= 10 dB
  on lines with echo; bypass engaged and not flapping on echo-free lines; measured delay and
  dispersion inside the configured ranges (GCC keep [5, 450] ms, Speex keep [0, 44] ms and its
  64 ms filter); plus the drift check. Each is PASS, FAIL or N/A (not applicable to the line's
  class).

## 3. Validate on simulated echo

```sh
# from the repository root (it needs the gateway and tests/phone.py)
uv run --locked python spikes/live_echo/simulate.py          # ~1 min per scenario, real time
cd spikes/live_echo && uv run --locked python validate.py captures/simulated/*/*
```

`simulate.py` runs the real gateway with capture on against the simulated AICC call
(`tests/phone.py`) and writes `ground_truth.json` beside each capture. `validate.py` compares the
analysis with it and writes `results/simulated.md` and `.json` (it also injects leaks into the
no-echo capture to test the burst detector). `selftest_recording.py <capture dir>` builds a
synthetic AICC stereo recording with known switch-side delay and ERL and injected pacer stalls, and
checks `--aicc-recording` (`results/recording_selftest.md`). The G.168 scenario runs only if the
git-ignored coefficients are present (`spikes/data/echo/g168_paths.json`).

## 4. Replaying echo rules on captures

`replay.py` re-runs the gateway's echo matcher (`mlx_realtime.text.similarity`) on every captured
turn, in the gateway's environment, so a rule can be evaluated offline before it is built:

```sh
uv run --locked python spikes/live_echo/replay.py <capture dir>... [--threshold 0.7] [--min-units 4] [--rebuild]
```

It prints each turn's transcript, the score and verdict the gateway recorded, the replayed ones,
and the turn's barge-in outcome. By default it scores against the recorded `played_text`;
`--rebuild` rebuilds the played text from `assistant_text` and `play_start` / `play_end`, so the
window (`--window-ms`, `--tail-ms`) can vary too. `replay(events, ...)` returns the rows for other
scripts: an ASR-confirmation rule (a barge-in is confirmed only if the transcript of its audio
does not match what was playing) is this function plus the burst list from the report.

## 5. The calls

The owner places them through AICC (AICC bot: a 20-30 s answer per caller question, never three in
a row; each answer is triggered by a short question; stereo recording on). Start the gateway with
`aec.capture_dir` set before the first call.

| call | softphone | script |
|---|---|---|
| C1 | Zoiper, handset | ask, stay silent through the answer, three times |
| C2 | Zoiper, speakerphone | as C1 |
| C2b | Mac Telephone app, speakers | as C1 |
| C3 | Zoiper, handset | barge in at about 3 s, 6 s and 10 s into three answers |
| C4 | Zoiper, speakerphone | as C3 |
| C4b | Mac Telephone app, speakers | as C3 |
| C5 | Mac Telephone app, headset (no echo) | barge in three times |

Per call collect:

- AICC: the call UUID, the start time (UTC), the phone and mode, the softphone's own AEC setting
  (on, off or unknown), the stereo recording, AICC's interrupt/truncate log lines (and its "ai call
  bridged" line), the Realtime `session.id`.
- Ours: the capture directory and the gateway log's `echo summary` line.

Then `analyze.py <capture dir> --aicc-recording <the call's wav>` for each, and keep the reports.

## 5. Measuring the hold (issue #5)

`bursts.py` measures `barge_in.confirm_while_playing` against the baseline (the same gateway with it
off) on the simulated AICC call, in real time, with the real Qwen3-ASR and Qwen3-TTS (never the
LLM: answers are verbatim lines, `create_response` is off). Residual-echo bursts of what the client
really played (316 / 506 ms late, -5 / -20 / -35 dB, 1.5-2.5 s, nonlinear: `hold_lib.py` documents the
model), real barge-ins (FLEURS zh / en, ASCEND mixed) on lines with and without steady echo, short
utterances and noise over playback, and the cost of the checks.

```sh
# from the repository root; about 95 min of real time; starts and stops the ASR and TTS backends
uv run --locked python spikes/live_echo/bursts.py run          # --only A,Am,B,Be,E0,C,D   --smoke   (Am is the mild leak model)
uv run --locked python spikes/live_echo/bursts.py report       # results/hold.md and hold.json
```

Raw rows and captures go to `captures/hold/` (git-ignored; finished jobs are skipped on a rerun, so
an interrupted run resumes). The conditions alternate chunk by chunk, so drift hits both. The datasets
come from `spikes/data` (git-ignored; the main checkout's when this is a worktree, or
`MLXRT_SPIKES_DATA`). Results: `results/hold.md` (tables, setup, what-ifs; the findings on top come from `results/hold_findings.md`, written by hand) and `results/hold.json` (every trial's row with its capture events). `tests/phone.py` has a hook (`_injected`) for the bursts and sounds.

The v3 re-run (`HOLD_DIR=hold_v3 uv run --locked python spikes/live_echo/bursts.py run --only F,A,Am,B,Be,C --conditions fix`, then `uv run --locked python spikes/live_echo/hold_report_v3.py`) writes `results/hold_v3.md` / `.json`; its baseline rows are the first run's. F (double talk with a leak) needs both conditions: run `--only F` without `--conditions`.

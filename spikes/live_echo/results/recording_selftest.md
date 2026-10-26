# AICC recording self-test

Synthetic stereo recording from `20261026T074938292Z_sess_1ku2zmsdny08ppf7`: switch-side path 40 ms, ERL 12 dB; faults injected: 250 ms of silence in the bot channel while an answer plays (about 27 s into the capture), and a 40 ms stall one second after a barge-in (a flush event added at 20 s of the capture).

## Clean

- switch-side delay: 40.1 ms (expected 40); ERL 14.6 dB (14.5 on this speech; configured 12); dispersion 1.8 ms
- end-to-end after the model 87.9 ms = switch side + remainder 47.8 ms
- bot channel aligned with our reference at 2500.0 ms (built with 2500)
- flags: []
- injected: none

## With faults

- switch-side delay: 40.1 ms (expected 40); ERL 14.6 dB (14.5 on this speech; configured 12); dispersion 1.8 ms
- end-to-end after the model 87.9 ms = switch side + remainder 47.8 ms
- bot channel aligned with our reference at 2500.0 ms (built with 2500)
- flags: [{"kind": "shift_after_barge_in", "recording_s": 23.0, "capture_ms": 20500.0, "shift_ms": 40.0, "reanchored_before": false, "barge_in_recording_s": 22.5, "last_recording_s": 27.0, "windows": 3}, {"kind": "silence_while_playing", "recording_s": 27.66, "capture_ms": 25160.0, "duration_ms": 250.0, "last_recording_s": 27.66, "windows": 1}, {"kind": "shift_after_barge_in", "recording_s": 28.0, "capture_ms": 25500.0, "shift_ms": 40.0, "reanchored_before": false, "barge_in_recording_s": 22.5, "last_recording_s": 44.0, "windows": 15}]
- injected: {'gap_recording_s': 27.62, 'stall_after_s': 22.5}

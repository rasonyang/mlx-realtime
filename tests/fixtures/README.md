# Test fixtures

## Speech (`speech/`)

Two utterances from FLEURS, used by the VAD, Smart Turn and speech-turn tests and by the backend
tests. Both are 16 kHz, mono, 16-bit PCM WAV: the gateway's canonical input format.

| File | Text | Source utterance | Speech (Silero, this file) |
|---|---|---|---|
| `zh_dinosaurs.wav` (3.96 s) | 但鸟类还是有很多地方跟恐龙相似。 | FLEURS `cmn_hans_cn`, test, `11432051235656569325.wav` (M0 set: `zh_001`) | 0.32–3.66 s |
| `en_dinosaurs.wav` (4.23 s) | Many people don't think about them as dinosaurs because they have feathers and can fly. | FLEURS `en_us`, test, `10233995782544396174.wav` (M0 set: `en_001`) | 0.24–3.92 s |

**Source.** FLEURS, <https://huggingface.co/datasets/google/fleurs>, revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Conneau et al., *FLEURS: Few-shot Learning Evaluation
of Universal Representations of Speech*, arXiv:2205.12446.

**License.** Creative Commons Attribution 4.0 International (CC-BY-4.0),
<https://creativecommons.org/licenses/by/4.0/>. These files are adapted from the originals.

**Changes.**

- Converted to 16 kHz mono PCM16 and levelled to −26 dBov active speech level (ITU-T P.56
  measure, peak-limited at −1 dBFS) by the M0 data spike (`spikes/datasets/a2_data.py`; gain
  +2.92 dB for the Chinese clip, +40.07 dB for the English one).
- Trimmed to the speech plus about 300 ms on each side: `zh_dinosaurs.wav` is 0.332–4.292 s of
  the levelled clip, `en_dinosaurs.wav` its first 4.228 s.

sha256: `zh_dinosaurs.wav` `d7fab9582df9c149e403a2ce11543616aba0352d519d73367c7c1d560dc36105`,
`en_dinosaurs.wav` `507b07c45545c33686a849d2d073f3bae0353049a391b288053e57a303c86da7`.

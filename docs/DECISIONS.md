# Decisions

One line per decision, newest last. Format: `YYYY-MM-DD — <decision> (<why>)`.

- 2026-10-22 — The integration target is ai-native-callcenter, as an OpenAI Realtime (GA) client; cascade-realtime-gateway is a design reference only, not a consumer (owner correction to the brief).
- 2026-10-22 — Astral uv manages every dependency and environment. Every `uv.lock` is committed. No pip, no venv, no `requirements.txt`. Nested projects use `uv init --no-workspace` (owner rule; `CLAUDE.md`).
- 2026-10-22 — D1–D19 in `docs/ARCHITECTURE.md` §12 approved as proposed (owner).
- 2026-10-22 — M0 test audio comes from public datasets only (owner).
- 2026-10-22 — FreeSWITCH provides no echo cancellation, so mlx-realtime cancels echo itself. D17 is revised to a per-session echo canceller referenced to our own output, with the onset guard and echo-turn suppression kept as a second line. The engine is chosen in M0 (S11, G7) on simulated telephone echo (owner).
- 2026-10-22 — The live echo measurement on a real AICC call is deferred (owner).
- 2026-10-22 — uv upgraded to 0.13.5 by the owner. Projects pin `[tool.uv] required-version = ">=0.13.5"`.
- 2026-10-22 — M0 is not started until the owner gives an explicit go-ahead (owner).
- 2026-10-22 — Echo cancellation (D17) is implemented in M4, alongside barge-in; M0 only selects the engine (S11, G7). Echo-triggered self-interruption is a barge-in correctness problem (owner).
- 2026-10-22 — AICC acceptance is split: M3 covers connectivity (trace replay; AICC's `probe` harness for the handshake, a spoken turn and the 3 s deadlines), and M4 covers barge-in (interruption, truncate/delete, simulated echo) (owner).
- 2026-10-23 — D17 wording corrected: FreeSWITCH does have a Speex-based echo canceller (`preprocess` with `echo_cancel`), but it is off by default and not enabled on these calls. The owner's decision stands: the service cancels echo (A3 research).
- 2026-10-23 — ITU-T G.168 echo-path coefficients are © ITU, all rights reserved: they are kept out of git (`.gitignore`), and are regenerated from the PDF when needed.

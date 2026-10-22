# mlx-realtime

A local OpenAI Realtime (GA) voice service on Apple Silicon.

- Design: `docs/ARCHITECTURE.md`.
- M0 plan and findings: `docs/SPIKE.md`.

## Dependencies and environments: uv only

Astral uv manages every dependency and every environment, from start to finish.

**Projects.**

- Every Python environment in this repository is a uv project:
  - the gateway, at the root;
  - `backends/llm` and `backends/audio`;
  - each `spikes/<name>`.
- Create the root project with `uv init`. Create every nested project with `uv init --no-workspace`.
  A plain `uv init` inside the repository joins a workspace and forces one shared resolution, which
  would undo the separate gateway, LLM and audio environments.

**Commands.**

- Change dependencies only with `uv add`, `uv add --dev` and `uv remove`.
- Build environments only with `uv sync`.
- Run everything with `uv run`.

**Files.**

- Commit every `uv.lock`, the spikes' included, so every environment and every measurement can be
  rebuilt exactly.
- Pin Python 3.12 with `uv python pin 3.12`, and commit `.python-version`.
- Every `pyproject.toml` sets `[tool.uv] required-version = ">=0.13.5"`.

**Automated runs.** `make dev`, launchd and CI use `uv sync --locked` and `uv run --locked`. A stale
lockfile then fails loudly instead of being re-resolved.

**Never:**

- `pip`, or `uv pip`;
- `python -m venv`;
- `requirements.txt`;
- `uv run --with` or `uvx` for anything the project depends on. Every dependency belongs in a
  lockfile.

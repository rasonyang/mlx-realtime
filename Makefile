# Every target runs in the locked uv environment: a stale uv.lock fails loudly (CLAUDE.md).
.PHONY: fmt lint typecheck test test-e2e test-backend test-llm-backend test-audio-backend models models-mlx dev dev-mock aicc-probe aicc-probe-mock bench bench-mock fairness

fmt:
	uv sync --locked
	uv run --locked ruff check --select I --fix .
	uv run --locked ruff format .

lint:
	uv sync --locked
	uv run --locked ruff check .
	uv run --locked ruff format --check .

typecheck:
	uv sync --locked
	uv run --locked pyright

test:
	uv sync --locked
	uv run --locked pytest -m "not e2e"

test-e2e:
	uv sync --locked
	uv run --locked pytest -m e2e

# Needs the real backends: run `make dev` in another terminal first.
test-backend:
	uv sync --locked
	uv run --locked pytest -m backend

# The LLM backend's own code (backends/llm: the lenient tool-call parser, the tool-definitions
# cache checkpoint) in its own environment; no GPU (the tokenizer test needs the Qwen3.6 snapshot).
test-llm-backend:
	uv sync --locked
	uv sync --locked --project backends/llm
	uv run --locked --project backends/llm pytest backends/llm/tests
	uv run --locked pyright -p backends/llm

# The audio backend's own code (backends/audio: the ASR context patch) in its own environment;
# no GPU, no model.
test-audio-backend:
	uv sync --locked
	uv sync --locked --project backends/audio
	uv run --locked --project backends/audio pytest backends/audio/tests
	uv run --locked pyright -p backends/audio

# The ONNX models (Silero VAD, Smart Turn) into models/, verified against models/MANIFEST.json.
models:
	uv sync --locked
	uv run --locked mlxrt models

# Also the MLX repositories pinned in config.yaml, through each backend's hf CLI (about 25 GB).
models-mlx:
	uv sync --locked
	uv run --locked mlxrt models --mlx

# The model backends of config.yaml (each in its own uv project) and the gateway.
dev: models
	uv run --locked mlxrt dev

# The gateway alone, with mock providers (the M1 setup) and the real CPU VAD (models/).
dev-mock: models
	uv run --locked mlxrt --config config.mock.yaml dev

# AICC's own Realtime client harness (probe, measureResponseCreated, the lines check, and the M4
# barge-in and keypress scenarios in tests/aicc/zz_mlx_live_test.go, added with go test -overlay)
# against a running gateway:
#   make aicc-probe AICC_REPO=~/src/ai-native-callcenter [AICC_URL=ws://127.0.0.1:8000/v1/realtime]
AICC_URL ?= ws://127.0.0.1:8000/v1/realtime
aicc-probe:
	uv sync --locked
	uv run --locked python tests/aicc/probe.py --aicc-repo "$(AICC_REPO)" --url "$(AICC_URL)"

# The same against a gateway of its own with mock providers (config.mock.yaml, no backends).
aicc-probe-mock: models
	uv run --locked python tests/aicc/probe.py --aicc-repo "$(AICC_REPO)" --serve config.mock.yaml

# The latency and ASR benchmark (docs/LATENCY.md) against a running gateway started with
# MLXRT__OBSERVABILITY__DEBUG_ENDPOINTS=true (for the server's stage durations):
#   make bench [BENCH_ARGS="--variant tel8k --concurrency 1 --out bench/results/<name>"]
BENCH_URL ?= ws://127.0.0.1:8000/v1/realtime
BENCH_ARGS ?=
bench:
	uv sync --locked
	uv run --locked python -m bench.bench --url "$(BENCH_URL)" $(BENCH_ARGS)

# The same against a gateway of its own on the mock stack (config.mock.yaml): the format example.
bench-mock: models
	uv run --locked python -m bench.bench --serve config.mock.yaml $(BENCH_ARGS)

# Multi-session TTS fairness, before and after the M6 mitigations (fake TTS at M0 S3's pace).
fairness:
	uv sync --locked
	uv run --locked python -m tests.fairness $(FAIRNESS_ARGS)

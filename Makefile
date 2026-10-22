# Every target runs in the locked uv environment: a stale uv.lock fails loudly (CLAUDE.md).
.PHONY: fmt lint typecheck test test-e2e test-backend models models-mlx dev dev-mock aicc-probe aicc-probe-mock

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

# The gateway alone, with mock providers (the M1 setup).
dev-mock:
	uv sync --locked
	uv run --locked mlxrt --config config.mock.yaml dev

# AICC's own Realtime client harness (probe, measureResponseCreated, and the lines check in
# tests/aicc/zz_mlx_live_test.go, added with go test -overlay) against a running gateway:
#   make aicc-probe AICC_REPO=~/src/ai-native-callcenter [AICC_URL=ws://127.0.0.1:8000/v1/realtime]
AICC_URL ?= ws://127.0.0.1:8000/v1/realtime
aicc-probe:
	uv sync --locked
	uv run --locked python tests/aicc/probe.py --aicc-repo "$(AICC_REPO)" --url "$(AICC_URL)"

# The same against a gateway of its own with mock providers (config.mock.yaml, no backends).
aicc-probe-mock:
	uv sync --locked
	uv run --locked python tests/aicc/probe.py --aicc-repo "$(AICC_REPO)" --serve config.mock.yaml

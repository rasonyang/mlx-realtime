# Every target runs in the locked uv environment: a stale uv.lock fails loudly (CLAUDE.md).
.PHONY: fmt lint typecheck test test-e2e test-backend models models-mlx dev dev-mock

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

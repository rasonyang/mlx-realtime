# Every target runs in the locked uv environment: a stale uv.lock fails loudly (CLAUDE.md).
.PHONY: fmt lint typecheck test test-e2e dev

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

# M1: the gateway with mock providers. From M2 on, the supervisor that also starts the backends.
dev:
	uv sync --locked
	uv run --locked mlxrt dev

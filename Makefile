# PropIntelli AI: developer convenience targets.
# All Python targets run inside the uv-managed virtual environment.

.PHONY: help install format lint typecheck test check samples e2e ui ui-llm clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install all extras + dev tools.
	uv sync --all-extras

format:  ## Auto-format the codebase with ruff.
	uv run ruff format .

lint:  ## Lint (ruff) and verify formatting without writing changes.
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Static type check (mypy strict).
	uv run mypy

test:  ## Run the test suite with coverage (fails under 80%).
	uv run pytest --cov=propintelli --cov-report=term-missing --cov-fail-under=80

check: lint typecheck test  ## Run the full local quality gate.

samples:  ## Generate the synthetic sample exposes + ground truth.
	uv run propintelli generate-samples

e2e: samples  ## Generate samples, batch-process them, then evaluate.
	uv run propintelli batch sample_data/raw
	uv run propintelli evaluate

ui:  ## Launch the Streamlit human-in-the-loop demo (deterministic, offline).
	uv run streamlit run app/streamlit_app.py

ui-llm:  ## Launch the demo with the local Ollama LLM second opinion enabled (slower).
	PROPINTELLI_LLM_PROVIDER=ollama \
	PROPINTELLI_LLM_PROMPT_VARIANT=v2_schema \
	PROPINTELLI_LLM_TIMEOUT_SECONDS=600 \
	uv run streamlit run app/streamlit_app.py

clean:  ## Remove caches and generated runtime data.
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml data
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

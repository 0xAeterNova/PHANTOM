PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help install install-audio-ml install-ml install-all format lint type test test-cov security check run-api run-demo docker-build docker-up docker-real docker-down

help: ## Show available development commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Project PHANTOM targets:\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the editable API, demo, and development toolchain.
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[api,demo,dev]"

install-ml: ## Add optional local machine-learning dependencies.
	$(PYTHON) -m pip install -e ".[ml]"

install-audio-ml: ## Add only the optional PyTorch spectrogram audio dependency.
	$(PYTHON) -m pip install -e ".[audio-ml]"

install-all: ## Install API, demo, development, and ML dependency groups.
	$(PYTHON) -m pip install -e ".[api,demo,ml,dev]"

format: ## Apply Ruff formatting and safe lint fixes.
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint: ## Check formatting and lint rules without changing files.
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

type: ## Run strict static type checking on the package.
	$(PYTHON) -m mypy src/phantom

test: ## Run the core test suite without slow or optional-ML tests.
	$(PYTHON) -m pytest -m "not slow and not optional_ml"

test-cov: ## Run core tests with branch coverage.
	$(PYTHON) -m pytest -m "not slow and not optional_ml" --cov=phantom --cov-report=term-missing --cov-report=xml

security: ## Scan source and the active environment for common vulnerabilities.
	$(PYTHON) -m bandit -c pyproject.toml -r src app scripts
	$(PYTHON) -m pip_audit --local --skip-editable

check: lint type test ## Run the local quality gate.

run-api: ## Start the local API on the loopback interface.
	$(PYTHON) -m uvicorn app.api:app --host 127.0.0.1 --port 8000

run-demo: ## Start the Streamlit demonstration on the loopback interface.
	$(PYTHON) -m streamlit run app/web_demo.py --server.address 127.0.0.1

docker-build: ## Build the lightweight integrated browser demo image.
	docker compose build

docker-up: ## Start the simulated browser demo on host loopback.
	docker compose up

docker-real: ## Build and start the real CPU browser profile on host loopback.
	docker compose -f docker-compose.yml -f compose.realtime.yaml up --build

docker-down: ## Stop the browser container; preserve downloaded model weights.
	docker compose down --remove-orphans

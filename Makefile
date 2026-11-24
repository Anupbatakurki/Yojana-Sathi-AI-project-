.PHONY: help install lock lint fmt type test eval check api ui up up-monitoring down clean

help:           ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:        ## Install pinned dev dependencies + package (editable) + git hooks
	pip install -r requirements/dev.txt && pip install --no-deps -e . && pre-commit install

lock:           ## Re-pin requirements/*.txt from pyproject.toml (needs uv)
	uv pip compile pyproject.toml -o requirements/api.txt --python-version 3.12
	uv pip compile pyproject.toml --extra ui -o requirements/ui.txt --python-version 3.12
	uv pip compile pyproject.toml --extra ui --extra dev -o requirements/dev.txt --python-version 3.12

lint:           ## Ruff lint + format check
	ruff check . && ruff format --check .

fmt:            ## Auto-format
	ruff check . --fix && ruff format .

type:           ## mypy
	mypy

test:           ## Tests with coverage gate
	pytest --cov

eval:           ## Retrieval quality gate
	python -m eval.evaluate --min-hit3 0.95

check: lint type test eval  ## Everything CI runs
	python scripts/validate_schemes.py

api:            ## Run API locally with reload
	uvicorn saathi.api.main:app --reload --port 8000

ui:             ## Run UI locally (set API_URL to use the API)
	streamlit run streamlit_app.py

up:             ## Build & start api + ui in Docker
	docker compose up --build

up-monitoring:  ## Same, plus Prometheus (9090) and Grafana (3000)
	docker compose --profile monitoring up --build

down:           ## Stop containers
	docker compose --profile monitoring down

clean:          ## Remove caches
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build *.egg-info

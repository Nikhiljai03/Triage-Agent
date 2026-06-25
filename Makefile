# Convenience targets for the Triage Agent.
# Usage: `make up`, `make test`, `make lint`, ...
.PHONY: up down logs test lint fmt demo help

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-8s %s\n", $$1, $$2}'

up:  ## Build images and start all services
	docker compose up --build

down:  ## Stop and remove containers, networks
	docker compose down

logs:  ## Follow logs from all services
	docker compose logs -f

demo:  ## Post a signed sample issues.opened webhook to the running API
	python -m scripts.simulate_webhook

test:  ## Run the test suite
	pytest

lint:  ## Check lint rules and formatting (no changes)
	ruff check . && black --check .

fmt:  ## Auto-format and auto-fix
	black . && ruff check --fix .

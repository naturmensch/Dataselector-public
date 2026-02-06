# Canonical package-first Makefile (CLI hard-cut)
.PHONY: clean clean-force format format-check test env-create env-update ensure-env check-env test-integration test-e2e test-e2e-auto test-e2e-ci archive-outputs restore-outputs gate-quick gate-batch-a gate-batch-b gate-comprehensive branch-supersets

PYTHON ?= python
ENV_NAME ?= dataselector

clean:
	@echo "Dry-run: show candidates to be cleaned"
	@$(PYTHON) -m dataselector clean-workspace --delete-outputs --delete-cache

clean-force:
	@echo "Deleting outputs/cache/venvs (protected paths are kept)"
	@$(PYTHON) -m dataselector clean-workspace --delete-outputs --delete-cache --delete-venvs --yes

format:
	@echo "Formatting code with isort, black and ruff"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install isort black ruff
	@isort .
	@black .
	@ruff check --fix .

format-check:
	@echo "Check formatting (isort/black/ruff)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install isort black ruff
	@isort --check-only .
	@black --check .
	@ruff check .

test:
	@echo "Running tests with canonical interpreter"
	@$(PYTHON) -m pytest -q

env-create:
	@echo "Creating/updating conda env '$(ENV_NAME)' from environment.yml"
	@conda env update -n $(ENV_NAME) -f environment.yml --prune || conda env create -n $(ENV_NAME) -f environment.yml

env-update:
	@echo "Updating conda env '$(ENV_NAME)' from environment.yml"
	@conda env update -n $(ENV_NAME) -f environment.yml --prune

ensure-env: env-create

check-env:
	@echo "Checking environment and command hygiene"
	@$(PYTHON) -m dataselector check-env dataselector tests Makefile .github/workflows

test-integration:
	@echo "Running curated integration tests"
	@$(PYTHON) -m pytest -q tests/test_integration_diversity_selector.py tests/test_full_pipeline_integration.py tests/test_full_pipeline_comprehensive.py

test-e2e-auto:
	@$(MAKE) ensure-env
	@$(MAKE) test-e2e-ci

test-e2e-ci:
	@echo "(CI) Running E2E tests in conda env '$(ENV_NAME)'"
	@conda run -n $(ENV_NAME) python -m pytest -q -m e2e

test-e2e:
	@echo "Running e2e smoke tests in current interpreter"
	@$(PYTHON) -m pytest -q -m e2e

archive-outputs:
	@echo "Archive outputs to data/archive/"
	@$(PYTHON) -m dataselector archive-outputs --outputs outputs --dest data/archive

restore-outputs:
	@echo "Restore latest archive from data/archive/"
	@$(PYTHON) -m dataselector list-archives --dir data/archive

gate-quick:
	@./scripts/validate_merge_gate.sh "Quick Gate" "make test"

gate-batch-a:
	@./scripts/validate_merge_gate.sh "Batch A Gate" "make test"

gate-batch-b:
	@./scripts/validate_merge_gate.sh "Batch B Gate" "make check-env && make test"

gate-comprehensive:
	@./scripts/validate_merge_gate.sh "Phase 2 Complete" "make check-env && make test && make test-integration"

branch-supersets:
	@./scripts/check_branch_supersets.sh origin/integration

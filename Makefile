# Makefile for common tasks
.PHONY: clean clean-force test

clean:
	@echo "Dry-run: show candidates to be cleaned"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -m dataselector tools clean-workspace --dry-run

clean-force:
	@echo "Deleting outputs and venvs (will skip protected paths)."
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -m dataselector tools clean-workspace --delete-outputs --delete-venvs

format:
	@echo "Formatting code with isort, black and ruff"
	$(EXEC_ENV) python -m pip install --upgrade pip
	$(EXEC_ENV) python -m pip install isort black ruff
	isort .
	black .
	ruff check --fix . || true

format-check:
	@echo "Check formatting (isort/black/ruff)"
	$(EXEC_ENV) python -m pip install --upgrade pip
	$(EXEC_ENV) python -m pip install isort black ruff
	isort --check-only .
	black --check .
	ruff check .

test:
	@echo "Running tests inside '$(ENV_NAME)' environment"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q

# Environment and integration helpers
.PHONY: env-create env-update check-env test-integration test-e2e

ENV_NAME := dataselector

env-create:
	@echo "Create/update conda env '$(ENV_NAME)'"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --create --ensure-packages "numpy==1.26.4 numba==0.63.1" --yes -- true

env-update:
	@echo "Update conda env '$(ENV_NAME)'"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --update --ensure-packages "numpy==1.26.4 numba==0.63.1" --yes -- true

# Ensure env exists and required package constraints are present (safe opt-in)
ensure-env:
	@echo "Create/update conda env '$(ENV_NAME)' with required packages"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --create --ensure-packages "numpy==1.26.4 numba==0.63.1" --yes -- true

check-env:
	@echo "Checking environment compatibility"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -m dataselector tools check-env scripts

test-integration:
	@echo "Running a curated set of integration tests inside $(ENV_NAME)" \
	&& ./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q tests/test_integration_diversity_selector.py tests/test_full_pipeline_integration.py tests/test_full_pipeline_comprehensive.py

# Convenient wrapper: ensure environment exists and run E2E tests (opt-in)
.PHONY: test-e2e-auto test-e2e-ci

test-e2e-auto:
	@echo "Ensuring env and running E2E smoke tests inside $(ENV_NAME)"
	@$(MAKE) ensure-env
	@$(MAKE) test-e2e

# CI-oriented target: non-interactive creation and run of E2E tests
test-e2e-ci:
	@echo "(CI) Creating env and running E2E tests non-interactively"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --create --ensure-packages "numpy==1.26.4 numba==0.63.1" --yes -- true \
	&& ./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q -m e2e

test-e2e:
	@echo "Running e2e smoke tests inside $(ENV_NAME)" \
	&& ./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q -m e2e

archive-outputs:
	@echo "Archive outputs to data/archive/"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -m dataselector tools archive-outputs --path outputs --output data/archive

restore-outputs:
	@echo "Restore latest archive from data/archive/"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -c "import pathlib,sys; import glob; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print('No archive found' if not a else a[-1]); sys.exit(0 if a else 1)" \
	&& @./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -m dataselector tools restore-archive --pattern data/archive/outputs_archive_*.tar.gz --dest .

# Phase 2 merge gate helpers
.PHONY: gate-quick gate-batch-a gate-batch-b gate-comprehensive

gate-quick:
	@./scripts/validate_merge_gate.sh "Quick Gate" "make test"

gate-batch-a:
	@./scripts/validate_merge_gate.sh "Batch A Gate" "make test"

gate-batch-b:
	@./scripts/validate_merge_gate.sh "Batch B Gate" "make check-env && make test"

gate-comprehensive:
	@./scripts/validate_merge_gate.sh "Phase 2 Complete" "make check-env && make test && make test-integration"

# Branch analysis
.PHONY: branch-supersets

branch-supersets:
	@./scripts/check_branch_supersets.sh origin/integration


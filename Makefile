# Makefile for common tasks
.PHONY: clean clean-force test

clean:
	@echo "Dry-run: show candidates to be cleaned"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python scripts/clean_workspace.py --dry-run

clean-force:
	@echo "Deleting outputs and venvs (will skip protected paths)."
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python scripts/clean_workspace.py --delete-outputs --delete-venvs

format:
	@echo "Formatting code with isort, black and ruff"
	python -m pip install --upgrade pip
	pip install isort black ruff
	isort .
	black .
	ruff check --fix . || true

format-check:
	@echo "Check formatting (isort/black/ruff)"
	python -m pip install --upgrade pip
	pip install isort black ruff
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
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --create --ensure-packages "numpy<2.4 numba=0.63.1" --yes -- true

env-update:
	@echo "Update conda env '$(ENV_NAME)'"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) --update --ensure-packages "numpy<2.4 numba=0.63.1" --yes -- true

check-env:
	@echo "Checking environment compatibility"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python scripts/check_env.py

test-integration:
	@echo "Running a curated set of integration tests inside $(ENV_NAME)" \
	&& ./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q tests/test_integration_diversity_selector.py tests/test_full_pipeline_integration.py tests/test_full_pipeline_comprehensive.py

test-e2e:
	@echo "Running e2e smoke tests inside $(ENV_NAME)" \
	&& ./scripts/exec_in_env.sh --env $(ENV_NAME) -- pytest -q -m e2e

archive-outputs:
	@echo "Archive outputs to data/archive/"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python scripts/manage_archives.py archive --outputs outputs --dest data/archive $(foreach p,$(EXCLUDE),--exclude $(p))

restore-outputs:
	@echo "Restore latest archive from data/archive/"
	@./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -c "import pathlib,sys; import glob; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print('No archive found' if not a else a[-1]); sys.exit(0 if a else 1)" \
	&& @./scripts/exec_in_env.sh --env $(ENV_NAME) -- python scripts/manage_archives.py restore --archive $(shell ./scripts/exec_in_env.sh --env $(ENV_NAME) -- python -c "import pathlib; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print(a[-1])") --dest .

# Makefile for common tasks
.PHONY: clean clean-force test

clean:
	@echo "Dry-run: show candidates to be cleaned"
	python scripts/clean_workspace.py --dry-run

clean-force:
	@echo "Deleting outputs and venvs (will skip protected paths)."
	python scripts/clean_workspace.py --delete-outputs --delete-venvs

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
	pytest

# Developer helpers
# Note: `requirements-optional.txt` is not maintained by default anymore.
# Install optional packages ad-hoc (e.g., `pip install dppy`) or open a PR to add a curated optional file.

check-deps:
	@echo "Run import scanner to detect unlisted imports"
	python tools/check_imports.py --requirements requirements-cpu.txt


# Create / refresh development conda/mamba environment
env:
	@echo "Create conda/mamba environment 'dataselector' (default: python=3.11)"
	./scripts/create_env.sh dataselector 3.11

env-force:
	@echo "Recreate 'dataselector' environment from scratch"
	./scripts/create_env.sh dataselector 3.11 --force

# Local venv targets (create .venv and install the project in editable mode)
venv:
	@echo "Create local python venv '.venv' and install project in editable mode"
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && (pip install -r requirements-cpu.txt || pip install -r requirements.txt || true)
	. .venv/bin/activate && pip install -e .

venv-force:
	@echo "Remove and recreate .venv"
	rm -rf .venv
	$(MAKE) venv

# Run the full test-suite inside the dataselector conda environment (non-interactive)
# Usage: make test-integration
test-integration:
	@echo "Activating 'dataselector' env and running pytest (requires conda/mamba installed)"
	bash -lc "source $(conda info --base)/etc/profile.d/conda.sh && conda activate dataselector && pytest"


archive-outputs:
	@echo "Archive outputs to data/archive/"
	python scripts/manage_archives.py archive --outputs outputs --dest data/archive $(foreach p,$(EXCLUDE),--exclude $(p))

restore-outputs:
	@echo "Restore latest archive from data/archive/"
	python -c "import pathlib,sys; import glob; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print('No archive found' if not a else a[-1]); sys.exit(0 if a else 1)" \
	&& python scripts/manage_archives.py restore --archive $(python -c "import pathlib; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print(a[-1])") --dest .

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
deps-optional:
	@echo "Install optional dependencies for experimentation (Jupyter, DPP, GP libs)"
	python -m pip install -r requirements-optional.txt

check-deps:
	@echo "Run import scanner to detect unlisted imports"
	python tools/check_imports.py --requirements requirements-cpu.txt


archive-outputs:
	@echo "Archive outputs to data/archive/"
	python scripts/manage_archives.py archive --outputs outputs --dest data/archive $(foreach p,$(EXCLUDE),--exclude $(p))

restore-outputs:
	@echo "Restore latest archive from data/archive/"
	python -c "import pathlib,sys; import glob; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print('No archive found' if not a else a[-1]); sys.exit(0 if a else 1)" \
	&& python scripts/manage_archives.py restore --archive $(python -c "import pathlib; a=list(pathlib.Path('data/archive').glob('outputs_archive_*.tar.gz')); a.sort(); print(a[-1])") --dest .

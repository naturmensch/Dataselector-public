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
	pytest -q

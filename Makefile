# Makefile for common tasks
.PHONY: clean clean-force test

clean:
	@echo "Dry-run: show candidates to be cleaned"
	python scripts/clean_workspace.py --dry-run

clean-force:
	@echo "Deleting outputs and venvs (will skip protected paths)."
	python scripts/clean_workspace.py --delete-outputs --delete-venvs

test:
	pytest -q

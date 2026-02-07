# Contributing

Thanks for contributing! This short guide explains the workflow for safe cleanup work in this repository.

## Branches
- Keep `main` stable.
- Work in feature branches (e.g., `cleanup/initial`, `fix/...`) and open PRs against `main`.

## Pre-commit
We use a light local pre-commit hook to prevent accidental commits to protected paths (e.g., `data/images`). To enable locally:

```bash
pip install pre-commit
pre-commit install
# Run checks manually
pre-commit run --all-files
```

There is also a GitHub Actions CI check that validates tracked files do not include protected paths.

## Cleanup workflow
- Use `scripts/clean_workspace.py --dry-run` to inspect candidates.
- `data/images` and other configured protected paths will be shown as PROTECTED and will not be deleted by default.
- To archive a folder safely, use:

```bash
python scripts/clean_workspace.py --archive data/images /path/to/archive.tar.gz
```

- To delete outputs or virtualenvs only (skips protected paths):

```bash
python scripts/clean_workspace.py --delete-outputs --delete-venvs
```

`make clean` runs a dry-run by default; use `make clean-force` to perform deletion (it will still skip protected paths).

## Testing
- Tests include safety checks for protected files. Run the test suite with `pytest`.
- CI runs `scripts/check_protected.py --all` to ensure no tracked files are in protected paths.

### Developer Quick Workflow
- **Debug:** `pytest --lf` (fast feedback loop for last failed tests)
- **Before commit:** `pytest` (run the full test suite)
- **CI Simulation:** `pytest --junitxml=test-results/junit.xml`

## CI Parity Local Run (Authoritative)

Use the `micromamba` environment `dataselector` for all authoritative local checks.
Running with system Python may fail collection due to missing optional dependencies and
is not considered a valid CI-parity signal.

```bash
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
micromamba run -n dataselector python -m pytest -q tests -k "not real_images"
git ls-files '*.py' | xargs micromamba run -n dataselector ruff check
git ls-files '*.py' | xargs micromamba run -n dataselector black --check
git ls-files '*.py' | xargs micromamba run -n dataselector isort --check-only
```

### Local-only artifacts

Some local analysis helpers and reports are intentionally not pushed. Current examples:
- `docs/reports/`
- `scripts/summarize_md.py`

Treat these as workstation-local unless a dedicated PR explicitly promotes them.

If you're unsure, open a draft PR and ask for a quick review before deleting anything important.

Developer notes

- Canonical runtime invocation in this repository: `micromamba run -n dataselector python -m dataselector <command>`.
- `scripts/exec_in_env.sh` remains available as a compatibility wrapper.

## Collaboration workflow

- Start each task with a short discovery pass in the real repository surface before choosing an implementation path.
- After discovery, send a short check-in that states the problem understanding, affected surface, planned approach, task-specific checks, and any optional expanded checks as exact commands.
- Treat `dataselector/` as the scientific core and `scripts/` as thin wrappers/orchestrators only.
- Use `config/pipeline_config.yaml`, `outputs/runs/`, and the package-first CLI flow as active defaults.
- Treat historical configs, legacy workflows, and archive docs as reference-only unless a task explicitly targets them.

## Validation policy

- Default verification is task-specific: run the smallest relevant checks for the area being changed.
- Do not widen the validation scope silently. If broader coverage is useful, provide the exact commands and let that expanded scope be explicit.
- Every closeout should state what was run, what was not run, and which expanded checks are recommended next when relevant.

### Typical expanded commands

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py
micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py
make format-check
make test
```

- venv: Use the project virtualenv at `.venv/` where present. Create via:

  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements-cpu.txt

- Alternativ: Verwende `make venv` oder `scripts/setup_local_venv.sh` um `.venv` automatisch zu erstellen und das Projekt editierbar zu installieren (`pip install -e .`).

- Optional dependencies:
  The project no longer maintains a curated `requirements-optional.txt` by default.
  Install optional packages as needed (e.g., `pip install dppy`) or open a PR to add a curated optional requirements file.

- Quick checks:
  make check-deps       # run import scanner that compares imports vs requirements

- Tests:
  `micromamba run -n dataselector python -m pytest -q`

### Integration tests (Optuna / Numba)
- Some integration tests require heavy dependencies (e.g., `optuna`, `numba` and compatible `numpy`). These tests are intentionally separated and must be run in a micromamba environment (or exec_in_env wrapper) to ensure correct binary compatibility.

- Local quick start:
  1. Install micromamba.
  2. Create the project environment:
    ```bash
    micromamba create -f environment.yml -n dataselector || micromamba env update -f environment.yml -n dataselector
    ```
  3. Install Python dependencies:
    ```bash
    micromamba run -n dataselector python -m pip install -r requirements.txt
    ```

- Important: Tests enforcement
  - This repository **enforces** running tests inside the `dataselector` micromamba environment (or exec_in_env wrapper). If you run `pytest` outside that environment the test run will abort with an instructive message telling you how to create & activate the environment.
  - If you need to explicitly bypass the guard on a developer machine (not recommended), set `SKIP_ENV_CHECK=1` in your shell to continue. Use this only for quick debugging; CI and reproducible test runs should use `dataselector`.
- Run the integration test subset (or the full suite):
  ```bash
  micromamba run -n dataselector python -m pytest -q
  # or to run only integration tests (marked):
  micromamba run -n dataselector python -m pytest -q -m integration
  ```

### Geo Development Quick Guide 🔧

- Install pinned geo stack in `dataselector` micromamba env (recommended via `environment.yml`):
  ```bash
  # Prefer creating from the lockfile for reproducibility, or use micromamba with environment.yml
  # Reproducible: conda-lock install --name dataselector locks/conda-lock-linux-64.lock
  micromamba create -f environment.yml -n dataselector || micromamba env update -f environment.yml -n dataselector
  # Install pip-only extras (PyTorch CPU wheel via official PyTorch index) from requirements-cpu.txt
  micromamba run -n dataselector python -m pip install -r requirements-cpu.txt
  ```

- Quick checks:
  - Run the geo smoke checker: `micromamba run -n dataselector python scripts/check_geo_env.py` (respects `config/pipeline_config.yaml: features.geo`).
  - Run the alignment audit (M2.5): `micromamba run -n dataselector python scripts/align_audit.py --csv data/new_all_tiles.csv --out outputs/align_audit.json --plot outputs/align_audit.png`.
  - Note: `outputs/` artifacts are git-ignored locally; CI runs publish artifacts (e.g., `align_audit_*.json`, plots) as workflow artifacts — see CI job `geo-smoke` for details.

- Regenerate lockfile (manual or via CI): We provide a `regenerate-lockfile` workflow (`.github/workflows/regenerate-lockfile.yml`) that can be triggered manually to create updated `locks/conda-lock-*.lock`. Prefer the CI workflow for consistent binaries (conda-forge, micromamba).

- In PRs: Ensure the `geo-smoke` job in `.github/workflows/geo-integration.yml` passes before merging. This job verifies geo imports and runs a small alignment audit in the locked environment.



- CI: A dedicated GitHub Actions workflow (`.github/workflows/integration-optuna-numba.yml`) runs these tests in a micromamba environment. If a CI job still fails because of system packages, please open an issue and include the workflow run log plus the affected run directory and its diagnostics artifacts (`run_metadata.json`, `logs/status.log`, optional `monitor/summary.json`) if applicable.

- Rationale: Running these tests in an isolated micromamba env prevents ABI mismatches (notably `numba` vs `numpy`) and keeps the main unit test jobs fast and deterministic.

- **Developer Quick Workflow:**
  - Debug: `micromamba run -n dataselector python -m pytest --lf` (fast feedback loop for last failed tests)
  - Before commit: `micromamba run -n dataselector python -m pytest` (run the full test suite to detect regressions)
  - Generating test reports: `micromamba run -n dataselector python -m pytest --junitxml=test-results/junit-<python-version>-<optuna-version>.xml` (CI uploads the generated JUnit XML as artifacts named `junit-results-<python-version>-<optuna-version>`; you can download them from the Actions run page)

- CI: There is an import-scanner step (`tools/check_imports.py`) that prints unlisted imports; maintainers should evaluate suggestions and add packages to `requirements-cpu.txt` or to `requirements-optional.txt` as appropriate.

### Pre-commit & Development Hooks ✅

We use `pre-commit` to enforce formatting, imports and basic static checks before commits. To set it up locally in the `dataselector` micromamba environment (or exec_in_env wrapper) do:

```bash
# inside a shell with access to micromamba or exec_in_env
micromamba run -n dataselector python -m pip install pre-commit
# install the git hooks
micromamba run -n dataselector pre-commit install
# run all hooks once over the repository
micromamba run -n dataselector pre-commit run --all-files
```

The repository already includes `.pre-commit-config.yaml` (Black, isort, Ruff, and a small local check script). Running the hooks before commits helps keep the CI green and avoids trivial style regressions.

## Script Coding Standards 📜

To ensure maintainability and testability (especially for CI smoke tests), all Python scripts in `scripts/` must follow these rules:

1.  **No module-level side effects**: Do not execute heavy logic, perform I/O, or print messages at the top level.
2.  **Deferred Imports**: Move heavy imports (like `optuna`, `torch`, `umap`) inside functions or `main()` to keep the initial import fast and environment-agnostic.
3.  **Encapsulation**: Wrap the primary execution logic in a `main() -> int` function.
4.  **Standard Entrypoint**: Use the following pattern at the end of the script:
    ```python
    if __name__ == "__main__":
        raise SystemExit(main())
    ```
5.  **Import Safety**: Every script must be importable via `import scripts.your_script` without triggering its CLI behavior. Verification is done via `tools/smoke_check_scripts.py`.

## Thesis run diagnostics 🔍

Canonical thesis runs write their diagnostics into the run directory under
`outputs/runs/<run_id>/`.

- Core artifacts:
  - `run_metadata.json`
  - `logs/status.log`
  - `manifest/`
- Optional runtime snapshots:
  - `monitor/summary.json`
- Human-readable summary:
  - `micromamba run -n dataselector python -m dataselector generate-monitor --run-dir outputs/runs/<run_id>`

Use these artifacts when filing issues about failed or suspicious thesis runs.
The old XXL monitor/resume metadata surface is archived and is not part of the
active developer contract anymore.

## Konfiguration & Verhalten: `n_samples`
- **Persistente Einstellung:** `selection.n_samples` in `config/pipeline_config.yaml` kann nun dauerhaft die finale Auswahlgröße steuern. Ist der Wert `null`, wird die adaptive Heuristik (`dataselector.pipeline.pipeline_utils.compute_adaptive_n_initial`) verwendet.
- **CLI-Override:** Das Flag `--n-samples` überschreibt `config.selection.n_samples` temporär für einen einzelnen Run.
- **Warum:** Vermeidet widersprüchliche harte Defaults in Code und Konfiguration; ermöglicht reproduzierbare Experimente durch Setzen in der Config.

## Log-Management & Run-Summaries (Hinweis)
- Canonical thesis runs already write run-local logs such as `logs/status.log` and `run_metadata.json`.
- Für eine nachträgliche kompakte Zusammenfassung verwende `micromamba run -n dataselector python -m dataselector generate-monitor --run-dir outputs/runs/<run_id>`.
- Historische XXL-Monitor-/Merge-Helfer gehören in den Archivkontext unter `docs/07_ARCHIVE/legacy_xxl_ops/` und sind kein aktiver Developer-Default mehr.

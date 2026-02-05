# Changelog

## [v1.1.0] - 2026-02-05 - Pattern C CLI Migration (Phase 1 Complete)

### Added
- **Decorator-Based CLI System**: Introduced `@cli_command` decorator for centralized command definitions
  - Single source of truth for all CLI commands (workflows + tools)
  - Automatic argparse parser generation via `build_parser_from_decorators()`
  - Automatic command dispatch via `dispatch_from_decorators()`
  - 27 commands registered in `_CLI_COMMANDS` global registry (17 workflows + 10 tools)

### Changed
- **cli.py Refactoring**: Replaced 804 LOC manual argparse dispatcher with 64 LOC decorator-based version (**92% reduction**)
  - Old: Manual argparse setup (476 LOC) + if/elif dispatcher (327 LOC)
  - New: Module imports + 2 decorator function calls
  - All commands now accessible via `dataselector <command>` (not just `dataselector tools <command>`)
  - Backup preserved as `cli_old_backup.py`

- **Tool Commands Migration**: Migrated 10 tool commands to `@cli_command` decorators
  - `check-env`, `check-protected`, `check-geo` (check.py)
  - `verify-archive`, `archive-outputs`, `list-archives` (archive.py)
  - `align-audit` (audit.py)
  - `clean-workspace` (clean.py)
  - `docs-link-check`, `docs-link-autofix` (docs_link.py)
  - All with comprehensive unit tests (20 tests)

### Fixed
- **Boolean Argument Handling**: Fixed 5 workflow commands with incorrect boolean argument definitions
  - `compare-samplers`: Added `type=bool` to `sequential` arg
  - `final-selection`: Added `type=bool` to `use_bootstrap_best` arg
  - `thesis-pipeline`: Added `type=bool` to `skip_*` and `dry_run` args
  - Fixed `cli_decorators.py` to not add `type` kwarg for boolean flags with `store_true`/`store_false` actions

- **Test Infrastructure**: Fixed test registry population in `conftest.py`
  - Import all CLI modules to ensure `_CLI_COMMANDS` is populated before test collection
  - All 46 Pattern C unit tests passing

### Documentation
- **ADRs (Architecture Decision Records)**:
  - ADR-001: Pattern C Migration rationale and implementation (Status: Implemented)
  - ADR-002: Workflow Decision Matrix (4 main workflows)
  - MADR format adopted for future decisions
- **WHICH_WORKFLOW.md**: Comprehensive workflow selection guide with decision tree
- **Test Baseline**: Created `test_baseline_20260205.txt` (246 tests) for regression detection
- **Git Tag**: `v1.0.0-pre-migration` (rollback point before Phase 1)

### Commits
- 757b1e5: Phase 0 Foundation (ADRs, WHICH_WORKFLOW.md, test baseline)
- d31a31b: Fix boolean argument handling in cli_decorators
- 8149257: Phase 1a.1 - Migrate check-env
- e2a5baa: Phase 1a.2 - Migrate check-protected (+ legacy cleanup)
- 69f1650: Phase 1a.3 - Migrate check-geo
- fd09148: Phase 1a.4-6 - Migrate archive.py (3 commands)
- 8ad7afd: Phase 1a.7-10 - Migrate remaining tools (4 commands)
- 785eae5: Phase 1b - Replace cli.py with decorator dispatcher
- 2827c21: Update ADR-001 status to Implemented
- 81b6d1f: Fix CLI module imports in conftest

### Known Issues
- `xxl.py` calls non-existent `bootstrap` command (should be `bootstrap-final` or `bootstrap-pareto`) - will be addressed in future update
- 20 `test_scripts_safety.py` failures are pre-existing from Phase 4 migration (scripts/ removal)

---

## [2026-02-02] - Phase 4: Code Migration — Hard Cutover src/ → dataselector/
- **Major Refactoring**: Executed hard cutover migration of `src/` directory to `dataselector/` package structure.
- **Subpackage Architecture**:
  - `dataselector/selection/`: clustering, diversity_selector, spatial_facility_location, multi_criteria_facility_location, lazy_facility_location, pareto
  - `dataselector/data/`: io, metadata_processor (+ existing load.py, tiles.py)
  - `dataselector/features/`: feature_extractor (+ existing pipeline.py)
  - `dataselector/pipeline/`: experiment_manager, experiments, cache, pipeline_utils, incremental_results, main
  - `dataselector/analysis/`: metrics, visualizer, wandb_logger
  - `dataselector/workflows/`: sampling_strategies (+ existing modules)
  - `dataselector/compat.py`: root-level compatibility module
- **Import Codemod**: 57+ files migrated with automated import replacements (`from src.x import` → `from dataselector.<pkg>.x import`)
- **Test Fixes**: Corrected 10+ test files with module path updates and fixed malformed importlib calls
- **Validation**:
  - Smoke tests: **11/11 passing** ✅
  - Pipeline integration tests: **3/3 passing** ✅
  - Archive: `src/` backed up to `archive_local/src_backup_20260202/` (immutable)
- **Commits**: e6930f5 (module migration), a7d0362 (test fixes)
- **Breaking Changes**: Direct imports from `src.*` no longer work; use `dataselector.<subpackage>.*` instead

## [2026-01-19] - Configuration: persistent `n_samples`, tests & monitor robustness
- Removed hard-coded CLI default `34` for `--n-samples`. The selection size is now **config-driven** (set `selection.n_samples` in `config/pipeline_config.yaml`) or can be overridden on the CLI with `--n-samples`.
- Added unit test `tests/test_run_adaptive_default_n_samples.py` to ensure help text and behavior remain consistent.
- Fixed monitor run detection to be more robust across environments (improved handling of `outputs/runs/` discovery).
- Added automatic reconstruction of missing `results/trials.csv` from `optuna_study.db` in `scripts/xxl_full_run_monitor.py` (safe: PRAGMA integrity_check, atomic write, provenance `trials_reconstruct_meta.json`, opt-out via `--no-reconstruct`).
- Added safe resume/restart support to `scripts/xxl_full_run_monitor.py` (`--restart last|<run_id>`, `--force-restart`, `--dry-run-restart`): DB integrity checks, automatic DB backup (`optuna_study.db.bak_resume_<ts>`), compute remaining trials and resume Optuna via `scripts/optuna_optimize.py --n-trials <remaining>`, and write `results/resume_meta.json` for provenance. The resume flow was extended to perform staged resume: when Optuna is already complete the monitor can continue downstream phases (reproducibility, finalization), append logs, and record per-phase metadata. Added unit tests `tests/test_monitor_resume.py`.
- Added unit/integration tests (`tests/test_monitor_reconstruct.py`, extended `tests/test_monitor_robustness.py`) to validate reconstruction behavior.
- Added optional helper `scripts/merge_xxl_logs.py` for post-hoc consolidation/annotation of run logs (intentionally **not** integrated automatically into the monitor).

## [2026-01-16] - Run orchestration: detach flag
- Added `--detach` flag to `scripts/run_full_experiment.sh` to start experiments detached; writes PID and session log into `outputs/experiments/`.
- Added test `tests/test_run_full_experiment_detach.py` to verify detach behavior.
- Documented usage in `README.md` (Detached runs section) and updated `docs/ENV_SETUP.md`.

## [2026-01-11] - Final master grid fixes
- Fixed 28 sheets using final master_coords list.
- See `data/changes_applied.csv` and `data/duplikate_manual_followup.csv` for remaining manual cases.

## [2026-01-11] - Ultimative Master-Korrektur (David Rumsey Grid)
- Applied verified master_coords to 33 sheets across Masuren, Westpreußen, Bremen-Verden, Mittelkette, Bayern.
- Recomputed RIGHT/BOTTOM/N; geometry validation: 0 errors.
- Filename fix for Sheet 649 → KDR_649.PNG.
- Saved final dataset: data/all_png_tiles_final_ultimative.csv.
- Remaining position duplicates: 28 rows (14 pairs) enumerated for review (e.g., 661↔670, 662↔671, 422↔043, 397↔495, 285↔309, 253↔520, 134↔228, 103↔197, 095↔158, 029↔102, 009↔032, 019↔053, 003↔104, 001↔075).
- Next: add authoritative coords for the above "Hausbesetzer" to master_coords for full deconfliction.

## [2026-01-16] - Modernization: Sampling, Heuristics, UQ & CI
- Added Sobol (QMC) sampling and made sampler selectable (`--sampler lhs|sobol`) for Phase 1 exploration.
- Introduced `compute_adaptive_n_initial` with `modern` (dimension-aware) and `legacy` strategies; default set to `modern` (2*D^2 rule-of-thumb).
- Enabled Optuna sampler selection: `--sampler {tpe,qmc,cmaes}` in `scripts/optuna_optimize.py`.
- Added Deep Ensembles UQ (`scripts/uncertainty_quantification.py`) and `--uq-method ensemble` for faster uncertainty estimation.
- Added sampling benchmark script: `scripts/benchmark_sampling_methods.py` (LHS vs Sobol comparison).
- CI: installed optional deps for benchmarks/optuna & added separate `torch-tests` job using `requirements-cpu.txt`.
- Added a mamba-first environment helper (`scripts/create_env.sh`) and environment spec `environment.yml` for reproducible installs. ✅
- Added a canonical execution wrapper `scripts/exec_in_env.sh` (mamba -> conda -> .venv fallback), and integrated it into `run_full_experiment.sh` and `run_adaptive_pipeline.py` for consistent env usage. ✅
- Added CI smoke job `env-smoke-run` that runs a dry adaptive pipeline inside the created `dataselector` env, and `conda-lock` job to generate and upload lockfiles. ✅
- Added conda lockfile `locks/conda-lock-linux-64.lock` for reproducible installs on linux-64. ✅
- Updated README and added `docs/ENV_SETUP.md` with environment and reproducibility guidance. ✅
- Added `scripts/watch_experiment.sh` enhancements: automatic selection of the latest `outputs/experiments/run_*` directory, `--filter '<regex>'` for live filtering, `--show-proc` to display PID/%CPU/%MEM (no new dependencies), and `--lines N` to set initial tail lines.
- Added smoke test README: `tests/smoke/README.md` with quick start, watch examples and checklist.

## [2026-01-11] - DBF-Reset und Master-Fix
- Converted original DBF to CSV via dbfread, dropping true duplicates by BLATTNUMME.
- Normalized columns and applied ultimate MASTER_COORDS.
- Outputs: data/all_png_tiles_from_dbf.csv, data/all_png_tiles_final_from_dbf.csv
- Validation: Geometry errors = 0; Remaining position duplicates = 26 rows (13 Paare), v.a. Bayern (635/648, 636/649) und Ostpreußen (z.B. 052/053, 030/031, 018/019, 073/103, 101/102, 075/104, 106/137 etc.).

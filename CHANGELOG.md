# Changelog

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

# Changelog

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

## [2026-01-11] - DBF-Reset und Master-Fix
- Converted original DBF to CSV via dbfread, dropping true duplicates by BLATTNUMME.
- Normalized columns and applied ultimate MASTER_COORDS.
- Outputs: data/all_png_tiles_from_dbf.csv, data/all_png_tiles_final_from_dbf.csv
- Validation: Geometry errors = 0; Remaining position duplicates = 26 rows (13 Paare), v.a. Bayern (635/648, 636/649) und Ostpreußen (z.B. 052/053, 030/031, 018/019, 073/103, 101/102, 075/104, 106/137 etc.).

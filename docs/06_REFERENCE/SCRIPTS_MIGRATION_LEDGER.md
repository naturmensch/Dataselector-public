# Historical/Legacy (Not Normative)

This ledger is preserved for migration tracking and historical references.
Entries may reference script-era commands that are no longer canonical runtime
entrypoints.

Authoritative operational docs:

1. `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
2. `docs/03_USER_GUIDES/PIPELINES.md`
3. `README.md`

# Scripts Migration Ledger

This file tracks script ownership, usage, and migration status.

## Status legend
- **supported**: part of the official 3-phase thesis pipeline (see `07_ARCHIVE/docs_consolidation_2026-04-19/MODERNIZATION_PLAN_20260123_legacy.md` for historical context)
- **tools**: maintenance / diagnostics / repo tooling
- **analysis**: plotting / benchmarking / analysis
- **pipeline_experiments**: experimental pipelines / runners (not part of the official flow)
- **misc**: uncategorized

## Ledger

| Script | Category | Refs (docs+tests) | Canonical command (target) | Migration status | Notes |
|---|---:|---:|---|---|---|
| `scripts/exec_in_env.sh` | supported | 38 | micromamba run -n dataselector -- <command> | delegated compatibility layer | Keep as compatibility wrapper; not primary policy |
| `scripts/optuna_optimize.py` | supported | 17 | python -m dataselector optuna-optimize -- <args> | migrated (CLI wrapper added) | Keep script as runner until logic moved into `dataselector/` |
| `scripts/xxl_KDR146_run_thesis_complete_modern.py` | supported | 10 | archived / removed | historical only | Former XXL runner; no longer part of the active package surface |
| `scripts/xxl_full_run_monitor.py` | supported | 9 | archived / removed | historical only | Former XXL monitor helper; no longer part of the active package surface |
| `scripts/run_thesis_sampler_suite.py` | supported | 5 | python -m dataselector sampler-suite -- <args> | migrated (CLI wrapper added) | Keep script as runner until logic moved into `dataselector/` |
| `scripts/final_selection.py` | supported | 4 | python -m dataselector final-selection -- <args> | migrated (CLI wrapper added) | Keep script as runner until logic moved into `dataselector/` |
| `scripts/run_complete_thesis_pipeline.sh` | supported | 4 | python -m dataselector thesis-orchestrate -- <args> | migrated (compat wrapper only) | Keep as compatibility entrypoint; scientific core lives in dataselector/workflows |
| `scripts/optuna_autoscale.py` | supported | 3 | python -m dataselector autoscale -- <args> | migrated (CLI wrapper added) | Keep script as runner until logic moved into `dataselector/` |
| `scripts/compare_samplers_multi_seed.py` | supported | 2 | python -m dataselector compare-samplers -- <args> | migrated (CLI wrapper added) | Keep script as runner until logic moved into `dataselector/` |
| `scripts/benchmark_sampling_methods.py` | analysis | 5 | python -m dataselector benchmark-sampling -- <args> | migrated (CLI wrapper added) | Writes `outputs/exploration_plan_latest.json` (+ timestamped copy) |
| `scripts/run_sensitivity_sweep.py` | analysis | 0 | python -m dataselector sensitivity-sweep -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/scientific_tools.py` |
| `scripts/run_ablation_study.py` | analysis | 0 | python -m dataselector ablation-study -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/scientific_tools.py` |
| `scripts/compare_feature_backbones.py` | analysis | 0 | python -m dataselector compare-backbones -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/scientific_tools.py` |
| `scripts/validate_kmeans_clustering.py` | analysis | 0 | python -m dataselector validate-kmeans -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/scientific_tools.py` |
| `scripts/validate_umap_convergence.py` | analysis | 0 | python -m dataselector validate-umap -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/scientific_tools.py` |
| `scripts/snapshot_final_config.py` | tools | 0 | python -m dataselector snapshot-config -- <args> | migrated (thin wrapper) | Uses runtime snapshot contract from `dataselector/runtime/parameter_snapshot.py` |
| `scripts/compare_min_distance_policies.py` | analysis | 4 | python -m dataselector compare-min-distance-policies -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/script_convergence_tools.py` |
| `scripts/compare_seed_vs_unseed.py` | analysis | 5 | python -m dataselector compare-seed-vs-unseeded -- <args> | migrated (thin wrapper) | Delegates to central workflow command |
| `scripts/seed_benchmark.py` | analysis | 3 | python -m dataselector seed-benchmark -- <args> | migrated (thin wrapper) | Delegates to central workflow command |
| `scripts/profile_selection.py` | analysis | 2 | python -m dataselector profile-selection -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/script_convergence_tools.py` |
| `scripts/test_temporal_sensitivity.py` | analysis | 2 | python -m dataselector temporal-sensitivity-test -- <args> | migrated (thin wrapper) | Scientific core moved to `dataselector/workflows/script_convergence_tools.py` |
| `scripts/analyze_dataset.py` | analysis | 2 | python scripts/analyze_dataset.py --help | not started | |
| `scripts/plot_bootstrap_summary.py` | analysis | 2 | python scripts/plot_bootstrap_summary.py --help | not started | |
| `scripts/profile_selection.py` | analysis | 2 | python -m dataselector profile-selection --help | migrated (thin wrapper) | Legacy entry retained for compatibility wrapper invocation |
| `scripts/analyze_bootstrap_convergence.py` | analysis | 1 | python scripts/analyze_bootstrap_convergence.py --help | not started | |
| `scripts/analyze_optuna_convergence.py` | analysis | 1 | python scripts/analyze_optuna_convergence.py --help | not started | |
| `scripts/plot_seeded_vs_unseeded.py` | analysis | 1 | python scripts/plot_seeded_vs_unseeded.py --help | not started | |
| `scripts/benchmark_speed.py` | analysis | 0 | python scripts/benchmark_speed.py --help | not started | |
| `scripts/compare_seed_vs_unseed.py` | misc | 5 | python -m dataselector compare-seed-vs-unseeded --help | migrated (thin wrapper) | Legacy entry retained for compatibility wrapper invocation |
| `scripts/generate_reports.py` | misc | 5 | python -m dataselector generate-reports -- <args> | migrated (CLI wrapper added) | Used by XXL finalization |
| `scripts/tune_weights_and_run.py` | misc | 4 | python scripts/tune_weights_and_run.py --help | not started | |
| `scripts/build_new_all_tiles.py` | misc | 2 | python scripts/build_new_all_tiles.py --help | not started | |
| `scripts/compare_distance_methods.py` | misc | 2 | python scripts/compare_distance_methods.py --help | not started | |
| `scripts/debug_rasterio.py` | misc | 2 | python scripts/debug_rasterio.py --help | not started | |
| `scripts/xxl_KDR146_run_thesis_complete_OLD.py` | misc | 2 | python scripts/xxl_KDR146_run_thesis_complete_OLD.py --help | not started | |
| `scripts/compare_samplers_on_kdr100.py` | misc | 1 | python scripts/compare_samplers_on_kdr100.py --help | not started | |
| `scripts/monitor_state.py` | misc | 1 | python scripts/monitor_state.py --help | not started | |
| `scripts/quick_benchmark.py` | misc | 1 | python scripts/quick_benchmark.py --help | not started | |
| `scripts/recovery.py` | misc | 1 | python scripts/recovery.py --help | not started | |
| `scripts/auto_fix_env_usage.py` | misc | 0 | python scripts/auto_fix_env_usage.py --help | not started | |
| `scripts/compare_methods.py` | misc | 0 | python scripts/compare_methods.py --help | not started | |
| `scripts/count_archived_tiles.py` | misc | 0 | python scripts/count_archived_tiles.py --help | not started | |
| `scripts/multi_criteria_temporal_test.py` | misc | 0 | python scripts/multi_criteria_temporal_test.py --help | not started | |
| `scripts/pre_run_sampler_comparison.sh` | misc | 0 | python scripts/pre_run_sampler_comparison.sh --help | not started | |
| `scripts/validate_pareto_candidates.py` | pipeline_experiments | 5 | python scripts/validate_pareto_candidates.py --help | not started | |
| `scripts/bootstrap_pareto_candidates.py` | pipeline_experiments | 3 | python scripts/bootstrap_pareto_candidates.py --help | not started | |
| `scripts/optimize_selection.py` | pipeline_experiments | 3 | python scripts/optimize_selection.py --help | not started | |
| `scripts/seed_benchmark.py` | pipeline_experiments | 3 | python -m dataselector seed-benchmark --help | migrated (thin wrapper) | Legacy entry retained for compatibility wrapper invocation |
| `scripts/run_diverse_experiments.py` | pipeline_experiments | 2 | python scripts/run_diverse_experiments.py --help | not started | |
| `scripts/run_pipeline.py` | pipeline_experiments | 2 | python scripts/run_pipeline.py --help | not started | |
| `scripts/uncertainty_quantification.py` | pipeline_experiments | 2 | python scripts/uncertainty_quantification.py --help | not started | |
| `scripts/validate_pareto_candidates_seeded.py` | pipeline_experiments | 2 | python scripts/validate_pareto_candidates_seeded.py --help | not started | |
| `scripts/apply_bootstrap_best.py` | pipeline_experiments | 1 | python scripts/apply_bootstrap_best.py --help | not started | |
| `scripts/apply_optuna_best.py` | pipeline_experiments | 1 | python scripts/apply_optuna_best.py --help | not started | |
| `scripts/bootstrap_final_selection.py` | pipeline_experiments | 1 | python scripts/bootstrap_final_selection.py --help | not started | |
| `scripts/run_full_experiment.sh` | pipeline_experiments | 1 | python scripts/run_full_experiment.sh --help | not started | |
| `scripts/check_env.py` | tools | 4 | python scripts/check_env.py --help | not started | |
| `scripts/manage_archives.py` | tools | 3 | python scripts/manage_archives.py --help | not started | |
| `scripts/verify_archive.py` | tools | 2 | python scripts/verify_archive.py --help | not started | |
| `scripts/check_protected.py` | tools | 1 | python scripts/check_protected.py --help | not started | |
| `scripts/clean_workspace.py` | tools | 1 | python scripts/clean_workspace.py --help | not started | |
| `scripts/collect_test_metrics.py` | tools | 1 | python scripts/collect_test_metrics.py --help | not started | |
| `scripts/diagnose_environment.py` | tools | 1 | python scripts/diagnose_environment.py --help | not started | |
| `scripts/docs_link_autofix.py` | tools | 1 | python scripts/docs_link_autofix.py --help | not started | |
| `scripts/docs_link_fix_patterns.py` | tools | 1 | python scripts/docs_link_fix_patterns.py --help | not started | |
| `scripts/docs_link_patch.py` | tools | 1 | python scripts/docs_link_patch.py --help | not started | |
| `scripts/migrate_feature_cache_to_hash.py` | tools | 1 | python scripts/migrate_feature_cache_to_hash.py --help | not started | |
| `scripts/check_dependency_pins.py` | tools | 0 | python scripts/check_dependency_pins.py --help | not started | |
| `scripts/check_env_usage.py` | tools | 0 | python scripts/check_env_usage.py --help | not started | |
| `scripts/common.py` | tools | 0 | python scripts/common.py --help | not started | |
| `scripts/diagnose_env_isolation.py` | tools | 0 | python scripts/diagnose_env_isolation.py --help | not started | |
| `scripts/diagnose_shape_mismatch.py` | tools | 0 | python scripts/diagnose_shape_mismatch.py --help | not started | |
| `scripts/install_git_hooks.py` | tools | 0 | python scripts/install_git_hooks.py --help | not started | |
| `scripts/install_pytorch.sh` | tools | 0 | python scripts/install_pytorch.sh --help | not started | |
| `scripts/restore_archived_data.py` | tools | 0 | python scripts/restore_archived_data.py --help | not started | |

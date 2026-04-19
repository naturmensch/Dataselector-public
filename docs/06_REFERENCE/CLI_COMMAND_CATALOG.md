# CLI Command Catalog (Active)

This catalog lists the currently registered `python -m dataselector` commands.

Canonical runtime pattern:

```bash
micromamba run -n dataselector python -m dataselector <command> --help
```

## Primary Thesis Commands

| Command | Purpose |
|---|---|
| `thesis-orchestrate` | Scientific trigger-all orchestration for thesis pipeline |
| `thesis-pipeline` | Complete thesis pipeline (4 phases + optional handoff bundle) |
| `generate-monitor` | Generate report from canonical thesis run artifacts |
| `thesis-build-annotation-plan` | Build deterministic patch manifest and split contract |

## Width Calibration Commands

| Command | Purpose |
|---|---|
| `orchestrate-width-calibration` | End-to-end width calibration orchestration |
| `build-width-calibration-roads-source` | Build canonical repo-local roads source GeoPackage |
| `prepare-width-calibration` | Prepare deterministic measurement task queue |
| `measure-width-calibration` | Open interactive two-click measurement viewer |
| `summarize-width-calibration` | Summarize accepted measurements |
| `audit-width-calibration-sensitivity` | Run mask-level sensitivity audit |
| `sync-width-calibration-source` | Sync editable roads source into repo-local path |
| `render-width-calibration-debug-masks` | Render debug/test-only fixed-width masks |

## Selection and Optimization Commands

| Command | Purpose |
|---|---|
| `autoscale` | Staged Optuna runner with progressive refinement |
| `optuna-autoscale` | Run staged autoscale and write compute artifacts |
| `optuna-optimize` | Optuna hyperparameter optimization |
| `optuna-import` | Import Optuna trials from CSV into storage |
| `adaptive-auto` | Autoscale + adaptive pipeline orchestrator |
| `adaptive-pipeline` | Exploration -> fine -> Optuna -> bootstrap pipeline |
| `final-selection` | Run final selection from given parameters |
| `apply-optuna-best` | Apply best Optuna trial to pipeline config |
| `compare-samplers` | Multi-seed sampler comparison |
| `sampler-suite` | Alias command for thesis sampler suite |
| `thesis-sampler-suite` | Thesis-grade sampler evaluation suite |
| `bootstrap-final` | Bootstrap uncertainty quantification for final selection |
| `bootstrap-pareto` | Bootstrap uncertainty quantification for Pareto candidates |

## Scientific and Validation Commands

| Command | Purpose |
|---|---|
| `benchmark-sampling` | Benchmark initial sampling methods |
| `validate-kmeans` | Validate clustering metrics for KMeans/UMAP |
| `validate-umap` | Validate UMAP topology preservation metrics |
| `compare-backbones` | Compare DINOv2 and ResNet50 backbones |
| `sensitivity-sweep` | Run hyperparameter sensitivity analysis |
| `ablation-study` | Run diversity ablation study |
| `snapshot-config` | Create final config snapshot with provenance/hash |
| `profile-selection` | Profile selection modes and export profiling artifacts |
| `compare-min-distance-policies` | Compare candidate `min_distance_km` policies |
| `compare-seed-vs-unseeded` | Compare seeded vs unseeded selection behavior |
| `seed-benchmark` | Benchmark seeded deterministic mode against baseline |
| `temporal-sensitivity-test` | Test temporal-weight sensitivity |

## Reporting and Audit Commands

| Command | Purpose |
|---|---|
| `generate-experiment` | Generate experiment report from run directory |
| `generate-thesis` | Generate thesis-specific report |
| `generate-thesis-final` | Generate final thesis report |
| `repo-evolution-audit-v3` | Generate final V3 repository evolution audit |
| `repo-evolution-audit-v4` | Generate complete repository evolution audit |

## Administrative Tools Commands

| Command | Purpose |
|---|---|
| `check-geo` | Validate geospatial dependency stack |
| `check-env` | Check environment usage policy |
| `check-protected` | Verify protected path modifications |
| `check-runtime-readiness` | Validate canonical runtime readiness |
| `check-script-wrappers` | Enforce wrapper/import policy |
| `align-audit` | Audit CSV vs raster alignment |
| `clean-workspace` | Cleanup workspace artifacts |
| `docs-link-check` | Validate documentation links |
| `docs-link-autofix` | Attempt automatic broken-link fixes |
| `archive-outputs` | Archive outputs directory |
| `list-archives` | List available archives |
| `verify-archive` | Verify archive reference hygiene |

## Data Command

| Command | Purpose |
|---|---|
| `build-tiles` | Build `new_all_tiles.csv` from image scan |

## Practical Discovery

List all available commands:

```bash
micromamba run -n dataselector python -m dataselector --help
```

Inspect a concrete command:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate --help
micromamba run -n dataselector python -m dataselector measure-width-calibration --help
```

## Notes

1. This catalog documents command availability and intent.
2. Detailed workflow semantics remain in user guides under
   [docs/03_USER_GUIDES/](../03_USER_GUIDES/).
3. Governance and contract constraints remain authoritative in
   [docs/08_GOVERNANCE/](../08_GOVERNANCE/).

# Thesis Pipeline How-To (Authoritative)

This is the authoritative runbook for thesis-grade tile selection before
annotation.

## 0) Official Pipeline Roles

1. `thesis-pipeline` is the production path for thesis/annotation decisions.
2. `adaptive-pipeline` is the advanced research path (deeper manual controls and
   diagnostics).
3. `adaptive-auto` is a convenience orchestrator around autoscale + adaptive.

Use `thesis-pipeline` unless you explicitly need research-grade experimentation.

## 1) Preconditions

1. Use the authoritative environment:

```bash
micromamba run -n dataselector python -m dataselector --help
```

Compatibility wrapper (optional):

```bash
./scripts/exec_in_env.sh --env dataselector -- python -m dataselector --help
```

2. Keep private images local-only (never commit image assets).
3. Keep spatial hard-cut contract (`ul_x/ul_y/lr_x/lr_y`), no legacy `N` path.

### 1.1 Interpreter Mismatch Quick Fix

If `python -m dataselector ...` fails locally with
`No module named dataselector`, you are likely using the wrong interpreter.

```bash
which python
python -c "import sys; print(sys.executable)"
micromamba run -n dataselector python -c "import dataselector; print(dataselector.__file__)"
```

For all thesis gates, run commands explicitly with:

```bash
micromamba run -n dataselector python
```

## 2) Required Runtime Gates

```bash
export RUN_FULL_INTEGRATION=1
export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images
```

## 3) Sampler Decision (Before Production Run)

Use the decision rule:

- `docs/status/thesis_sampler_decision_rule_2026-02-08.md`

This determines:

1. exploration sampler (`lhs` or `sobol`)
2. Optuna sampler (`TPESampler`, `QMCSampler`, `CmaEsSampler`)
3. fixed `n_initial_final` and run seed.

### 3.1 Optional evidence: Hamburg-seeded vs unseeded/random baseline

If Hamburg is already annotated and you want to justify starting with Hamburg as a
seed, create an explicit comparison artifact:

```bash
micromamba run -n dataselector python - <<'PY'
from pathlib import Path
from dataselector.workflows.compare_samplers import benchmark_seed

out = benchmark_seed(
    seeds=[42, 43, 44, 45, 46],
    subset_n=50,
    output_dir=Path("outputs/seed_vs_unseed_validation"),
)
print(out)
PY
```

Interpretation target:

1. compare `n_selected`
2. compare diversity/spatial metrics
3. compare objective values over multiple seeds (`mean`, `std`, worst-seed)
4. document whether Hamburg-seeding changes outcome materially or only marginally

### 3.2 Spatial Distance Policy (Current Thesis Standard)

1. Geometric reference from canonical metadata (`median_nn`) is currently `45.0 km`.
2. Operational thesis policy is currently `min_distance_km = 28.5`.
3. Rationale: decision-gate comparison (`28.5/40.0/45.0`) showed equivalent quality
   for `28.5` and `40.0`; tie-break prefers the smaller distance to keep more
   downstream combination space while preserving hardcut feasibility.

Evidence:

- `reports_2026-02-09/MIN_DISTANCE_DECISION_2026-02-09.md`
- `reports_2026-02-09/min_distance_policy_summary_20260209T233849Z.md`

## 4) End-to-End Thesis Flow (Copy/Paste)

### 4.0 Trigger-All Orchestration (Recommended)

Use the canonical orchestrator to enforce scientific precompute -> snapshot ->
validated run in one flow:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/thesis_orchestrated_$(date -u +%Y%m%dT%H%M%SZ)
```

This command performs:

1. precompute artifacts under `parameter_resolution/`
2. resolver snapshot stage (`--compute-params --snapshot-config --no-auto-continue`)
3. snapshot + contract validation
4. production run via `--use-params <final_config.yaml>`

### 4.0.1 Scientific Hardening Flags (v5)

Use these flags when you need explicit scientific contracts:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --cache-mode read_write \
  --strict-evidence-root run_dir \
  --strict-real-data true \
  --output-dir outputs/runs/thesis_orchestrated_$(date -u +%Y%m%dT%H%M%SZ)
```

Flag semantics:

1. `--cache-mode`: `off|read_only|write_only|read_write`.
2. `--strict-evidence-root`: evidence lookup scope (`run_dir` recommended).
3. `--strict-real-data`: forbids synthetic autoscale fallback in production.
4. `--force` requires `--force-override-reason "<reason>"` and is recorded in metadata.
5. `--tile-exclusion-policy` applies explicit tile removal rules before split/audit.
6. `--split-policy` controls leakage calibration and split ratios.
7. `--leakage-buffer-km auto|<value>` selects auto-calibration or fixed leakage buffer.
8. `--build-splits auto|true|false` controls split/audit artifact generation.

### 4.0.1.1 Leakage-Safe Split Artifacts

When split building is enabled (default in `thesis_repro`), the orchestrator writes:

1. `policy/distance_policy.json`
2. `policy/leakage_calibration.csv`
3. `splits/split_manifest.json`
4. `splits/leakage_audit.csv`

Scientific rules:

1. edge-to-edge distance in metric CRS (`EPSG:25832`) is used for leakage checks.
2. connected components under `edge_distance_km < d_leak` are not split across train/val/test.
3. in `thesis_repro`, any inter-split leakage violation triggers fail-fast.

Sampler evidence for contract validation is persisted run-locally at:
`outputs/runs/<run_id>/parameter_resolution/sampler_resolution/selected_sampler.json`.

### 4.A Parameter Resolution + Snapshot Contract

The canonical thesis path now supports explicit resolver/snapshot flags:

1. `--compute-params`: compute unresolved parameters and persist provenance.
2. `--snapshot-config`: write `final_config_<timestamp>.yaml` into run output.
3. `--use-params <snapshot.yaml>`: load + validate snapshot before execution.
4. `--no-auto-continue`: stop after resolution/snapshot stage.
5. `--force`: continue despite validation mismatch (flagged in `run_metadata.json`).

Feature-extraction provenance is pinned in the snapshot for scientific traceability:
1. `feature_extraction.model`
2. `feature_extraction.model_variant`
3. `feature_extraction.dinov2_repo`
4. `feature_extraction.dinov2_ref`
5. `feature_extraction.pooling`
6. `feature_extraction.preprocess_pipeline_id`

Feature cache identity is validated against:
`model_name`, `model_variant`, `dinov2_repo`, `dinov2_ref`, `pooling`,
`input_size`, `crop_size`, `preprocess_pipeline_id`, optional `config_sha256`.

Example (resolution-only preflight):

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --compute-params \
  --snapshot-config \
  --no-auto-continue \
  --output-dir outputs/runs/thesis_resolution_preflight
```

### 4.B Scientific Helper Commands (CLI-First)

Scientific diagnostics are now centralized as CLI commands (no scientific core
logic in top-level scripts):

```bash
micromamba run -n dataselector python -m dataselector sensitivity-sweep --config config/pipeline_config.yaml --variation-percent 20
micromamba run -n dataselector python -m dataselector ablation-study --config config/pipeline_config.yaml
micromamba run -n dataselector python -m dataselector compare-backbones --config config/pipeline_config.yaml
micromamba run -n dataselector python -m dataselector validate-kmeans --config config/pipeline_config.yaml
micromamba run -n dataselector python -m dataselector validate-umap --config config/pipeline_config.yaml
micromamba run -n dataselector python -m dataselector snapshot-config --config config/pipeline_config.yaml --output-dir outputs/runs
```

Legacy script names still exist as compatibility wrappers and delegate to these
CLI commands.

### 4.0 Selection Target Contract (No Silent Fallbacks)

`n_samples` resolution in production is:

1. explicit `--n-samples`
2. `config/pipeline_config.yaml` -> `selection.n_samples`
3. autoscale artifact (`autoscale_selected_n_samples.txt` or `optuna_autoscale_selected_n_samples.txt`)
4. fail-fast (error)

For thesis runs, pass `--n-samples` explicitly in commands below.
Current thesis policy baseline is `24` (minimum-sufficient decision):
`reports_2026-02-09/N_SAMPLES_DECISION_2026-02-09.md`.

Note on autoscale `full` stage:
When a stage uses full candidate coverage (`n_samples == total candidates`), the
autoscale objective enforces `min_distance_km = 0` for that stage to avoid
cardinality infeasibility artifacts. This stage is diagnostic and should not be
interpreted as the production min-distance policy by itself.

Additional production rule:
`diagnostic_only=true` stages are excluded from final production parameter
selection in hardening profile.

### 4.0.2 CRS and Distance Strictness

In `thesis_repro` profile, metric distance context is enforced:

1. target metric CRS defaults to `EPSG:25832`
2. unknown/unresolved CRS is treated as hard error
3. run metadata must include: `source_crs`, `metric_crs`, `transform_applied`
4. geometric min-distance references (`compute_min_distance_km`) are computed
   from metric projected coordinates, not from raw Web-Mercator display units

### 4.1 Build metadata from local images

```bash
micromamba run -n dataselector python -m dataselector build-tiles \
  --image-dir "$DATASELECTOR_IMAGE_DIR" \
  --name-source-csv data/KDR100_foliage_with_files_epsg3857.csv \
  --city-overrides data/city_overrides.csv \
  --out data/new_all_tiles.csv
```

If canonical city resolution is incomplete, `build-tiles` now fails fast and tells
you to update `data/city_overrides.csv` for the remaining IDs.

### 4.2 Validate raster/CSV alignment

```bash
micromamba run -n dataselector python -m dataselector align-audit \
  --csv data/new_all_tiles.csv \
  --base-dir "$DATASELECTOR_IMAGE_DIR" \
  --out outputs/align_audit.json
```

### 4.3 Dry-run (safety check)

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --dry-run \
  --output-dir outputs/runs/thesis_preflight
```

### 4.4 Production run A (quick gate)

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --output-dir "outputs/runs/thesis_run_A_${RUN_TAG}"
```

### 4.5 Production run B (determinism check, quick gate)

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --output-dir "outputs/runs/thesis_run_B_${RUN_TAG}"
```

### 4.6 Optional: Hamburg as start tile (separate output, no overwrite)

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --hamburg \
  --output-dir "outputs/runs/thesis_run_hamburg_${RUN_TAG}"
```

Equivalent explicit form:

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 24 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --pre-names Hamburg \
  --output-dir "outputs/runs/thesis_run_hamburg_explicit_${RUN_TAG}"
```

`--hamburg` and `--pre-names Hamburg` are intentionally equivalent.

### 4.7 Hamburg/City Resolution Contract (Authoritative)

Preselection name resolution follows this chain:

1. `shortName` exact match.
2. `city` exact match (case-insensitive).
3. `longName` substring match.
4. Stable alias shortcut: `Hamburg -> KDR_146`.

Data provenance for `city`:

1. For canonical `build-tiles` output (`data/new_all_tiles.csv`), the builder
   loads legacy name metadata from `data/KDR100_foliage_with_files_epsg3857.csv`
   when available.
2. `city` is derived from `longName` pattern (`KDR_<id>_<city>_<year>.png`) if
   not already present in the source.
3. Key variants are normalized (`KDR_079a` vs `KDR_079A`, optional letter
   suffixes like `KDR_155b -> KDR_155`) before matching source rows.
4. If still unresolved, deterministic backup fill is applied from the best
   matching `new_all_tiles.backup_*.csv` in `data/`.
5. Final unresolved rest cases are filled from `data/city_overrides.csv`
   (`manual_override`), and every row gets a trace in `city_source`
   (`longname_parse`, `variant_base`, `backup_fill`, `manual_override`).
6. Stable alias fallback remains: `Hamburg -> KDR_146`.

This guarantees that `--hamburg` and `--pre-names Hamburg` resolve to the
documented anchor tile as long as canonical metadata remains unchanged.

## 5) Required Validation Gates

### 5.1 E2E and real-image tests

```bash
micromamba run -n dataselector python -m pytest -q -m e2e -rs
micromamba run -n dataselector python -m pytest -q -m real_images -rs
```

### 5.2 Guard test

```bash
micromamba run -n dataselector python -m pytest -q \
  tests/unit/test_no_legacy_script_references.py
```

### 5.3 Determinism gate

The annotation gate is only open when:

1. timestamped run dirs for A and B both exist (e.g. `outputs/runs/thesis_run_A_<RUN_TAG>`, `outputs/runs/thesis_run_B_<RUN_TAG>`).
2. selected tile IDs are identical in A and B.
3. `run_metadata.json` exists in run outputs and includes:
   - commit SHA
   - command
   - seed
   - execution profile
4. sampled `image_path` entries resolve to real files.

### 5.4 Quick vs Full validation mode

`quick_gate` (default recommendation for iterative checks):

- `--validation-seeds 42`
- `--validation-min-distances 28.5`

`full_gate` (final robustness evidence):

- e.g. `--validation-seeds 42 43 44 45 46`
- e.g. `--validation-min-distances 25 35 50`

## 6) Go/No-Go Before Annotation

Start annotation only if all are true:

1. Latest `main` core workflows are green.
2. Feature-gap scan reports no `missing_required` and no `implemented_but_misaligned`.
3. Gates in section 5 pass.
4. No open blocking PR/issue.

If a gate fails: open one focused fix branch per root cause. Do not use broad
skip/xfail or `continue-on-error` expansion.

## 7) If report shows `0/60 non-empty`

`0/60 non-empty` means: in 60 validated parameter configurations, each had
`n_selected == 0` in validation. It does **not** automatically prove the whole
pipeline selected nothing in earlier phases.

Quick diagnosis:

```bash
# 1) Inspect raw validation outcomes
micromamba run -n dataselector python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/runs/thesis_run_A_<RUN_TAG>/validation/validation_results.csv")
print(df[["n_selected"]].describe(include="all"))
print(df.head(10))
PY

# 2) Re-run validation with explicit settings to test sensitivity
micromamba run -n dataselector python - <<'PY'
from dataselector.workflows.validation import validate_pareto_candidates
print("validate_pareto_candidates callable:", callable(validate_pareto_candidates))
PY
```

Check especially:

1. `min_distance_km` strictness
2. preselection constraints
3. candidate subset sizes per validation run

## 8) Long-Run Observation Rule (>10 min)

For long-running checks and pipelines:

1. Start run and share one command/run link.
2. Do not poll continuously in chat.
3. Continue only after explicit user feedback (`fertig`, `gruen`, `fehlgeschlagen`).
4. Store logs under `outputs/run_logs/` with UTC timestamp suffixes.

Example:

```bash
mkdir -p outputs/run_logs
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --output-dir "outputs/runs/thesis_run_A_${RUN_TAG}" \
  | tee "outputs/run_logs/thesis_run_A_${RUN_TAG}.log"
```

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

### 3.1 Optional supplementary diagnostics: Hamburg-seeded vs unseeded/random baseline

If Hamburg is already annotated and you want to characterize seed behavior, create
an explicit comparison artifact:

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
5. treat this as supplementary diagnostics only; primary thesis claims remain
   `Core-only` under the Core+Case contract

### 3.2 Spatial Distance Policy (Current Thesis Standard)

1. Geometric reference from canonical metadata (`median_nn`) is currently `45.0 km`.
2. Operational thesis policy is currently `min_distance_km = 28.5`.
3. `40.0 km` was the explicit intermediate comparison candidate in the
   pre-registered decision panel.
4. Rationale: decision-gate comparison (`28.5/40.0/45.0`) showed equivalent quality
   for `28.5` and `40.0`; tie-break prefers the smaller distance to keep more
   downstream combination space while preserving hardcut feasibility.
5. Free optimization into the very small-distance regime was deliberately not
   adopted as thesis production policy: supplementary historical full runs under
   otherwise fixed profile resolved to `5 km` and `8 km`, indicating that the
   objective can drift into a permissive low-distance region without an
   immediate feasibility penalty.
6. Interpretation: the spatial objective remains active, but the hard floor is
   kept as a conservative production admissibility guard.

Evidence:

- `reports_2026-02-09/MIN_DISTANCE_DECISION_2026-02-09.md`
- `reports_2026-02-09/min_distance_policy_summary_20260209T233849Z.md`
- `docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md`

## 4) End-to-End Thesis Flow (Copy/Paste)

### 4.0 Trigger-All Orchestration (Recommended)

Use the canonical orchestrator to enforce scientific precompute -> snapshot ->
validated run in one flow. Default production policy: `n_trials = 370`, `selection.optuna_sampler = tpe` (see `docs/PARAMETER_POLICY_LEDGER.md`).

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/thesis_orchestrated_$(date -u +%Y%m%dT%H%M%SZ)
```

> Tip: do **not** insert an extra `--` between `micromamba run -n dataselector` and the command. Use `micromamba run -n dataselector python -m dataselector <command>` — some micromamba/shell combinations fail to forward CLI args when `--` is used.
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
8. `--build-splits auto|true|false` controls split/audit artifact generation (default: `false`).
9. autoscale `min_distance_km` bounds are policy-driven via
   `selection.autoscale_min_distance_floor_km|ceiling_km|global_search` in
   `config/pipeline_config.yaml`.
10. `--build-handoffs` runs the optional post-freeze packaging bundle (tile handoff, annotation plan, patch handoff). Default: `false`.
11. `--patches-per-tile` controls integrated patch-plan density when `--build-handoffs` is enabled.
12. `--patch-include-case true|false` controls whether patch packaging uses the final core+case set or only the core set. Default: `false` (`core-only`).
13. `--handoff-root` changes the root output directory for integrated handoff bundles (default: `handoff`).

### 4.0.1.0 Optional Phase 5: Annotation Plan + Handoff Bundle

If you want one canonical run directory plus packaging artifacts in one flow, add:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/thesis_orchestrated_$(date -u +%Y%m%dT%H%M%SZ) \
  --build-handoffs \
  --patches-per-tile 2 \
  --patch-include-case false
```

Scientific boundary:

1. Phase 5 is **post-freeze operational packaging**, not reselection.
2. Default integrated patch scope is `core-only`.
3. Snapshot and `selection_*` artifacts remain the authoritative frozen dataset.
4. If Phase 5 fails, the run fails operationally, but the scientific freeze boundary must remain unchanged.

### 4.0.1.2 Global n_samples Corridor Policy (Core)

`n_samples` autoscale staging is resolved in the core workflow
(`dataselector/workflows/optuna_autoscale.py`), not in helper scripts.

Policy keys in `config/pipeline_config.yaml`:

1. `selection.autoscale_n_samples_mode: corridor|fixed`
2. `selection.autoscale_n_samples_fixed`
3. `selection.autoscale_n_samples_corridor_min_pct`
4. `selection.autoscale_n_samples_corridor_target_pct`
5. `selection.autoscale_n_samples_corridor_max_pct`
6. `selection.autoscale_n_samples_step` (default `1`: every integer in corridor)
7. `selection.autoscale_n_samples_corridor_min_abs`
8. `selection.autoscale_n_samples_corridor_max_abs`
9. `selection.autoscale_n_samples_plateau_delta`

Selection rule:

1. build stage panel from `N_eff` (effective candidates after exclusions),
2. keep only feasible stages (`infeasible=false`, hard target met),
3. choose smallest `n` within plateau:
   `score(n) >= best_score * (1 - plateau_delta)`.

Methodological interpretation:

1. thesis objective is a core-set style annotation strategy: minimize annotation
   load while preserving selection quality.
2. default corridor is centered around `5%` of `N_eff` with bounded exploration
   in `4-8%` to make the chosen sample size robust against local noise.
3. default `step=1` evaluates every integer `n` in the bounded corridor for a
   transparent minimum-annotation decision.
4. the plateau rule enforces "minimum n with near-optimal score" instead of
   blindly maximizing score at larger `n`.
5. this is a selection contract, not an evaluation split contract; if splitting
   is done by a downstream training tool, keep `--build-splits false`.
6. supplementary downstream-model evidence is consistent with this policy:
   foundation-model-based approaches such as MapSAM and historical-map few-shot
   segmentation make the `~5%` center look plausible rather than implausibly
   small, while UNet++ acts as the conservative anchor that justifies keeping
   the upper side of the corridor.
7. this architecture-specific evidence is supplementary only; the Dataselector
   policy remains model-agnostic and is not directly derived from
   SegFormer/MapSAM/UNet++ result papers.

Artifacts:

1. `parameter_resolution/optuna_autoscale_stage_policy.json`
2. `parameter_resolution/optuna_autoscale_best_latest.json`
3. `parameter_resolution/optuna_autoscale_selected_n_samples.txt`
4. `docs/N_SAMPLES_EVIDENCE_ADDENDUM.md`

### 4.0.1.1 Leakage-Safe Split Artifacts

When split building is enabled (`--build-splits true` or explicit `auto`), the orchestrator writes:

1. `policy/distance_policy.json`
2. `policy/leakage_calibration.csv`
3. `splits/split_manifest.json`
4. `splits/leakage_audit.csv`

Scientific rules:

1. edge-to-edge distance in metric CRS (`EPSG:25832`) is used for leakage checks.
2. connected components under `edge_distance_km < d_leak` are not split across train/val/test.
3. in `thesis_repro`, any inter-split leakage violation triggers fail-fast.

### 4.0.1.3 Responsibility Boundary: Selection vs. Split

For thesis workflows, responsibilities are intentionally separated:

1. `dataselector thesis-orchestrate` selects the global annotation candidate set
   (core-set selection contract).
2. split generation and split leakage enforcement are optional in Dataselector and
   controlled by `--build-splits`.
3. if your downstream training tool performs the authoritative train/val/test split
   and leakage checks, keep `--build-splits false` in Dataselector runs.
4. document this boundary explicitly in the thesis methods section as:
   `Selection contract (Dataselector)` vs. `Evaluation contract (training tool)`.

### 4.0.1.5 Core+Case Contract (Hamburg als Zusatzfall)

Für thesis-grade Vergleichbarkeit gilt:

1. Core-Selektion ist die primäre wissenschaftliche Stichprobe.
2. Case-Tiles (z. B. Hamburg) werden separat geführt und erst danach angehängt.
3. Primärmetriken sind als `Core-only` zu interpretieren.

Relevante Config-Keys:

1. `selection.case_tile_names`
2. `selection.case_exclude_from_core` (empfohlen: `true`)
3. `selection.case_attach_mode` (`append_unique|append_all`)

Aktueller Default in `config/pipeline_config.yaml`:
`selection.case_tile_names: ["Hamburg"]`.

Erwartete Run-Artefakte:

1. `selection_core.csv`
2. `selection_case.csv`
3. `selection_final_with_cases.csv`
4. `selection_contract.json`

Interpretation des aktuellen Thesis-Freezes:

Dataset authority: `selection_core.csv`, `selection_final_with_cases.csv`, und
`selection_contract.json` definieren den eingefrorenen Thesis-Datensatz.

Parameter authority: validierter Snapshot und Parameter-Resolution-Artefakte
definieren den aufgelösten Parameterkontext.

1. `selection_contract.json` muss `selection_source` und
   `selection_source_file` enthalten, damit die materialisierte Auswahlquelle
   explizit nachvollziehbar bleibt.
2. Thesis-Default ist `selection.selection_authority: snapshot_primary`.
   Die Core-Auswahl wird direkt aus den validierten Snapshot-Parametern
   materialisiert.
3. Legacy-Kompatibilität bleibt über
   `selection.selection_authority: materialized_csv_primary` erhalten.
   Dieser Modus ist nur für historische Runs gedacht.
4. `selection.objective_authority: unified_normalized` ist der Thesis-Default.
   Exploration und Autoscale nutzen damit dieselbe Objective-Definition.
5. Für neue kanonische v2-Runs ist `Selection Reconciliation = aligned`
   der erwartete Status.
6. `documented_difference` ist nur in Legacy-/Historienkontexten zulässig und
   muss explizit im Report begründet sein.
7. Die Freeze-Selektion ist architektur-neutral / model-agnostic.
8. Der Freeze ist ein `frozen dataset`; Modellvergleiche erfolgen danach.
9. No direct model-metric optimization (SegFormer/MapSAM/UNet++).
10. `alpha_visual` ist ein optimierter Parameter, aber keine harte
    Dominanzbedingung.
11. Visual-biased oder model-aware Selektion ist ein separater
    Ablationspfad und erfordert einen neuen Freeze.

### 4.0.1.6 Validation/UQ Contract

Für inferenzielle Unsicherheitsaussagen:

1. `validation.replicate_mode: bootstrap_candidates`
2. `validation.n_bootstrap` (Default: `200`)
3. `validation.bootstrap_sample_frac` (Default: `1.0`)

`seed_replay` bleibt als deterministischer Replay-Check erhalten, ist aber kein
Ersatz für unabhängige inferenzielle Replikation.

Erwartete Validation-Artefakte:

1. `validation/validation_method_contract.md`
2. `validation/validation_results_bootstrap.csv`
3. `validation/validation_summary_stats.csv`

Sampler evidence for contract validation is persisted run-locally at:
`outputs/runs/<run_id>/parameter_resolution/sampler_resolution/selected_sampler.json`.

### 4.0.1.4 Sampler Resolution Contract (Critical Interpretation)

The thesis orchestrator intentionally separates sampler decision from production
execution:

1. resolution stage runs with `--compute-params` and resolves the sampler in
   this order: explicit policy -> existing artifact -> multi-seed auto-compare.
2. when auto-compare is used, benchmark evidence is written under
   `parameter_resolution/sampler_resolution/` (including `summary.csv` and
   `selected_sampler.json`).
3. the resolved sampler is frozen into the resolution snapshot
   (`final_config_<timestamp>.yaml`) with parameter provenance.
4. production stage runs from `--use-params <snapshot.yaml>` and applies that
   frozen value.
5. therefore `resolved_sampler_source=config_policy` in production metadata is
   expected and means snapshot application, not sampler re-selection.

Scientific implication:

1. this preserves methodological comparability across A/B reproducibility runs.
2. sampler choice is benchmarked once per policy context and then fixed.
3. re-evaluate sampler only when data domain, constraints, feature extractor, or
   trial budget changes materially.

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
2. `thesis_repro` requires explicit source CRS provenance from sidecars/raster
   metadata; heuristic inference is fallback-only outside strict thesis mode
3. unknown/unresolved or only-heuristic CRS is treated as hard error
4. run metadata must include: `source_crs`, `metric_crs`, `transform_applied`,
   `crs_provenance_status`, `crs_provenance_audit_path`
5. thesis runs emit `data_quality/crs_provenance_audit.csv`
6. geometric min-distance references (`compute_min_distance_km`) are computed
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

## 9) Handoff to masterarbeit-strassenerkennung (Server Workflow)

Dataselector liefert die finale Auswahl. Train/Val/Test-Splits bleiben im
Trainings-Repo autoritativ (`split_authority = masterarbeit_strassenerkennung_cv`).

### 9.1 Handoff-Artefakte erzeugen

Für den aktuellen patch-basierten Thesis-Datensatz (z. B. `29x2`) ist der
empfohlene Flow:

```bash
bash scripts/handoff_check.sh prepare-patches \
  --run-dir outputs/runs/<run_id> \
  --out handoff/<selection_id>

bash scripts/handoff_check.sh verify-patches \
  --handoff-dir handoff/<selection_id>
```

`verify-patches` prüft:

1. Handoff-Schema (`selected_patches.csv`, `patch_handoff_manifest.json`, `patch_mask_requirements.csv`, `patch_split_manifest.json`)
2. Konsistenz von Patch-Bounds/Fold-Zuordnung (`patch_to_fold`)
3. Bildverfügbarkeit unter `data/images` (Quell-Tiles)
4. PNG-Sidecars (`*.png.aux.xml`) für Quell-Tiles
5. Patch-Quicklooks als GeoTIFF im Handoff (`quicklooks/*.tif`) inkl. eingebetteter Georeferenz (CRS + Transform)
6. Exclusion-Policy-Verstöße auf Tile-Ebene

Warum der Patch-Handoff auf GeoTIFF-v2 standardisiert ist:

1. Georeferenz liegt im Quicklook selbst (kein fragiles Sidecar-Only-Modell).
2. VRT-basierte Annotation in QGIS ist stabiler und reduziert manuelle Patch-Auswahl.
3. Der Vertrag ist maschinell erzwingbar (`handoff_patch_format_v2`, `geotiff_deflate_rgb`).
4. Der Speicheraufpreis bleibt im niedrigen einstelligen Bereich und ist für den Workflow vertretbar.

Der tile-basierte Legacy-Flow bleibt verfügbar:

```bash
bash scripts/handoff_check.sh prepare \
  --run-dir outputs/runs/<run_id> \
  --out handoff/<selection_id>

bash scripts/handoff_check.sh verify-local \
  --handoff-dir handoff/<selection_id>
```

`verify-local` prüft:

1. Handoff-Schema (`selected_maps.csv`, `handoff_manifest.json`, `mask_requirements.csv`)
2. Bildverfügbarkeit unter `data/images`
3. PNG-Sidecars (`*.png.aux.xml`)
4. Exclusion-Policy-Verstöße (z. B. ausgeschlossene Sonderkacheln)

Exit-Codes:

1. `0`: alles ok
2. `2`: Schema-/Manifestfehler
3. `3`: Bilddaten unvollständig
4. `4`: Policy-Verstoß

### 9.2 Transfer lokal -> Server

```bash
# Handoff-Ordner übertragen
rsync -avh handoff/<selection_id>/ <server>:/path/to/handoff/<selection_id>/

# Nur geforderte Masken übertragen (patch-basiert)
cut -d, -f2 handoff/<selection_id>/patch_mask_requirements.csv | tail -n +2 > /tmp/mask_files.txt
rsync -avh --files-from=/tmp/mask_files.txt annotations/masks/ <server>:/path/to/masks/
```

Für den tile-basierten Legacy-Flow stattdessen:

```bash
cut -d, -f2 handoff/<selection_id>/mask_requirements.csv | tail -n +2 > /tmp/mask_files.txt
```

Falls Rohkarten auf dem Server fehlen: zusätzlich selektierte Bilder inklusive
Sidecars übertragen.

### 9.3 Server-Check im Trainings-Repo

Im Repo `masterarbeit-strassenerkennung`:

```bash
bash scripts/setup/handoff_check.sh verify-server \
  --handoff-dir /path/to/handoff/<selection_id> \
  --raw-tiles-dir /path/to/raw/Tiles \
  --masks-dir /path/to/masks

bash scripts/setup/handoff_check.sh materialize \
  --handoff-dir /path/to/handoff/<selection_id> \
  --out-root data/integration
```

Erst nach grünem `verify-local` + `verify-server` Training/Splits starten.

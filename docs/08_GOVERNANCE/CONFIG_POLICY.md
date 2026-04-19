# Config Policy (Authoritative)

## Active vs Historical

1. Active default config for production/thesis runs is `config/pipeline_config.yaml`.
2. Historical/reference configs (forensics, prior experiments) must not be treated as defaults.
3. Historical configs must include a visible header note stating non-default status.

## Override and Precedence

For `dataselector thesis-pipeline` the effective configuration resolves in this order:

1. Explicit CLI flags
2. `--use-params <snapshot.yaml>` (validated; blocks on mismatch unless `--force`)
3. Active config (`config/pipeline_config.yaml`)
4. Resolver artifacts only where explicitly contracted (e.g. selection target resolver)
5. Fail-fast (no silent fallback)

## Parameter Source Contract

Every critical parameter in the thesis path must be either:

1. computed (`method: computed_*`), or
2. explicit policy/manual (`method: policy|manual|config_policy|snapshot_policy`).

Snapshot/provenance metadata must be written for traceability.
Sampler resolution evidence for thesis runs must be run-local under:
`outputs/runs/<run_id>/parameter_resolution/sampler_resolution/selected_sampler.json`.
- See `../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md` for the authoritative ledger of parameter defaults and the scientific rationale (e.g. `n_trials = 370`, pinned sampler policy).
Model provenance for DINOv2 must include variant/repo/ref/pooling in snapshot metadata.

Core+Case selection parameters are policy-relevant and must be explicit:

1. `selection.case_tile_names`
2. `selection.case_exclude_from_core`
3. `selection.case_attach_mode`
4. `selection.selection_authority`
5. `selection.objective_authority`

Current default policy keeps `selection.case_tile_names: ["Hamburg"]` as a
separate Case tile while `selection.case_exclude_from_core=true`.

Validation/UQ mode must be explicit:

1. `validation.replicate_mode` (`bootstrap_candidates` for inferential UQ)
2. `validation.n_bootstrap`
3. `validation.bootstrap_sample_frac`

## Selection Authority Contract

Dataset authority: `selection_core.csv`, `selection_final_with_cases.csv`, and
`selection_contract.json` define the frozen thesis dataset.

Parameter authority: validated snapshot and parameter-resolution artifacts
define the resolved parameter context.

1. `selection.selection_authority: snapshot_primary` is the thesis default:
   core selection is materialized directly from validated snapshot parameters.
2. `selection.selection_authority: materialized_csv_primary` is legacy
   compatibility mode; it may rely on `tuning_weights/selection_a...csv`.
3. `selection.objective_authority: unified_normalized` is the thesis default:
   exploration and autoscale use the same normalized objective.
4. `selection_source` and `selection_source_file` in `selection_contract.json`
   remain mandatory provenance fields for the frozen dataset.
5. In `snapshot_primary`, selection reconciliation is expected to be `aligned`.
   In `materialized_csv_primary`, documented differences are allowed if they are
   explicit in reports and method notes.

## Runtime/Invocation Policy

1. Canonical runtime invocation is `micromamba run -n dataselector <command>`.
2. Canonical scientific trigger-all orchestration is
   `python -m dataselector thesis-orchestrate`.
3. `scripts/exec_in_env.sh` is a compatibility wrapper layer and must delegate to micromamba/conda where available.
4. Canonical run root is `outputs/runs/`.
5. Scripts may orchestrate, but scientific core logic belongs in `dataselector/*`.
6. Runtime readiness can be checked via `python -m dataselector check-runtime-readiness`.
7. Wrapper governance can be checked via `python -m dataselector check-script-wrappers`.
8. `thesis-orchestrate` requires a fresh run directory; non-empty `--output-dir` is fail-fast.
9. Scientific artifacts are resolved run-local only; legacy global `outputs/*` resolver fallbacks are not used in the thesis path.
10. Each orchestrated run writes `manifest/artifact_hashes.json` in its run directory.

## Warning Policy (Thesis Gates)

1. Repository-internal warnings in thesis-relevant paths must be fixed, not broadly silenced.
2. Known external dependency warnings may be filtered only with a narrow, documented rule.
3. Current narrow exception: pyproj/numpy deprecation path
   (`Conversion of an array with ndim > 0 to a scalar is deprecated...`)
   in `pytest.ini`.
4. Broad warning suppression (e.g. global `DeprecationWarning` ignore) is not allowed.

## Feature Cache Identity Policy

Scientific runs must never reuse feature caches across different model/preprocess
setups. Cache identity must include:

1. `model_name`, `model_variant`, `dinov2_repo`, `dinov2_ref`
2. `pooling`, `input_size`, `crop_size`
3. `preprocess_pipeline_id`
4. optional `config_sha256`

Cache metadata must persist `feature_identity` and `model_provenance`.
Cache hits are valid only for full identity matches.

## Cache Mode Matrix (Scientific Contract)

1. `off`: no cache read, no cache write.
2. `read_only`: cache read only; miss/identity mismatch is fail-fast.
3. `write_only`: always recompute, then write cache.
4. `read_write`: read on match, otherwise recompute and write.

Default for thesis scientific orchestration is `read_write`.

## Evidence Scope Policy

`parameter_resolution_contract.yaml` evidence resolution rules:

1. default: run-relative evidence (`outputs/runs/<run_id>/...`)
2. explicit repo-relative evidence via `repo:<path>`
3. missing evidence: fail-fast (unless `--force` + reason)

Force override usage must be captured in `run_metadata.json`:
`force_override_used`, `force_override_reason`.

## CRS / Distance Policy (Thesis-Repro)

1. Thesis production path requires metric distance context (`EPSG:25832` target).
2. `thesis_repro` requires explicit source CRS provenance from sidecar/raster
   evidence; heuristic CRS inference is a fallback only outside strict thesis mode.
3. Unknown or non-explicit CRS in strict thesis profile is a hard error.
4. Runtime metadata must include:
   `source_crs`, `metric_crs`, `transform_applied`, `crs_provenance_status`,
   and `crs_provenance_audit_path`.
5. Thesis runs must emit `data_quality/crs_provenance_audit.csv`.
6. `compute_min_distance_km(...)` must derive nearest-neighbor distances from
   projected metric coordinates (`_proj_x`, `_proj_y`) only.

## Autoscale Production vs Diagnostic Policy

1. Diagnostic full-coverage stages are marked `diagnostic_only=true`.
2. Production parameter selection must come from non-diagnostic stages.
3. Real-data enforcement (`strict_real_data=true`) is required for thesis runs.

## Objective Comparability Policy

Autoscale/Optuna scoring uses normalized components:

1. `diversity_norm = diversity / baseline_diversity`
2. `spread_norm = spread / baseline_spread`
3. weighted objective from normalized components
4. infeasible selections (`n_selected < target`) are explicitly tagged and penalized

## Global n_samples Corridor Policy

`n_samples` autoscale defaults are resolved in core workflow logic, not in wrapper scripts.

1. Policy keys live in `selection.autoscale_n_samples_*` in
   `config/pipeline_config.yaml`.
2. `mode=corridor` derives stage values from effective candidates (`N_eff`) after
   tile exclusions.
3. Corridor staging uses integer range exploration with
   `selection.autoscale_n_samples_step` (default: `1`).
4. `mode=fixed` optimizes exactly one fixed `n_samples` value.
5. Final `n_samples` is selected via minimal-feasible plateau rule:
   `score(n) >= best_score * (1 - plateau_delta)`.
6. Policy and resolved stages/trials are written to:
   `parameter_resolution/optuna_autoscale_stage_policy.json`.

### Scientific Method Rationale (n_samples)

This policy is intentionally aligned with a core-set thesis workflow where
annotation effort is the main constraint.

1. Goal is not maximal sample count, but the smallest scientifically defensible
   annotation set for stable downstream model comparison.
2. The corridor is centered around `5%` of effective candidates (`N_eff`) with
   controlled local exploration (`4-8%`) to avoid overfitting to one arbitrary
   fixed sample size.
3. With `step=1`, the core evaluates every integer `n` inside the bounded
   corridor, not only a few anchor points.
4. The final choice uses a minimal-feasible plateau rule:
   among feasible candidates, select the smallest `n` that is within
   `plateau_delta` of the best score. This formalizes
   "as few annotations as possible, without meaningful quality loss."
5. `N_eff` (after tile exclusions) is used instead of raw dataset size so that
   policy remains stable when exclusion rules change.
6. This policy governs selection only. Train/val/test split leakage control is a
   separate downstream evaluation contract.
7. Architecture-specific evidence for MapSAM / SegFormer / UNet++ is
   supplementary only: it supports plausibility and conservatism of the
   corridor for downstream training, but does not replace the Dataselector
   selection-policy rationale.

Supplementary evidence note:

- `docs/N_SAMPLES_EVIDENCE_ADDENDUM.md`

## Feature Cache Scope Policy

1. Feature caching supports two scopes:
   `feature_cache.scope: global_shared|run_local`.
2. Thesis default is `global_shared` with root `outputs/cache/features`.
3. Cache objects are immutable per `cache_key` (`<root>/<cache_key>/features.npy`,
   `<root>/<cache_key>/meta.json`).
4. Cache key derives only from scientific inputs (metadata hash + feature identity),
   not from unrelated repo file changes.

## Leakage-Safe Split Policy (Phase 4H)

1. Tile exclusions are policy-driven via `config/tile_exclusion_policy.yaml`.
2. In thesis-repro orchestration, exclusions are applied before candidate pooling.
3. Leakage buffer policy is defined in `config/spatial_split_policy.yaml`.
4. `d_leak` is calibrated from feature-similarity decay unless explicitly overridden.
5. Split construction uses connected components under:
   `edge_distance_km < d_leak`.
6. Inter-split leakage is audited and stored in:
   `splits/leakage_audit.csv`.
7. In `thesis_repro`, leakage violations are fail-fast.
8. `repo:` evidence paths are opt-in; default evidence scope is run-local.

## Integrated Phase 5 Policy

Optional post-freeze packaging may be integrated into the canonical thesis run path.

1. `--build-handoffs` is the gate flag and defaults to `false`.
2. Integrated Phase 5 is operational packaging only:
   tile handoff `prepare` + `verify-local`, annotation plan build, patch handoff
   `prepare-patches` + `verify-patches`.
3. Once annotation uses the integrated Phase 5 patches, the annotated Phase 5
   patch dataset is frozen.
4. The frozen patch contract includes `selected_patches.csv`, `patch_id`,
   patch bounds / quicklook extents, patch-mask assignment, and the upstream
   `patch_split_manifest.json`.
5. `split_authority = masterarbeit_strassenerkennung_cv` means the downstream
   training repo may build and manage the materialized training contract it
   actually runs, but it must not redefine the annotated Phase 5 patch
   dataset.
6. For primary thesis results, the frozen upstream patch split regime remains
   the methodological reference. Alternative downstream-local split generation
   is fallback or sensitivity analysis only.
7. Default integrated patch scope is `core-only`
   (`--patch-include-case false`).
8. Integrated patch density defaults to `--patches-per-tile 2`.
9. Integrated handoff artifacts may write into `annotation_plan/` inside the run
   directory and into `handoff/` outside the run directory.
10. Integrated Phase 5 must not mutate snapshot files, `selection_*`,
   `selection_contract.json`, or parameter-resolution artifacts.
11. If Integrated Phase 5 fails, the overall run fails operationally, but the
   scientific freeze remains the authoritative boundary.

## Handoff Schema Policy (v1)

Externer Übergabevertrag für das Trainings-Repo:

1. `selected_maps.csv`
   - Pflichtspalten: `shortName,image_path,image_filename,selection_rank`
2. `handoff_manifest.json`
   - Pflichtfelder inkl. Provenance, Snapshot-Hash und Exclusion-Policy
3. `mask_requirements.csv`
   - Spalten: `shortName,required_mask_filename`

Split-Verantwortung:

1. `split_authority` ist fest auf `masterarbeit_strassenerkennung_cv` gesetzt.
2. Dataselector erzeugt in diesem Vertrag keine autoritativen CV-Splits.
3. Das Trainings-Repo validiert Handoff + Masken und übernimmt Split-Erzeugung.
4. Für den patch-basierten Phase-5-Handoff bleibt der annotated Phase 5 patch
   dataset dennoch eingefroren: `selected_patches.csv`, `patch_id`,
   Patch-Geometrie / Quicklook-Ausschnitte, Patch-Mask-Zuordnung und das
   Upstream-`patch_split_manifest.json` bleiben unverändert.
5. Downstream-Split-Materialisierung ist ein technischer Schritt zur Erzeugung
   des materialized training contract; sie ist kein stillschweigendes Redesign
   des annotierten Datensatzes oder des Hauptevaluationsprotokolls.

## Phase-5 Width Calibration Policy

1. Canonical runtime path for width calibration is:
   `python -m dataselector orchestrate-width-calibration`.
2. Canonical scientific default is `quota_mode=proportional`.
3. Legacy `quota_mode=fixed` remains allowed for compatibility and controlled
   baseline comparison runs.
4. The deterministic queue contract includes at least:
   `handoff_dir`, roads source provenance, `seed`, `crop_size_px`, and the
   resolved sampling parameters.
5. Seed changes are scientific policy changes for queue generation and must be
   documented when deviating from the default comparison baseline (`seed=42`).
6. Width calibration must run against a frozen handoff patch set for the active
   run; no silent upstream patch re-selection is allowed during measurement.
7. Summary outputs must include both pixel and meter statistics where metric
   conversion is valid. Pixel widths remain the operational calibration target.

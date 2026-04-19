# Parameter Policy Ledger (Authoritative)

## Purpose

This ledger defines how critical thesis parameters are sourced and validated.
Every critical value must be either:

1. `computed` via a documented method, or
2. `policy` with explicit rationale and evidence artifacts.

This document is authoritative for `dataselector thesis-pipeline`.

## Canonical Scope

1. Active config: `config/pipeline_config.yaml`
2. Canonical execution path: `python -m dataselector thesis-pipeline`
3. Canonical runtime invocation: `micromamba run -n dataselector <command>`
4. Canonical run root: `outputs/runs/`

## Critical Parameters

| Parameter | Class | Method / Rationale | Artifact / Evidence | Data Scope | Decision Date |
| --- | --- | --- | --- | --- |
| `selection.n_samples` | `computed_required` (default) or `policy` (`fixed`) | Global autoscale corridor policy from `N_eff` + minimal-feasible plateau; primary basis: annotation-budgeted core-set policy; supplementary architecture-specific evidence indicates that `~5%` is reasonable for foundation-model-based downstream training while the upper side remains conservative for UNet++ | `parameter_resolution/optuna_autoscale_stage_policy.json`, `parameter_resolution/optuna_autoscale_selected_n_samples.txt`, `docs/N_SAMPLES_EVIDENCE_ADDENDUM.md`, run metadata | Canonical metadata | 2026-02-12 |
| `selection.min_distance_km` | `policy` or `computed` | Operational policy floor informed by the pre-registered `28.5/40.0/45.0` comparison; computed references use strict metric NN distances in EPSG:25832; supplementary historical runs show low-distance objective flatness / stochastic drift toward `5-8 km` under otherwise fixed profile | `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_DECISION_2026-02-09.md`, `docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md` | Canonical metadata | 2026-02-09 |
| `selection.alpha_visual` | `computed` or `policy` | Optuna/LHS optimization or policy-tagged fixed value | snapshot provenance (`selection._provenance.alpha_visual`) | Current run data hash | 2026-02-11 |
| `selection.beta_spatial` | `computed` or `policy` | Optuna/LHS optimization or policy-tagged fixed value | snapshot provenance (`selection._provenance.beta_spatial`) | Current run data hash | 2026-02-11 |
| `selection.gamma_temporal` | `computed` or `policy` | Optuna/LHS optimization or policy-tagged fixed value | snapshot provenance (`selection._provenance.gamma_temporal`) | Current run data hash | 2026-02-11 |
| `selection.optuna_sampler` | `computed` or `policy` | Resolution order: config policy > artifact > controlled determination run | run-local sampler evidence: `outputs/runs/<run_id>/parameter_resolution/sampler_resolution/selected_sampler.json` + run metadata (`resolved_sampler_source`) | Current run data hash | 2026-02-11 |
| `selection.anchor_aliases` | `policy` | Configurable anchor alias mapping (default keeps Hamburg -> KDR_146 behavior) | config snapshot + selection provenance | Current run policy | 2026-02-11 |
| `clustering.n_clusters` | `computed` or `policy` | Elbow/Silhouette evidence or explicit policy | snapshot provenance (`clustering._provenance.n_clusters`) | Current run data hash | 2026-02-11 |
| `clustering.umap_n_neighbors` | `computed` or `policy` | sensitivity/grid or explicit policy | snapshot provenance (`clustering._provenance.umap_n_neighbors`) | Current run data hash | 2026-02-11 |
| `clustering.umap_min_dist` | `computed` or `policy` | sensitivity/grid or explicit policy | snapshot provenance (`clustering._provenance.umap_min_dist`) | Current run data hash | 2026-02-11 |
| `feature_extraction.batch_size` | `computed` or `technical` | autoscale/system constrained | snapshot provenance (`feature_extraction._provenance.batch_size`) | Runtime environment hash | 2026-02-11 |
| `feature_extraction.pooling` | `technical` | DINOv2 embedding policy (`cls` or `mean`), fixed per run | snapshot provenance (`feature_extraction._provenance.pooling`) | Current run data hash | 2026-02-11 |
| `feature_extraction.model_variant` | `technical` | Pinned DINOv2 model variant (e.g. `dinov2_vits14`) | snapshot provenance (`feature_extraction._provenance.model_variant`) | Current run data hash | 2026-02-11 |
| `feature_extraction.dinov2_repo` | `technical` | Explicit upstream model repository | snapshot provenance (`feature_extraction._provenance.dinov2_repo`) | Current run data hash | 2026-02-11 |
| `feature_extraction.dinov2_ref` | `technical` | Explicit repo ref (branch/tag/commit) for reproducibility | snapshot provenance (`feature_extraction._provenance.dinov2_ref`) | Current run data hash | 2026-02-11 |
| `feature_extraction.preprocess_pipeline_id` | `technical` | Fixed preprocessing chain identifier used in cache identity and provenance | feature cache meta + snapshot provenance | Current run data hash | 2026-02-11 |
| `thesis_runtime.n_trials` | `policy` | explicit optimization budget (default: 370) — derived from Phase‑0 convergence baseline; evidence: `outputs/runs/recompute_tpe_370/` | run metadata + snapshot provenance | Current run data hash | 2026-02-13 |
| `thesis_runtime.cache_mode` | `policy` | explicit cache semantics (`off|read_only|write_only|read_write`) | run metadata + CLI args snapshot | Current run execution policy | 2026-02-11 |
| `selection.validation_seeds` | `policy` | deterministic reproducibility panel | run metadata + snapshot provenance | Current run data hash | 2026-02-11 |
| `validation.replicate_mode` | `policy` | inferenzielle UQ über `bootstrap_candidates`; `seed_replay` nur für Replay/Determinismus | `validation/validation_method_contract.md`, run metadata | Current run validation scope | 2026-02-23 |
| `validation.n_bootstrap` | `policy` | Anzahl Bootstrap-Replikate für UQ (Default 200) | run metadata + validation summary | Current run validation scope | 2026-02-23 |
| `validation.bootstrap_sample_frac` | `policy` | Anteil resampelter Kandidaten pro Bootstrap-Replikat | run metadata + validation summary | Current run validation scope | 2026-02-23 |
| `runtime.metadata_crs` | `computed` | CRS provenance from metadata load; `thesis_repro` requires explicit source CRS from sidecar/raster evidence and records status/counts plus heuristic fallback diagnostics | run metadata (`metadata_crs`), `data_quality/crs_provenance_audit.csv` | Current run metadata | 2026-03-13 |
| `selection.tile_exclusions` | `policy` | Candidate pool exclusions loaded from tile exclusion policy; `KDR_155b` stays excluded as same-site duplicate representation, while all tiles outside the named policy constant `kdr_core_publication_frame` (`1878-1945`) remain in-pool but must be reported as retained temporal outliers | `config/tile_exclusion_policy.yaml`, run metadata fields (`tile_exclusions_*`, `tile_flagged_*`) | Current run metadata | 2026-03-13 |
| `selection.case_tile_names` | `policy` | Liste zusätzlicher Case-Tiles getrennt vom Core-Sampling (Default: `["Hamburg"]`) | `selection_case.csv`, `selection_contract.json` | Current run selection scope | 2026-02-23 |
| `selection.case_exclude_from_core` | `policy` | Steuert, ob Case-Tiles aus der Core-Selektion ausgeschlossen sind | `selection_contract.json` | Current run selection scope | 2026-02-23 |
| `selection.case_attach_mode` | `policy` | Anhängemodus für Case-Tiles (`append_unique|append_all`) | `selection_contract.json` | Current run selection scope | 2026-02-23 |
| `selection.leakage_buffer_km` | `computed` or `policy` | Data-driven calibration (`auto`) or explicit override | `policy/distance_policy.json`, `policy/leakage_calibration.csv` | Current run metadata + feature space | 2026-02-11 |
| `evaluation.split_manifest` | `computed` | Deterministic connected-component split under edge-distance leakage rule | `splits/split_manifest.json`, `splits/leakage_audit.csv` | Current run metadata | 2026-02-11 |
| `evaluation.split_manifest_sha256` | `computed` | Fair model comparison requires identical split manifest hash across all model runs | `split_manifest_sha256` in run metadata | Cross-model comparison scope | 2026-02-11 |

## Selection Authority (Thesis v2 Contract)

Dataset authority: `selection_core.csv`, `selection_final_with_cases.csv`, and
`selection_contract.json` define the frozen thesis dataset.

Parameter authority: validated snapshot and parameter-resolution artifacts
define the resolved parameter context.

1. Thesis default is `selection.selection_authority: snapshot_primary`.
   Core selection is materialized directly from validated snapshot parameters.
2. Legacy compatibility mode is
   `selection.selection_authority: materialized_csv_primary`.
   It remains valid for historical runs, but is not the thesis default.
3. Thesis default is `selection.objective_authority: unified_normalized` so
   exploration and autoscale stay on one shared objective.
4. `selection_source` and `selection_source_file` in `selection_contract.json`
   remain mandatory provenance fields for the frozen dataset.
5. For canonical v2 runs, reconciliation status is expected to be `aligned`.
   `documented_difference` is acceptable only for explicit legacy runs.

## Snapshot Contract

Snapshots must include:

1. `hashes.parameters_hash`
2. `hashes.snapshot_content_sha256`
3. additive provenance blocks:
   - `selection._provenance`
   - `clustering._provenance`
   - `feature_extraction._provenance`
4. source hashes where applicable (`source_hash`)

Validation is mandatory for `--use-params`; mismatches fail unless `--force` is
explicitly set. Forced runs must be flagged in `run_metadata.json`.

## Scientific Helper Command Contract

Scientific diagnostics are centralized behind CLI commands in
`dataselector/workflows/scientific_tools.py`:

1. `sensitivity-sweep`
2. `ablation-study`
3. `compare-backbones`
4. `validate-kmeans`
5. `validate-umap`
6. `snapshot-config`

Top-level scripts with matching names are compatibility wrappers only and must
delegate to these CLI commands.

## Decision Discipline

No silent fallback for critical values in thesis production path.

1. Unresolved critical parameter + no compute policy => fail-fast.
2. Any policy override must be documented in status/report artifacts.
3. Historical configs may be referenced, but never promoted as active defaults.

## Additional Scientific Rules (Hardening v5)

1. Autoscale diagnostic stages (`diagnostic_only=true`) must not determine final production parameters.
2. Objective comparisons must use normalized components; infeasible trials are explicitly tagged and penalized.
3. Evidence resolution is run-relative by default; `repo:` paths are opt-in and explicit.

## Handoff Artifacts (v1)

| Parameter | Class | Method / Rationale | Artifact / Evidence | Data Scope | Decision Date |
| --- | --- | --- | --- | --- | --- |
| `handoff_manifest` | `computed` | Reproducible transfer contract from resolved run selection to training repo; includes snapshot/source hashes and exclusion policy provenance | `handoff/handoff_manifest.json`, `handoff/selected_maps.csv`, `handoff/mask_requirements.csv` | Selected map subset of current run | 2026-02-12 |

Integrated Phase 5 policy:

1. `build_handoffs` is optional and defaults to `false`.
2. Integrated Phase 5 is post-freeze packaging only; it must not mutate snapshot or `selection_*` artifacts.
3. Once annotation uses the Phase 5 patches, the annotated Phase 5 patch dataset is frozen: `selected_patches.csv`, `patch_id`, patch bounds / quicklook extents, patch-mask assignment, and the upstream `patch_split_manifest.json`.
4. `split_authority = masterarbeit_strassenerkennung_cv` allows the downstream repo to manage the materialized training contract it actually runs, but not to redefine the annotated Phase 5 patch dataset.
5. For primary thesis results, the frozen upstream patch split regime remains the methodological reference; alternative downstream-local split generation is fallback or sensitivity analysis only.
6. Default integrated patch scope is `core-only` (`patch_include_case=false`).

## Width Calibration Parameters (v3 policy)

| Parameter | Class | Method / Rationale | Artifact / Evidence | Data Scope | Decision Date |
| --- | --- | --- | --- | --- | --- |
| `width_calibration.quota_mode` | `policy` | Standardisiert auf `proportional` zur kontrollierten, verteilungsnahen und budgetierten Messstichprobe; `fixed` bleibt nur Legacy/Baseline | `width_calibration_manifest.json` (`quota_mode`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.sampling_rate` | `policy` | Proportionaler Anteil pro Klasse für Primärtasks | `width_calibration_manifest.json` (`quota_mode_parameters.sampling_rate`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.min_per_class` | `policy` | Mindestabdeckung seltener Klassen trotz proportionaler Auswahl | `width_calibration_manifest.json` (`quota_mode_parameters.min_per_class`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.max_per_class` | `policy` | Optionale Kappung dominanter Klassen, 0 bedeutet keine Kappung | `width_calibration_manifest.json` (`quota_mode_parameters.max_per_class`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.repeat_sampling_rate` | `policy` | Steuerung der Repeat-Stichprobe zur Reliabilitätsabschätzung | `width_calibration_manifest.json` (`quota_mode_parameters.repeat_sampling_rate`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.repeat_min_per_class` | `policy` | Mindestabdeckung je Klasse für Repeat-Messungen | `width_calibration_manifest.json` (`quota_mode_parameters.repeat_min_per_class`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.seed` | `policy` | Deterministische Queue-Reproduzierbarkeit; Seed-Wechsel ist methodische Änderung | `width_calibration_manifest.json` (`seed`) | Active width-calibration run | 2026-04-18 |
| `width_calibration.summary_units` | `policy` | Summary berichtet px+m; px bleibt operative Kalibrierbasis, m wissenschaftliche Zusatzsicht bei gültiger Metrik | `width_calibration_summary.csv`, `width_calibration_summary.json` | Active width-calibration run | 2026-04-18 |

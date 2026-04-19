# Min-Distance Evidence Addendum

This note consolidates the evidence chain behind the active thesis production
policy `selection.min_distance_km = 28.5`.

## Purpose

1. Keep the normative decision evidence separate from supplementary historical
   run evidence.
2. Make explicit why `min_distance_km` remains a hard production floor even
   though the selection objective already includes a spatial component.
3. Document `45.0 km`, `40.0 km`, `28.5 km`, and the historical `5 km` / `8 km`
   results in one place.

## Primary Decision Evidence

The normative thesis decision comes from the pre-registered comparison panel on
`2026-02-09`:

1. `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_PRE_REGISTRATION_2026-02-09.md`
2. `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_DECISION_2026-02-09.md`
3. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_20260209T233849Z.csv`
4. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_20260209T233849Z.md`

The panel fixed the selection path and compared exactly three candidate
distances:

1. `28.5 km` as the current operational policy candidate
2. `40.0 km` as an intermediate comparison candidate
3. `45.0 km` as the geometric reference from canonical metadata

Recorded result:

1. `28.5 km` and `40.0 km` were tied on the decision summary for feasibility,
   stability, cluster coverage, temporal spread, spatial mean distance, and
   stability Jaccard.
2. `45.0 km` was slightly weaker on coverage in the same panel.
3. The pre-registered tie-breaker selected the smaller distance.

This makes `28.5 km` the active operational thesis policy. `40.0 km` remains a
validated comparison candidate, not a failed value. `45.0 km` remains a
geometric descriptor, not the production default.

## Supplementary Historical Evidence

The following historical full runs are supplementary evidence only. They are
not replacement decision runs and do not override the `2026-02-09`
pre-registration.

1. `docs/status/thesis_pipeline_double_run_analysis_2026-02-11.md`
2. `outputs/runs/thesis_orchestrate_full_20260211T172730Z_A/parameter_resolution/optuna_autoscale_best_latest.json`
3. `outputs/runs/thesis_orchestrate_full_20260211T172730Z_B/parameter_resolution/optuna_autoscale_best_latest.json`

Key observations from that evidence:

1. Under otherwise fixed thesis-repro profile, run A resolved to
   `min_distance_km = 5`.
2. Under the same profile class, run B resolved to `min_distance_km = 8`.
3. Both runs remained fully feasible (`n_selected = 100`,
   `feasibility_ratio = 1.0`, `best_from_production_stage = true`).

Interpretation:

1. These historical `5 km` / `8 km` results do not prove that low distances are
   scientifically wrong.
2. They do show that, under earlier unrestricted or weaker bounded autoscale
   settings, the objective could drift into a very small-distance region without
   an immediate feasibility penalty.
3. This is supplementary evidence for a comparatively flat low-distance region
   in which stochastic optimization can settle on methodologically permissive
   values.

## Why A Hard Floor Still Exists

The selection objective already includes a spatial component. The hard
`min_distance_km` floor therefore serves a different role:

1. the objective scores spatial quality softly,
2. the floor defines which solutions remain admissible as thesis production
   selections,
3. the floor prevents the production path from drifting into a low-distance
   region that is only weakly worse in the objective but methodologically more
   permissive.

This means:

1. `min_distance_km` is not a claim that the spatial objective is useless,
2. it is not a claim that `40.0 km` is wrong,
3. it is not a claim that `5 km` or `8 km` automatically produce bad
   downstream models,
4. it is a conservative production guardrail justified by primary pre-registered
   evidence plus supplementary historical drift evidence.

## Current Scientific Reading

1. `45.0 km` = geometric reference from canonical tile geometry
2. `40.0 km` = tested comparison candidate in the decision panel
3. `28.5 km` = active production policy selected by pre-registered rule
4. `5 km` / `8 km` = historical supplementary evidence that fully free or weakly
   constrained optimization can drift into a permissive low-distance zone

## Use In Active Docs

Active thesis documentation should cite this addendum when explaining why
`selection.min_distance_km` is a policy floor instead of a fully unconstrained
optimizer output.

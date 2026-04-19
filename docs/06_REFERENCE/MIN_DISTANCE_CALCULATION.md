# Min-Distance Calculation and Policy

## Scope
This document defines the scientific split between:

1. **Computed geometric reference** from canonical metadata.
2. **Operational thesis policy** used in production selection runs.

Canonical metadata source is `data/new_all_tiles.csv`.

## Current Values (2026-02-09)

1. `min_distance_geom_ref = 45.0 km`
2. `min_distance_operational = 28.5 km`

These values are intentionally different and serve different purposes.

## Why Two Values Exist

1. `45.0 km` is the median nearest-neighbor distance computed from tile geometry.
2. `28.5 km` is the operational policy selected via a pre-registered decision gate to preserve feasible combination space while keeping hardcut feasibility and stability.
3. `40.0 km` was the intermediate comparison candidate in the pre-registered
   decision panel, not a discarded error value.

Decision artifacts:

1. `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_PRE_REGISTRATION_2026-02-09.md`
2. `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_DECISION_2026-02-09.md`
3. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_20260209T233849Z.csv`
4. `docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md`

## Reproducible Computation of Geometric Reference

Function:

- `dataselector.pipeline.pipeline_utils.compute_min_distance_km`

Manual command:

```bash
micromamba run -n dataselector -- python - <<'PY'
from dataselector.pipeline.pipeline_utils import compute_min_distance_km
print(compute_min_distance_km('data/new_all_tiles.csv'))
PY
```

Expected current output on canonical data: `45.0`.

## Reproducible Policy Comparison

Comparison command:

- `dataselector compare-min-distance-policies`

Command used for thesis decision panel:

```bash
micromamba run -n dataselector -- python -m dataselector compare-min-distance-policies \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 40 45 \
  --seeds 42 43 44 45 46 \
  --output-dir docs/06_REFERENCE/thesis_decision_evidence
```

Decision rule (pre-registered, applied in order):

1. mean shortfall-rate `< 0.10`
2. `std(n_selected) < 3`
3. maximize quality/diversity proxy
4. tie-break: smaller distance

Selected operational policy: `28.5 km`.

Why the policy is not fully delegated to the optimizer:

1. the spatial component of the objective remains active and meaningful,
2. the hard floor serves as a production admissibility guard, not as a
   replacement for the objective,
3. supplementary historical full runs show that earlier weakly constrained
   autoscale could drift to `5 km` and `8 km` under otherwise fixed profile,
   even though feasibility remained intact,
4. these `5 km` / `8 km` results are not normative decision evidence, but they
   support keeping a conservative floor outside the pre-registered panel.

## Pipeline Contract Integration

### Thesis production path

1. Uses operational policy (`selection.min_distance_km`) from `config/pipeline_config.yaml` unless explicitly overridden via CLI.
2. Hardcut is enforced during selection (no silent constraint breaking).

### Adaptive evidence path

1. Computes geometric reference (`compute_min_distance_km`) as data-driven reference.
2. Uses comparison scripts/reports for policy evidence.
3. Does not silently mutate thesis policy after decision freeze.

## Scientific Classification

1. `45.0 km` = computed descriptor of dataset geometry.
2. `40.0 km` = tested comparison candidate in the decision panel.
3. `28.5 km` = policy decision with documented evidence.
4. `5 km` / `8 km` = supplementary historical evidence for low-distance drift,
   not an operational recommendation.

This separation avoids post-hoc confusion between “what data geometry suggests” and “what production policy uses”.

## Notes

1. If canonical metadata changes materially, rerun both reference computation and policy comparison.
2. Policy must only be changed through a new pre-registration + decision artifact pair.

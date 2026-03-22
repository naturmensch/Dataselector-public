# Min Distance Decision (2026-02-09)

## Context

1. Pre-registration: `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_PRE_REGISTRATION_2026-02-09.md`
2. Dataset: `data/new_all_tiles.csv` (`676` rows)
3. Fixed `n_samples`: `34` (policy baseline for this decision run)
4. Candidate distances: `28.5`, `40.0`, `45.0`
5. Seed panel: `42, 43, 44, 45, 46`

## Executed Command

```bash
/opt/miniconda3/envs/dataselector/bin/python scripts/compare_min_distance_policies.py \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 40 45 \
  --seeds 42 43 44 45 46 \
  --output-dir docs/06_REFERENCE/thesis_decision_evidence
```

## Artifacts

1. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_runs_20260209T233849Z.csv`
2. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_20260209T233849Z.csv`
3. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_20260209T233849Z.md`
4. Runner log: `outputs/phase4h/logs/20260209T232349Z_wave2_distance.log`

## Rule Evaluation (pre-registered order)

1. Feasibility gate (`mean shortfall-rate < 0.10`): passed by all candidates.
2. Stability gate (`std(n_selected) < 3`): passed by all candidates.
3. Quality criterion: `28.5` and `40.0` tie on key quality proxies; `45.0` slightly lower cluster coverage.
4. Tie-breaker: choose smaller distance.

## Decision

1. Geometric reference value: `45.0` (`compute_min_distance_km` output).
2. Operational policy value: **`28.5`**.

## Rationale

1. All candidates pass hard gates.
2. `28.5` is non-inferior on feasibility/stability and wins the pre-registered tie-breaker.
3. Smaller distance preserves downstream combination space without reducing hardcut success in this panel.

## Threshold Interpretation (scientific framing)

1. The gates (`mean shortfall-rate < 0.10`, `std(n_selected) < 3`) are
   **pre-registered operational thresholds** for this project context.
2. They are not claimed as universal geospatial standards; instead they define
   a transparent tolerance band and stability requirement for this dataset and workflow.
3. Scientific grounding:
   - replicate-seed stability follows the logic of stability-selection style evaluation,
   - shortfall tolerance follows minimum-sufficient / plateau reasoning from
     sample-size and learning-curve practice.
4. Because all candidates passed hard gates, the decision was resolved by the
   pre-registered tie-breaker (smaller distance), not by post-hoc narrative selection.

## Literature anchors (concept-level)

1. Meinshausen, N., and Buhlmann, P. (2010). Stability Selection.
2. Shah, R. D., and Samworth, R. J. (2013). Complementary Pairs Stability Selection.
3. Geostatistical and learning-curve literature that uses practical tolerance bands
   and diminishing-return logic for design decisions.

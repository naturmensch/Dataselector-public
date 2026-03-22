# Min Distance Pre-Registration (2026-02-09)

## Metadata

1. Date (UTC): 2026-02-09
2. Commit baseline: `aa522c3`
3. Dataset: `data/new_all_tiles.csv` (`676` rows)
4. Fixed thesis target: `n_samples=34` (current policy baseline for this decision run)

## Research Question

Which operational `min_distance_km` policy best balances hardcut feasibility, stability, and selection quality for the thesis production path?

## Candidates

1. `28.5` (current operational policy)
2. `40.0` (intermediate reference candidate)
3. `45.0` (geometric reference from `compute_min_distance_km`)

## Fixed Execution Path

1. Same selector code path for all candidates.
2. Same metadata/features/cache source.
3. Same seeds panel: `42, 43, 44, 45, 46`.
4. Same weights/cluster defaults from `config/pipeline_config.yaml`.

## Decision Rule (apply in order)

1. Feasibility gate: mean shortfall-rate `< 0.10`.
2. Stability gate: `std(n_selected) < 3`.
3. Quality criterion: maximize objective/diversity score among feasible candidates.
4. Tie-breaker: choose smaller distance (preserve combination space).

## Metrics To Record

1. `hardcut_target_met` rate
2. shortfall rate
3. `mean_n_selected`, `std_n_selected`
4. objective/diversity proxies from comparison summary
5. seed stability (pairwise Jaccard mean/min)

## Command

```bash
/opt/miniconda3/envs/dataselector/bin/python scripts/compare_min_distance_policies.py \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 40 45 \
  --seeds 42 43 44 45 46 \
  --output-dir docs/06_REFERENCE/thesis_decision_evidence
```

## Output Artifacts (expected)

1. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_runs_<timestamp>.csv`
2. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_<timestamp>.csv`
3. `docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_<timestamp>.md`
4. Consolidated decision: `docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_DECISION_2026-02-09.md`

## Non-Goals

1. No algorithm redesign.
2. No post-hoc threshold changes after results.


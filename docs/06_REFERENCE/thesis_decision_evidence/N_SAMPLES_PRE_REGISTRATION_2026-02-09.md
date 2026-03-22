# N-Samples Pre-Registration (2026-02-09)

## Metadata

1. Date (UTC): 2026-02-09
2. Commit baseline: `aa522c3`
3. Dataset: `data/new_all_tiles.csv` (`676` rows)
4. Fixed distance policy for this panel: `min_distance_km=28.5` (current operational policy baseline)

## Research Question

What is the minimum sufficient thesis sample size `n_samples` that satisfies feasibility, stability, and quality constraints under the current production contract?

## Candidate Panel

1. Primary candidates: `24, 28, 32, 34`
2. Reference-only candidate: `40`

## Fixed Execution Path

1. Same selector code path for all candidates.
2. Same metadata/features/cache source.
3. Same seeds panel: `42, 43, 44, 45, 46`.
4. Same distance (`28.5`) and same weight defaults.

## Decision Rule (mechanical)

1. Reject candidates with mean shortfall-rate `>= 0.10`.
2. Reject candidates with `std(n_selected) >= 3`.
3. Reject candidates with materially degraded quality/diversity proxies vs panel median.
4. From remaining candidates, choose the **smallest** `n_samples`.
5. If needed, sampler elegance (`2^k`, multiple of 8) is tie-breaker only, not primary criterion.

## Metrics To Record

1. `hardcut_target_met` rate
2. shortfall rate
3. `mean_n_selected`, `std_n_selected`
4. diversity/spread/objective proxies
5. seed stability (pairwise Jaccard mean/min)

## Planned Commands

Run per candidate `n`:

```bash
/opt/miniconda3/envs/dataselector/bin/python scripts/compare_min_distance_policies.py \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 \
  --seeds 42 43 44 45 46 \
  --n-samples <n> \
  --output-dir docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_<n>
```

## Output Artifacts (expected)

1. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_<n>/min_distance_policy_runs_<timestamp>.csv`
2. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_<n>/min_distance_policy_summary_<timestamp>.csv`
3. Consolidated decision: `docs/06_REFERENCE/thesis_decision_evidence/N_SAMPLES_DECISION_2026-02-09.md`

## Scientific Classification Constraint

1. Final chosen `n_samples` must be marked as:
   - policy value with explicit external constraint evidence, or
   - derived value with reproducible adequacy evidence.


# N-Samples Decision (2026-02-09)

## Context

1. Pre-registration: `docs/06_REFERENCE/thesis_decision_evidence/N_SAMPLES_PRE_REGISTRATION_2026-02-09.md`
2. Dataset: `data/new_all_tiles.csv` (`676` rows)
3. Fixed distance policy for this panel: `min_distance_km=28.5`
4. Candidate panel: `24, 28, 32, 34` (+ reference `40`)
5. Seed panel: `42, 43, 44, 45, 46`

## Executed Commands

Per candidate `n`:

```bash
/opt/miniconda3/envs/dataselector/bin/python scripts/compare_min_distance_policies.py \
  --metadata-path data/new_all_tiles.csv \
  --distances 28.5 \
  --seeds 42 43 44 45 46 \
  --n-samples <n> \
  --output-dir docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_<n>
```

## Artifacts

1. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_24/min_distance_policy_summary_20260209T233908Z.csv`
2. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_28/min_distance_policy_summary_20260209T233912Z.csv`
3. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_32/min_distance_policy_summary_20260209T233916Z.csv`
4. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_34/min_distance_policy_summary_20260209T233919Z.csv`
5. `docs/06_REFERENCE/thesis_decision_evidence/n_samples/n_40/min_distance_policy_summary_20260209T233923Z.csv`
6. Runner log: `outputs/phase4h/logs/20260209T233904Z_wave3_nsamples.log`

## Rule Evaluation (mechanical)

Hard gates:

1. Mean shortfall-rate `< 0.10`
2. `std(n_selected) < 3`

Observed:

1. All candidates pass hard gates.
2. Stability is identical across candidates in this panel (`std(n_selected)=0`, Jaccard mean `1.0`).
3. Quality proxies are close; no hard-gate quality failure observed.

Given pre-registered minimum-sufficient rule ("choose smallest `n` among passing candidates"), the selected value is:

## Decision

**`n_samples = 24`** (minimum sufficient in this panel).

## Scientific Classification

1. This is a policy decision supported by reproducibility evidence.
2. Sampler elegance (`2^k`, multiple of 8) was not needed as primary criterion because the minimum-sufficient rule already resolved the choice.

## Threshold Interpretation (scientific framing)

1. The thresholds `mean_shortfall < 0.10` and `std(n_selected) < 3` are
   **pre-registered design thresholds**, not universal constants from a single standard.
2. Their role is to operationalize:
   - feasibility loss tolerance (shortfall),
   - stability across replicate seeds (selection variance).
3. This structure is aligned with established ideas from:
   - stability selection / replicate stability assessment,
   - learning-curve and minimum-sufficient sample-size practice.
4. For this decision run, the thresholds were fixed *before* evaluation and then
   applied mechanically to avoid post-hoc tuning.
5. Interpretation for thesis text:
   - literature supports the criterion structure (stability + tolerable loss),
   - exact numeric cutoffs remain context-bound and are documented as conservative policy choices.

## Literature anchors (concept-level)

1. Meinshausen, N., and Buhlmann, P. (2010). Stability Selection.
2. Shah, R. D., and Samworth, R. J. (2013). Complementary Pairs Stability Selection.
3. Recent sample-size/learning-curve work (e.g., supervised ML sample-size optimization studies, 2024-2025) supporting minimum-sufficient/plateau logic.

## Follow-up

1. ✅ Config/docs/tests were aligned to the selected policy value (`24`) in the phase4h merge stack.
2. If thesis resource constraints require a higher target than minimum-sufficient, document that explicitly as policy override.

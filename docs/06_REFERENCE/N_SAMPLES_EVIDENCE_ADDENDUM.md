# n_samples Corridor Evidence Addendum

This note consolidates the evidence chain behind the active thesis production
policy for `selection.n_samples` autoscale staging.

## Purpose

1. Keep the primary Dataselector policy rationale separate from supplementary
   downstream model evidence.
2. Document why the active corridor is centered around `5%` of effective
   candidates (`N_eff`) and explored in the bounded range `4-8%`.
3. Make explicit that architecture-specific literature supports plausibility and
   conservatism of the corridor, but does not replace the Dataselector-internal
   selection policy rationale.

## Current Operational Policy

The active thesis production policy is:

1. `selection.autoscale_n_samples_mode = corridor`
2. center at `5%` of `N_eff`
3. bounded local exploration in `4-8%`
4. integer staging with `step = 1`
5. final choice by `minimal_feasible_plateau`

This means the core workflow evaluates every integer `n` inside the bounded
corridor and then selects the smallest feasible `n` that remains within
`plateau_delta` of the best feasible score.

## Primary Policy Rationale

The primary rationale is methodological and internal to the Dataselector thesis
selection workflow:

1. The task is an annotation-budget-constrained core-set selection problem, not
   a "maximize sample count at all costs" problem.
2. The goal is the smallest scientifically defensible annotation set that still
   supports stable downstream model comparison.
3. A bounded corridor around `5%` avoids anchoring the thesis workflow to one
   arbitrary fixed sample size while still keeping selection exploration local
   and interpretable.
4. The plateau rule formalizes: use as few annotations as possible without
   meaningful loss in the selection objective.

This is the normative basis of the policy. The corridor exists because of the
selection contract and annotation-budget logic, not because any one external
paper directly proves `4-8%` for KDR100.

## Supplementary Historical and Internal Evidence

Internal workflow semantics provide supplementary support for this corridor
policy:

1. `parameter_resolution/optuna_autoscale_stage_policy.json` records the
   resolved corridor stages from `N_eff`.
2. `parameter_resolution/optuna_autoscale_best_latest.json` records the
   `minimal_feasible_plateau` selection rule and the selected `n_samples`.
3. The autoscale workflow therefore does not guess a single fixed annotation
   size; it tests all admissible integers in the corridor and keeps the minimum
   feasible near-best solution.

Interpretation:

1. This is evidence that the workflow treats `n_samples` as a bounded
   optimization problem, not as a hard-coded constant.
2. It is not external scientific proof for the exact percentages themselves.
3. It does show why the corridor policy is operationally robust against local
   optimizer noise.

## Supplementary Architecture-Specific Evidence

The following evidence is supplementary and model-facing. It supports the claim
that a corridor centered around `~5%` of `N_eff` is plausible for modern
downstream segmentation training, while remaining conservative enough for the
least data-efficient architecture in the trio.

### Strongest direct support

1. *Few-Shot Segmentation of Historical Maps via Linear Probing of Vision
   Foundation Models* provides the strongest direct support because it is about
   historical maps and foundation-model-based segmentation. It supports the
   claim that very small annotated cores can already be effective when a strong
   pretrained backbone is used.
   Reference: `https://www.arxiv.org/abs/2506.21826`
2. The user-supplied MapSAM-family preprint is also treated as strong
   supplementary support for low-data viability under parameter-efficient
   adaptation of foundation-model backbones. This addendum uses `MapSAM` as the
   active thesis model label and does not claim that Dataselector itself is tied
   to a specific MapSAM version.
   Reference: `https://arxiv.org/html/2510.27547v1`

### Indirect support

1. SegFormer evidence is weaker and more indirect for this thesis context: it
   supports the broader claim that strong pretraining can reduce sensitivity to
   raw data volume, but it is not historical-map-specific evidence for KDR100.
   Reference: `https://ar5iv.labs.arxiv.org/html/2105.15203`

### Conservative cautionary support

1. The cited UNet++ evidence is used conservatively as transfer evidence that a
   less strongly pretrained architecture remains more data-sensitive than
   foundation-model-based alternatives.
2. This does not prove that UNet++ needs exactly `8%` or that `4%` would fail;
   it supports keeping the upper side of the corridor conservative enough that
   UNet++ is not implicitly disadvantaged by an overly aggressive low-data
   policy.
   Reference: `http://arxiv.org/pdf/2410.19623.pdf`

## Current Scientific Reading

1. Primary thesis policy truth remains internal: annotation-budgeted core-set
   selection with bounded corridor exploration and minimal-feasible plateau.
2. The architecture-specific papers do not define the Dataselector policy.
3. They do support the plausibility of a corridor centered around `~5%`:
   foundation-model-based approaches make this center look reasonable rather
   than implausibly small.
4. The same papers also support using the upper side of the corridor as a
   conservative guardrail so that UNet++-style downstream training is not
   under-provisioned by design.

## What This Addendum Does Not Claim

1. It does not claim that the external papers prove `4-8%` for KDR100
   specifically.
2. It does not claim that `4-8%` is universally optimal for all downstream
   segmentation setups.
3. It does not claim that Dataselector directly optimizes for
   SegFormer/MapSAM/UNet++ model metrics.

## Use In Active Docs

Active thesis documentation should cite this addendum when explaining why the
`n_samples` corridor remains centered around `5%` with bounded exploration in
`4-8%`: the core rationale is internal and methodological, while the cited
architecture evidence serves as supplementary downstream plausibility support.

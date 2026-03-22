# Critique Resolution Matrix (2026-02-09)

## Scope and Status Model

- Scope: Valid critique points for thesis-readiness on current HEAD.
- Status values:
  - `open`: reproducible and unresolved
  - `fixed`: resolved in code/tests with evidence
  - `not reproducible`: could not be reproduced on current HEAD
  - `accepted limitation`: known limitation documented with bounded impact

## Baseline Facts (Frozen)

- Date (UTC): 2026-02-09
- Branch: `phase4h/master-closeout`
- Base SHA: `99a7724`
- Canonical metadata source: `data/new_all_tiles.csv`
- Canonical row count: `676`
- Hamburg rows (`city == Hamburg`): `1` (`shortName=KDR_146`)
- Open non-blocking issue: `#72` (`/dev/shm` performance warning)

## Resolution Table

| ID | Critique | Source(s) | File/Function | Repro Command | Current Status | Closure Evidence |
|---|---|---|---|---|---|---|
| C1 | Canonical metadata drift / stale row-count references | `MASTER_FINDINGS_SUMMARY.md`, report set | reports + canonical CSV | `python - <<'PY'\nimport pandas as pd\nprint(len(pd.read_csv('data/new_all_tiles.csv')) )\nPY` | fixed | Reports updated to `676`; facts documented in `ANALYSIS_FACTS_2026-02-09.md` |
| C2 | Hamburg should resolve via city/alias to correct tile | user critique + preselection tests | `dataselector/selection/diversity_selector.py` | `python -m pytest -q tests/test_preselection.py::test_preselection_hamburg_alias_resolves_to_kdr146 -rs` | fixed | Alias chain (`Hamburg -> KDR_146`) plus regression test |
| C3 | Spatial hardcut is silently bypassed | `SCIENTIFIC_VALIDITY_CRITICAL_ISSUES.md`, `ANALYSIS_EXECUTIVE_SUMMARY.md` | `DiversitySelector._apply_spatial_constraint` | `python -m pytest -q tests/test_spatial_constraint.py -rs` | fixed | Hardcut enforced: no fill-on-violation; shortfall explicitly logged; regression tests green on HEAD |
| C4 | Adaptive pipeline contains hardcoded legacy `n_candidates` fallback | code review | `dataselector/workflows/adaptive_pipeline.py` | `rg -n \"defaulting n_candidates\" dataselector/workflows/adaptive_pipeline.py` | fixed | Fallback removed; fail-fast with canonical path expectation |
| C5 | Bootstrap claim overreach (“CI confirms validity”) | report corpus | `dataselector/workflows/bootstrap.py`, report docs | `python -m pytest -q tests/test_bootstrap_module.py -rs` | accepted limitation (bounded) | Terminology changed to robustness interval; reproducibility tests added |
| C6 | Report overclaims without dated evidence blocks | risk/executive/scientific reports | report markdowns | `rg -n "100% confirmed" docs/06_REFERENCE/thesis_decision_evidence -S` | fixed | Evidence-gate sections added; hard claims routed through this matrix |
| C7 | `0/60 non-empty` can be misread as total pipeline failure | report diagnostics | `dataselector/workflows/generate_reports.py` + tests | `python -m pytest -q tests/test_generate_reports_diagnostics.py -rs` | fixed | Diagnostic hint behavior covered by regression test |
| C8 | Runtime pass-governance may drift after refactors | status/test findings | allowlist + unit test | `python -m pytest -q tests/unit/test_runtime_pass_allowlist.py -rs` | fixed | Fingerprint allowlist check remains green on current branch |
| C9 | Exploration confounder: selection target implicitly became `len(metadata)` | distance-policy gate / workflow review | `dataselector/workflows/tune_weights.py` | `python -m pytest -q tests/test_tune_weights_workflow.py -rs` | fixed | LHS points and selection target decoupled; resolution is explicit -> config -> autoscale artifact -> fail-fast (no numeric implicit fallback) |

## Notes

- This matrix is the authoritative reconciliation layer between exploratory reports and current code reality.
- Any new claim in reports must add/update an entry here with reproducible evidence.

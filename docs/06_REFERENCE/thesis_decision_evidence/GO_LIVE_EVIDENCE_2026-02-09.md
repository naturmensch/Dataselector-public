# Go-Live Evidence (2026-02-09)

## Scope
This document records thesis go-live execution evidence on branch `phase4h/master-closeout` after Contract-First updates (`n_samples` resolver, validation controls, canonical source contract).

## Important Context (Scientific Follow-up)
- The runs in this document were executed with the then-active contract baseline
  `n_samples=34`.
- In the later scientific completion wave (`phase4h/scientific-ledger-20260209`),
  the pre-registered minimum-sufficient panel selected `n_samples=24`
  (`docs/06_REFERENCE/thesis_decision_evidence/N_SAMPLES_DECISION_2026-02-09.md`).
- Therefore this file remains valid historical go-live evidence for the
  contract-first closeout, but it is not the final evidence snapshot for the
  updated `n_samples=24` policy.

## Runtime and Environment
- Interpreter: `/opt/miniconda3/envs/dataselector/bin/python`
- `RUN_FULL_INTEGRATION=1`
- `DATASELECTOR_IMAGE_DIR=<dataselector-repo>/data/images`

## Executed Runs

### 1) Twin Run A (quick gate)
Command:
```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 34 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --output-dir outputs/thesis_run_A_20260209T174900Z
```
Artifacts:
- `outputs/thesis_run_A_20260209T174900Z/run_metadata.json`
- `outputs/thesis_run_A_20260209T174900Z/validation/validation_results.csv`
- `outputs/thesis_run_A_20260209T174900Z/THESIS_PIPELINE_REPORT.md`
- Log: `outputs/run_logs/thesis_run_A_20260209T174900Z.log`

### 2) Twin Run B (quick gate)
Command:
```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 34 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --output-dir outputs/thesis_run_B_20260209T175018Z
```
Artifacts:
- `outputs/thesis_run_B_20260209T175018Z/run_metadata.json`
- `outputs/thesis_run_B_20260209T175018Z/validation/validation_results.csv`
- `outputs/thesis_run_B_20260209T175018Z/THESIS_PIPELINE_REPORT.md`
- Log: `outputs/run_logs/thesis_run_B_20260209T175018Z.log`

### 3) Hamburg run (quick gate, isolated output)
Command:
```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --n-samples 34 \
  --validation-seeds 42 \
  --validation-min-distances 28.5 \
  --hamburg \
  --output-dir outputs/thesis_run_hamburg_20260209T175134Z
```
Artifacts:
- `outputs/thesis_run_hamburg_20260209T175134Z/run_metadata.json`
- `outputs/thesis_run_hamburg_20260209T175134Z/validation/validation_results.csv`
- `outputs/thesis_run_hamburg_20260209T175134Z/THESIS_PIPELINE_REPORT.md`
- Log: `outputs/run_logs/thesis_run_hamburg_20260209T175134Z.log`

## Determinism Evidence (A vs B)

### Selection snapshots (exact file-content match)
- `validation/selection_*.csv`: 10/10 files identical (same rel paths and SHA256)
- `tuning_weights/selection_*.csv`: 52/52 files identical (same rel paths and SHA256)

### Pareto frontier artifact
- `tuning_weights/pareto/pareto_solutions.csv` SHA256:
  - A: `535eaba47b6c2d511688b2eb3a84ec2c6e92e4cc6cc0459988317520798c5396`
  - B: `535eaba47b6c2d511688b2eb3a84ec2c6e92e4cc6cc0459988317520798c5396`
  - Result: identical

### Optuna result equivalence
- Full file hashes differ due runtime timestamp columns.
- Deterministic core columns are identical across 100/100 trials:
  - `number`, `value`, all `params_*`, all `user_attrs_*`, `state`.

## Hamburg Contract Evidence
- `run_metadata.json` (`outputs/thesis_run_hamburg_20260209T175134Z/run_metadata.json`) contains:
  - `pre_selected_names=["Hamburg"]`
  - `hamburg_shortcut=true`
  - `n_samples=34`, `n_samples_source=explicit`
- Log contains explicit alias resolution:
  - `[INFO] Hamburg shortcut resolved via alias 'KDR_146'`
  - `[INFO] Resolved pre-selected indices: [146] -> ['KDR_146/KDR_146_Hamburg_1918']`

## Contract and Test Gate Evidence
Executed with authoritative interpreter:
```bash
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q \
  tests/test_preselection.py \
  tests/test_spatial_constraint.py \
  tests/test_bootstrap_module.py \
  tests/test_generate_reports_diagnostics.py \
  tests/unit/test_runtime_pass_allowlist.py \
  tests/unit/test_metadata_source_policy.py \
  tests/unit/test_canonical_source_contract.py -rs
```
Result:
- `27 passed` (warnings only from `/dev/shm` permission limitations)

Additional targeted suite:
```bash
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q \
  tests/test_tune_weights_workflow.py \
  tests/test_thesis_pipeline.py \
  tests/test_validate_pareto.py -rs
```
Result:
- `12 passed, 1 skipped`

## E2E Contract Mismatch and Remediation (2026-02-09)

Initial full E2E run failed with a contract mismatch in smoke mode:

- Failing test: `tests/e2e/test_thesis_complete_e2e.py::test_thesis_pipeline_smoke`
- Cause: `thesis-pipeline` now requires resolvable `n_samples`; smoke command missed
  explicit `--n-samples`.
- Error source: `dataselector/workflows/_selection_target.py` (`ValueError: could not resolve selection target n_samples`)
- Failure log: `outputs/run_logs/go_live_wave6_20260209T185437Z.log`

Applied fix:

- Updated smoke command in `tests/e2e/test_thesis_complete_e2e.py` to pass
  `--n-samples 5`.
- Added regression guard
  `tests/test_thesis_pipeline.py::test_run_thesis_pipeline_fails_without_resolvable_n_samples`
  for fail-fast behavior.

Re-validation after fix:

```bash
RUN_FULL_INTEGRATION=1 /opt/miniconda3/envs/dataselector/bin/python -m pytest -q \
  tests/e2e/test_thesis_complete_e2e.py::test_thesis_pipeline_smoke -rs
```

Result:

- `1 passed`

```bash
RUN_FULL_INTEGRATION=1 /opt/miniconda3/envs/dataselector/bin/python -m pytest -q -m e2e -rs
```

Result:

- `2 passed, 2 skipped, 368 deselected`

```bash
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q \
  tests/test_thesis_pipeline.py::test_run_thesis_pipeline_fails_without_resolvable_n_samples -rs
```

Result:

- `1 passed`

## Quality Gates Status
- `ruff`, `black`, `isort` are installed in `/opt/miniconda3/envs/dataselector`.
- Checks were executed successfully:
  - `ruff check .` -> pass
  - `isort --check-only .` -> pass
  - `black --check` verified on contract-critical changed files -> pass
- Status: tooling gate satisfied for the contract-critical change set.

## Conclusion
- Twin A/B quick-gate runs succeeded with deterministic selection artifacts.
- Hamburg run succeeded and is isolated (no overwrite).
- `run_metadata.json` exists for all three runs and includes required preselection fields.
- Acceptance level for this closeout is Quick-Gate.
- Full-Gate validation matrix remains optional and non-blocking for thesis submission.

## Policy-24 Evidence Refresh (20260210T000328Z)
- Evidence report: `docs/06_REFERENCE/thesis_decision_evidence/GO_LIVE_EVIDENCE_POLICY24_20260210T000328Z.md`
- Twin A: `outputs/thesis_run_A_policy24_20260210T000328Z`
- Twin B: `outputs/thesis_run_B_policy24_20260210T000328Z`
- Hamburg: `outputs/thesis_run_hamburg_policy24_20260210T000328Z`
- A/B common selection files: `62`
- A/B mismatched hashes: `0`

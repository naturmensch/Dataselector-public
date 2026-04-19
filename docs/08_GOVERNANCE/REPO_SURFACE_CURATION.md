# Repository Surface Curation

This note defines the **active vs. secondary vs. historical** surface for the
Dataselector showcase-release state. It exists to make pruning, demotion, and
archive decisions explicit instead of relying on tribal knowledge.

## 1. Canonical active surface

These paths define the current thesis-v2 contract and should remain the public
default story of the repository.

- `python -m dataselector thesis-orchestrate`
- `python -m dataselector thesis-pipeline`
- `config/pipeline_config.yaml`
- snapshot/selection freeze artifacts and `selection_contract.json`
- explicit CRS provenance in `thesis_repro`
- optional Phase 5 handoff bundle
- active policy, method, and evidence docs
- active governance and contract tests

## 2. Secondary active surface

These paths remain valid and useful, but they are **not** the default thesis
entry story. They may remain in the repository and test suite when they still
protect a live behavior.

- Optuna-related workflows and smoke coverage used for parameter search or
  reproducibility
- selected audit/comparison helpers and evidence reruns
- operational copy/export helpers
- compatibility or shim coverage that still protects a live boundary
- package API and reference material used for secondary experiments

Secondary active components should be documented as optional or advanced, not
presented as equivalent to the canonical thesis path.

## 3. Historical / archived surface

These paths may remain for traceability or historical reference, but they
should not shape the active repository narrative.

- shell-heavy XXL/monitor closeout workflows that no longer define the default
  thesis story
- merge-era cleanup or migration notes
- superseded script-era instructions
- old tests preserving dead contracts
- historical configs and historical run narratives

Historical material belongs in archive or explicitly historical sections and
must not be promoted as current defaults.

## 4. Directory-level map

### Docs

- Active entry docs:
  - `README.md`, `README_EN.md`
  - `../../00_OVERVIEW/OVERVIEW.md`
  - `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
  - `../../08_GOVERNANCE/METHODOLOGY.md`
  - `../../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md`
  - `docs/CONFIG_POLICY.md`
  - `../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md`
  - `docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md`
  - `docs/N_SAMPLES_EVIDENCE_ADDENDUM.md`
- Secondary active docs:
  - package/API/reference material
  - `docs/06_REFERENCE/thesis_decision_evidence/` as the stable repo-side home
    for tracked thesis decision evidence
  - `docs/06_REFERENCE/scripts_reference.md` as a curated secondary / historical
    reference, not a default entrypoint
- Historical docs:
  - `docs/07_ARCHIVE/`
  - legacy generated or closeout reporting under `docs/reports/`
  - `docs/reports/` is outside the default `docs-link-check` gate and should be
    treated as historical/generated reporting output

### Scripts

- Active operational wrappers:
  - `scripts/handoff_check.sh`
  - `scripts/copy_selection_tiles.py`
  - `scripts/check_docs_cli_drift.py`
  - `scripts/check_dependency_pins.py`
- Secondary active helper families:
  - evidence and decision reruns such as `compare_min_distance_policies.py`,
    `compare_seed_vs_unseed.py`, `seed_benchmark.py`
  - retained monitoring/inspection helpers such as `dataselector generate-monitor`
    and `scripts/monitor_state.py`
  - repro helpers such as `reproduce_min_distance_decision.sh`,
    `reproduce_n_samples_decision.sh`, `snapshot_final_config.py`
  - analysis or local export helpers such as `profile_selection.py`,
    `uncertainty_quantification.py`, `validate_*`, and
    `run_thesis_orchestrate_double.sh`
- Historical / closeout script families:
  - `scripts/phase4h/`
  - archived XXL/monitor/recovery shell helpers
  - older thesis or experiment runners that should no longer define the active
    repo story

### Tests

- Canonical active tests:
  - `tests/unit/` contract and governance checks
  - `tests/test_thesis_pipeline.py` as the long canonical pipeline gate
- Secondary active tests:
  - selected root-level workflow smoke or compatibility coverage that still
    protects live behaviors
  - opt-in `tests/integration/` and `tests/e2e/`
- Historical tests:
  - `tests/archive/`
  - removed skip-only merge artifacts should stay removed

### Config and policy

- Active defaults:
  - `config/pipeline_config.yaml`
  - thesis-v2 policy and evidence docs
- Historical reference:
  - `config/pipeline_config.best_trial_70.yaml`
  - archived reports or older decision narratives that are no longer current

### Archive zones

- `docs/07_ARCHIVE/`: repo-facing historical documentation
- `tests/archive/`: repo-facing historical tests
- `docs/06_REFERENCE/thesis_decision_evidence/`: active repo-side evidence zone
  for current thesis parameter decisions
- `docs/reports/`: generated or preserved historical reports, not current
  authority and not part of the active workflow surface
- `archive/`: retained repo history material
- `archive_local/`: local migration backups and cleanup mass; useful for
  forensics, but not authoritative and not part of the active entry surface

## 5. Practical curation rules

1. **Prune before repair.**
   If a test, doc, or workflow is no longer in the active or retained
   secondary surface, do not spend time repairing it first.

2. **Archive/demote by default.**
   Prefer demotion or archival over deletion unless something is clearly dead,
   superseded, and unreferenced.

3. **Keep scripts thin.**
   Wrapper scripts may remain when useful, but scientific core logic belongs in
   the package.

4. **Keep active docs sharp.**
   Active entry docs must tell one thesis-v2 story: canonical runtime,
   ownership boundary, explicit CRS provenance, snapshot authority, and
   optional post-freeze packaging.

5. **Keep aggregate gates meaningful.**
   `make format-check` and `make test` should reflect the retained
   active/secondary surface, not a pile of historical leftovers.

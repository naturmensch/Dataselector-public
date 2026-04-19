# Scripts Reference (Secondary / Historical)

This document is a **secondary / historical** reference for the `scripts/`
surface. It does **not** define the canonical thesis workflow.

Authoritative runtime guidance lives in:

- `README.md`
- `docs/00_OVERVIEW/OVERVIEW.md`
- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
- `docs/08_GOVERNANCE/REPO_SURFACE_CURATION.md`

## 1. Canonical CLI-first paths

For release-grade thesis work, prefer the package commands over script entry
points:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>

micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml

micromamba run -n dataselector python -m dataselector generate-monitor \
  --run-dir outputs/runs/<run_id>

micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m dataselector thesis-build-annotation-plan \
  --run-dir outputs/runs/<run_id>
```

If a workflow already exists in the package, the CLI path is canonical and the
script layer is secondary at most.

## 2. Current script surface

### Active operational wrappers

These remain useful and align with the current repo story:

| Script | Role |
|---|---|
| `scripts/handoff_check.sh` | Thin wrapper around the package handoff bundle logic |
| `scripts/copy_selection_tiles.py` | Local QGIS/GIS export helper for selected tiles |
| `scripts/check_docs_cli_drift.py` | Detects drift between docs and registered CLI surface |
| `scripts/check_dependency_pins.py` | Dependency pin governance helper |

### Secondary active helpers

These scripts still support live work, but they should be documented as
optional, evidence-facing, or advanced:

- retained reporting/inspection helpers such as `dataselector generate-monitor`
  and `scripts/monitor_state.py`
- decision/evidence reruns such as `compare_min_distance_policies.py`,
  `compare_seed_vs_unseed.py`, `seed_benchmark.py`
- reproducibility helpers such as `reproduce_min_distance_decision.sh`,
  `reproduce_n_samples_decision.sh`, `snapshot_final_config.py`
- analysis helpers such as `profile_selection.py`,
  `uncertainty_quantification.py`, `validate_*`
- operational helpers such as `run_thesis_orchestrate_double.sh`

### Historical / closeout automation

These paths remain useful for traceability or deep-dive debugging, but they are
not part of the default release story:

- `scripts/phase4h/` closeout automation
- archived XXL/monitor/recovery helpers now bundled under
  `docs/07_ARCHIVE/legacy_xxl_ops/`
- older runner scripts such as `run_complete_thesis_pipeline.sh`,
  `phase4_runner.sh`, `run_full_experiment.sh`
- legacy shell-oriented monitoring wrappers that no longer define the active
  thesis workflow

## 3. Rules for keeping scripts

1. `scripts/*` are wrappers or operational helpers, not homes for scientific
   core logic.
2. If the workflow lives in `dataselector/*`, reference the package/CLI path
   first.
3. Secondary active scripts may remain, but they should not compete with
   `thesis-orchestrate` or `thesis-pipeline` in active docs.
4. Historical scripts should be called historical explicitly or be referenced
   via archive/closeout docs.

## 4. Archive and history boundary

- Repo-facing historical documentation belongs under `docs/07_ARCHIVE/`.
- Historical tests belong under `tests/archive/`.
- `docs/06_REFERENCE/thesis_decision_evidence/` is the stable repo-side home
  for active tracked thesis decision evidence.
- Local migration backups under `archive_local/` are useful for forensics, but
  they are not authoritative runtime guidance.
- `scripts/phase4h/README.md` documents the preserved Phase4H closeout
  automation and should be treated as historical operational context.

## 5. Practical reader guidance

If you are deciding whether to use a script:

1. If the task is thesis selection/freeze: use `thesis-orchestrate` or
   `thesis-pipeline`.
2. If the task is post-freeze handoff packaging: use the package command or the
   thin `scripts/handoff_check.sh` wrapper.
3. If the task is local GIS export: `scripts/copy_selection_tiles.py` is an
   acceptable operational helper.
4. If the task is evidence rerun, old monitoring, or closeout automation: treat
   the script as secondary or historical unless an active doc explicitly says
   otherwise.

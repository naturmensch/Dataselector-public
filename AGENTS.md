# AGENTS

## Working Contract (Phase 4H closeout)
1. Canonical production path: `python -m dataselector thesis-pipeline`
2. Canonical runtime: `micromamba run -n dataselector python -m dataselector <command>`
3. Script policy: `scripts/*` are wrappers/orchestrators; no duplicated scientific core logic
4. Active output root: `outputs/runs/`
5. Active config: `config/pipeline_config.yaml`
6. Historical config: `config/pipeline_config.best_trial_70.yaml` (reference only)

## Default Collaboration Workflow
1. Start each task with a short discovery pass in the real repository surface before making assumptions.
2. After discovery, give a short check-in that states:
   - problem understanding
   - affected surface
   - planned approach
   - task-specific checks
   - optional expanded checks as exact commands
3. Keep scientific core logic in `dataselector/` and treat `scripts/` as wrappers/orchestrators only.
4. Use active docs/config/runtime as defaults; use historical docs/config only as explicit reference material.

## Validation Policy
1. Default verification is task-specific: run only the smallest relevant checks for the area being changed.
2. Do not silently widen validation scope; when broader coverage is useful, present the exact commands explicitly.
3. Typical expanded commands:
   - Runtime and governance:
     - `micromamba run -n dataselector python -m dataselector check-runtime-readiness`
     - `micromamba run -n dataselector python -m dataselector check-script-wrappers --strict`
   - Policy and documentation:
     - `micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py`
     - `micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py`
     - `micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py`
   - Canonical thesis gate:
     - `micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py`
   - Full local quality check:
     - `make format-check`
     - `make test`
4. Every closeout must say what was run, what was not run, and which expanded commands are recommended next when relevant.

## Fast Setup
```bash
make env-create
micromamba run -n dataselector python -m pip install -e .
micromamba run -n dataselector python -m pip install -r requirements-cpu.txt
```

## Canonical Commands
### Expanded Runtime and Governance Checks
```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
```

### Expanded Local Gates
```bash
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py
micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py
```

### Full Local Quality Check
```bash
make format-check
make test
```

## Scientific Orchestration (recommended)
Use orchestration when you want precompute + snapshot + run in one controlled flow.
```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

## Foreground Run with Timestamped Logs
```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="outputs/runs/thesis_orchestrate_${ts}"
log="outputs/runs/thesis_orchestrate_${ts}.log"
mkdir -p outputs/runs

env XDG_CACHE_HOME=/tmp/mamba-cache \
  micromamba run -n dataselector python -u -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir "$run_dir" \
  2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush(); }' | tee "$log"
```

## Scientific Parameter Policy (must hold)
1. Critical parameters are either `computed` or explicit `policy-tagged`
2. Snapshot validation is mandatory unless `--force`
3. If `--force` is used, metadata must record override flag and reason
4. Scientific rationale and provenance must be reflected in:
   - `docs/08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md`
   - `docs/08_GOVERNANCE/CONFIG_POLICY.md`
   - `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## Repository Map (active)
- `dataselector/` package and canonical logic
- `tests/` pytest suite
- `scripts/` wrappers and operational helpers
- `config/` active and historical configs
- `docs/` authoritative docs and archive docs
- `docs/07_ARCHIVE/` archived legacy documentation

## Footguns
1. Do not re-enable GitHub Actions until billing is restored (`.github/workflows-disabled/`)
2. Do not commit generated run artifacts from `outputs/`
3. Do not add scientific core logic into `scripts/*.py`
4. Do not use historical docs/config as defaults

## Definition of Done
1. Discovery and short check-in happened before substantial implementation/review work
2. Task-specific verification was run for the affected surface
3. Output path and config policy remain canonical
4. Documentation stays consistent with active architecture
5. Any broader recommended validation is provided as exact commands

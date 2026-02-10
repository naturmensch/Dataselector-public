# AGENTS

## Working Contract (Phase 4H closeout)
1. Canonical production path: `python -m dataselector thesis-pipeline`
2. Canonical runtime: `micromamba run -n dataselector -- <command>`
3. Script policy: `scripts/*` are wrappers/orchestrators; no duplicated scientific core logic
4. Active output root: `outputs/runs/`
5. Active config: `config/pipeline_config.yaml`
6. Historical config: `config/pipeline_config.best_trial_70.yaml` (reference only)

## Fast Setup
```bash
make env-create
micromamba run -n dataselector python -m pip install -e .
micromamba run -n dataselector python -m pip install -r requirements-cpu.txt
```

## Canonical Commands
### Runtime and Governance
```bash
micromamba run -n dataselector -- python -m dataselector check-runtime-readiness
micromamba run -n dataselector -- python -m dataselector check-script-wrappers --strict
```

### Core Local Gates
```bash
micromamba run -n dataselector -- pytest -q tests/unit/test_no_legacy_script_references.py
micromamba run -n dataselector -- pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector -- pytest -q tests/unit/test_authoritative_docs_consistency.py
micromamba run -n dataselector -- pytest -q tests/test_thesis_pipeline.py
```

### Full Local Quality Check
```bash
make format-check
make test
```

## Scientific Orchestration (recommended)
Use orchestration when you want precompute + snapshot + run in one controlled flow.
```bash
micromamba run -n dataselector -- \
  python -m dataselector thesis-orchestrate \
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
  micromamba run -n dataselector -- \
  python -u -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir "$run_dir" \
  2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush(); }' | tee "$log"
```

## Scientific Parameter Policy (must hold)
1. Critical parameters are either `computed` or explicit `policy-tagged`
2. Snapshot validation is mandatory unless `--force`
3. If `--force` is used, metadata must record override flag and reason
4. Scientific rationale and provenance must be reflected in:
   - `docs/PARAMETER_POLICY_LEDGER.md`
   - `docs/CONFIG_POLICY.md`
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
1. Runtime readiness and strict wrapper checks pass
2. Core policy/doc tests pass
3. Thesis pipeline tests pass
4. Output path and config policy remain canonical
5. Documentation stays consistent with active architecture

# Reproducibility Guide (Current)

This is the active reproducibility guide for the CLI-first architecture.

## Canonical Contracts

1. Canonical runtime: `micromamba run -n dataselector <command>`
2. Canonical production path: `python -m dataselector thesis-pipeline`
3. Canonical output root: `outputs/runs/<timestamped_run>/`
4. Canonical active config: `config/pipeline_config.yaml`
5. Orchestrated runs must use a fresh output directory (non-empty target dirs fail-fast)
6. Feature caches are scope-controlled:
   - `feature_cache.scope=global_shared` (thesis default) stores immutable cache objects in `outputs/cache/features/<cache_key>/`
   - `feature_cache.scope=run_local` stores cache objects inside the current run output dir

`scripts/exec_in_env.sh` remains available as a compatibility wrapper, but it is
not the primary guidance path.

## Minimal Reproducible Run

```bash
micromamba run -n dataselector \
  python -m dataselector thesis-pipeline \
    --compute-params \
    --snapshot-config \
    --output-dir outputs/runs/thesis_repro_smoke
```

## Parameter Resolution and Snapshot Flow

The thesis pipeline resolves critical parameters centrally. Required flags:

1. `--compute-params`: compute unresolved critical parameters
2. `--use-params <snapshot.yaml>`: load + validate an existing snapshot
3. `--snapshot-config`: write `final_config.yaml` alias in the run dir
4. `--no-auto-continue`: stop after resolution stage (audit mode)
5. `--force`: allow snapshot mismatch continuation with explicit metadata flag

Each run writes:

1. `run_metadata.json`
2. `final_config_<timestamp>.yaml`
3. optional `final_config.yaml` (stable alias; convenience only)
4. `manifest/artifact_hashes.json` (run-local artifact hash manifest)
5. cache provenance in run metadata (`cache_scope`, cache key/hash references)

Snapshots include:

1. `hashes.parameters_hash`
2. `hashes.snapshot_content_sha256`
3. additive provenance blocks:
   - `selection._provenance`
   - `clustering._provenance`
   - `feature_extraction._provenance`

## Verification Commands

```bash
micromamba run -n dataselector \
  python -m dataselector check-runtime-readiness

micromamba run -n dataselector \
  python -m dataselector check-script-wrappers --strict

micromamba run -n dataselector \
  pytest -q tests/test_thesis_pipeline.py
```

## Historical Note

Older references to `src/*` modules and `outputs/experiments/*` were part of a
legacy migration phase and are non-authoritative.

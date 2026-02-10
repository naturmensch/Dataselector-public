# Troubleshooting & FAQ

Common issues and quick fixes for the current CLI-first architecture.

1. `wandb` not installed:
   - `micromamba run -n dataselector pip install wandb`
2. Missing feature cache or mismatch:
   - run the canonical pipeline path and let it resolve/cache features:
   - `micromamba run -n dataselector python -m dataselector thesis-pipeline --compute-params --no-auto-continue`
3. Snapshot validation fails:
   - inspect `run_metadata.json` and snapshot hashes
   - only use `--force` when audit-signoff explicitly allows it
4. Monitor resume failures:
   - check run dir under `outputs/runs/`
   - validate Optuna DB integrity (`PRAGMA integrity_check`)

If unresolved, open an issue with:

1. command used
2. relevant `run_metadata.json`
3. relevant log excerpt

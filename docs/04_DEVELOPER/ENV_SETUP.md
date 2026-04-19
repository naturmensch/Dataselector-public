# Environment & Reproducibility Guide

This guide documents the recommended environment setup and reproducible installation workflow for Dataselector.

## Why use micromamba / conda?
- Prebuilt scientific packages (pandas, scipy, PyTorch) are distributed as conda packages for many platforms.
- `micromamba` is a fast solver and works the same as `conda` while being much faster for large environments.
- We provide an `environment.yml` (canonical spec) and `locks/conda-lock-*.lock` (generated lockfiles) for reproducible installs.

## Quick start (recommended)

```bash
# 1) Install micromamba (user-local; no base env needed)
${SHELL} <(curl -L micro.mamba.pm/install.sh)

# 2) Create the env from environment.yml
micromamba create -n dataselector -f environment.yml -y

# 3) Install pip-only extras in the canonical runtime
micromamba run -n dataselector pip install -r requirements-cpu.txt
```

## Using the provided lockfiles

We include generated lockfiles under `locks/` (e.g., `locks/conda-lock-linux-64.lock`) to pin exact versions for the platform. To install from a lockfile:

```bash
# Install to env name 'dataselector' (or another name)
micromamba create -n dataselector -f locks/conda-lock-linux-64.lock
```

Note: pip-only extras listed in `requirements-cpu.txt` are not included in the lockfile (`conda-lock` cannot parse pip `-r` references). Keep a small `pip install -r requirements-cpu.txt` step if needed.

Tip: Use the helper `./scripts/setup_local_venv.sh` — it now prefers `conda-lock`/`micromamba` for reproducible installs and falls back to creating a local `.venv` if no conda/micromamba is available.

CI note: Generated runtime artifacts (reports, diagnostic plots) are intentionally git-ignored locally; the `geo-smoke` CI job publishes these outputs as job artifacts for inspection.

## Running repository scripts inside the canonical env

Canonical invocation is `micromamba run -n dataselector <command>`.
For compatibility, `scripts/exec_in_env.sh` remains available and delegates to
micromamba/conda where possible.

Examples:

```bash
# Dry-run the adaptive pipeline inside the canonical env
micromamba run -n dataselector python scripts/run_adaptive_pipeline.py --dry-run --yes --n-lhs 5 --n-trials 5

# Run a full local experiment in background with safe thread caps
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
micromamba run -n dataselector nohup ./scripts/run_full_experiment.sh --adaptive --n-trials 300 --n-boot 300 --yes > outputs/runs/run_full_top.log 2>&1 &
```

## Generating lockfiles

To generate lockfiles locally (used by CI), run:

```bash
./scripts/generate_conda_lock.sh --platform linux-64
```

This will create sanitized lockfiles under `locks/`.

## Intensive run & monitoring

For full experiments we recommend running inside the canonical `dataselector` environment. Use `nohup`/`tmux`/`screen` so runs survive terminal disconnects and store logs in `outputs/runs`.

Example (background, moderate budgets for smoke/full test):

```bash
# prefer the canonical env (micromamba or .venv)
micromamba run -n dataselector bash -lc "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; nohup ./scripts/run_full_experiment.sh --adaptive --n-trials 50 --n-boot 50 --n-candidates 200 --yes > outputs/runs/run_full_mamba_smoke.log 2>&1 & echo \$!"

# After this you will receive the PID. To monitor logs:
tail -f outputs/runs/run_full_mamba_smoke.log
```

Watch helper: `scripts/watch_experiment.sh`

We provide a small watcher script to simplify monitoring of experiments. It automatically picks the latest `outputs/runs/run_*` folder if no path is given, lists available `.log` files and follows them live. Useful options:

- `--filter '<regex>'`  : Only show lines that match the regex (live, use quotes); e.g. `--filter 'FAILED|Traceback'`.
- `--show-proc`         : Try to locate related processes (pgrep/ps) and print PID/%CPU/%MEM/ETIME for found PIDs (no extra dependencies required).
- `--lines N`           : Set how many lines to show initially (default: 200).

Example usages:

```bash
# Follow the latest run and filter for errors:
./scripts/watch_experiment.sh --filter 'FAILED|Traceback'

# Follow a specific run and show related processes:
./scripts/watch_experiment.sh outputs/runs/run_20260116T021615Z --show-proc
```

Quick smoke test (recommended):

1) Start a short run (smoke):

```bash
micromamba run -n dataselector bash -lc "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; nohup ./scripts/run_full_experiment.sh --adaptive --n-trials 5 --n-boot 5 --n-candidates 50 --yes > outputs/runs/run_smoke.log 2>&1 & echo \$!"
```

2) Monitor with the watcher and filter for problems:

```bash
./scripts/watch_experiment.sh --filter 'FAILED|Traceback|ERROR|Exception' --lines 100
```

Notes & tips:
- Set conservative thread caps (`OMP_NUM_THREADS` / `MKL_NUM_THREADS`) to avoid CPU oversubscription.
- Use `nohup` or run inside `tmux` for long experiments; the wrapper will make sure the canonical env is used.
- Check `outputs/runs/run_<TIMESTAMP>/` for step-specific logs (optuna.log, bootstrap.log).

## Troubleshooting

- If a package fails to install via pip (e.g., `pandas` for Python 3.13), prefer creating the conda env with Python 3.11 (the recommended pinned version).
- Use `./scripts/create_env.sh` to create or refresh the env in a micromamba-first way (`./scripts/create_env.sh dataselector 3.11`).
- If you see mismatching shared libs when importing torch, try reinstalling torch using the CPU wheels from the official PyTorch index (`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`) inside the conda env.

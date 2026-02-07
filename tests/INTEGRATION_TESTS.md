# Running Full Integration / E2E Tests

This document describes how to run the heavy/full E2E tests locally or in CI.

Purpose
- Ensure the real pipeline (feature extraction, UMAP, Optuna, Weight Sweep, Monitor) works end-to-end.
- These tests can take a long time and require native dependencies (Numba/UMAP, possibly GPU).

Local run (recommended pre-conditions)
- Use the `dataselector` conda environment described in `environment.yml` or `requirements.txt`.
- Ensure `numba` and `umap-learn` are compatible with your NumPy version.

Run a full E2E test locally (manual)

1) Activate env (example) with conda/mamba:

   conda env create -f environment.yml -n dataselector
   conda activate dataselector

# Quick compatibility check

Run the environment compatibility check which prints versions and exits non-zero when an incompatibility is detected:

   python -m dataselector check-env

If it prints errors about NumPy/Numba versions, recreate the environment from `environment.yml` and re-check.

2) Run the full integration tests (explicit marker):

   export RUN_FULL_INTEGRATION=1
   pytest -q -m integration

Real-image strategy (local only)
- CI is metadata-compatible and must not depend on private image folders.
- For local real-image runs, point the loader to your private image directory:

   export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images

- This keeps private image assets out of the repository while allowing strict local feature extraction.

Notes for CI
- `.github/workflows/integration.yml` contains a gated `e2e-integration` job that runs on `workflow_dispatch` or nightly schedule.
- For reliable runs, prefer a self-hosted runner or a runner with the same conda stack (the official GitHub runners may require time to install native deps).

Debugging
- If the E2E job fails with Numba/NumPy incompatibilities, ensure the environment uses a NumPy version supported by the installed Numba.
- Use `pytest -q tests/test_full_pipeline_comprehensive.py::test_full_pipeline_simulation` as a fast debug ground-truth (this test is lightweight and uses stubs).

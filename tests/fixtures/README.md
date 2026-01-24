# Test Fixtures

This directory contains shared fixtures and helpers used across the test suite.

Guidelines
- Prefer using fixtures from `tests/conftest.py` for temporary directories and standard layouts.
- Add small, focused helper functions for constructing in-memory datasets (e.g., `make_dummy_metadata`, `make_features`).
- Keep fixtures deterministic (use `seed=` arguments and set `numpy` random seeds when needed).

Common fixtures
- `make_dummy_metadata(n)`: returns a lightweight pandas DataFrame with `n` rows and the columns expected by the pipeline.
- `make_features(n, dim=32, seed=0)`: returns an `n x dim` numpy array with deterministic values for selection tests.

Usage
- Import fixtures in tests by requesting them as function arguments in the test signatures.
- Document new fixtures here with a short example of usage.

Maintenance
- Keep fixtures small and focused; if a fixture grows complex, consider moving it to `tests/utils.py` with tests.
- Add a short docstring to each fixture in code to clarify intent.
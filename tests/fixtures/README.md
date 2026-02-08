# Test Fixtures

This directory contains shared fixtures and helpers used across the test suite.

## Directories

- **real_tiles/**: Public mini-fixture for CI and local integration sanity checks
  - Contains 5 KDR tiles (`KDR_001` ... `KDR_005`) as PNG + aux.xml
  - Current footprint: ~84 MB (Git LFS-managed)
  - See `real_tiles_metadata.csv` for corresponding fixture metadata

## Fixture Policy
- `tests/fixtures/real_tiles/` is intentionally versioned as a reproducible, small baseline fixture.
- It is **not** your full thesis dataset.
- Private/full image corpora stay local-only and must be provided through `DATASELECTOR_IMAGE_DIR`.
- Tests marked `real_images` must still honor the env-gate policy from `tests/INTEGRATION_TESTS.md`.

## Guidelines
- Prefer using fixtures from `tests/conftest.py` for temporary directories and standard layouts.
- Add small, focused helper functions for constructing in-memory datasets (e.g., `make_dummy_metadata`, `make_features`).
- Keep fixtures deterministic (use `seed=` arguments and set `numpy` random seeds when needed).

## Common Fixtures
- `make_dummy_metadata(n)`: returns a lightweight pandas DataFrame with `n` rows and the columns expected by the pipeline.
- `make_features(n, dim=32, seed=0)`: returns an `n x dim` numpy array with deterministic values for selection tests.
- `sample_csv`: Generates 50 synthetic tiles with valid German coordinates
- `real_tiles_csv`: Returns path to CSV with 5 real KDR tiles (requires real_tiles/ directory)

## Usage
- Import fixtures in tests by requesting them as function arguments in the test signatures.
- For real tile tests, mark with `@pytest.mark.real_tiles`
- Document new fixtures here with a short example of usage.

## Maintenance
- Keep fixtures small and focused; if a fixture grows complex, consider moving it to `tests/utils.py` with tests.
- Add a short docstring to each fixture in code to clarify intent.
- Real tiles are version-controlled via Git LFS to manage size.
- Do not add new large tiles here without updating fixture policy thresholds and justification.

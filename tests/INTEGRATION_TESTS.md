# Running Integration and Real-Image Test Profiles

This document defines the authoritative test profiles after Phase 4 dependency
partitioning. It keeps CI deterministic while preserving strict local real-image
validation.

## Authoritative Runtime

Use the project environment (micromamba/conda env `dataselector`), not system Python.

```bash
micromamba run -n dataselector python -m dataselector --help
```

## Profile Matrix

| Profile | Markers/Selector | Requires | Intended Use |
|---|---|---|---|
| CI parity default | `-k "not real_images"` | project env only | Matches default CI expectations without private image data |
| Full E2E (gated) | `-m e2e` | `RUN_FULL_INTEGRATION=1` | Long-running end-to-end validation |
| Real images (local only) | `-m real_images` | `DATASELECTOR_IMAGE_DIR` | Strict local tests with private image assets |
| Integration-heavy subset | `-m integration` | native deps (optuna/numba/umap etc.) | Focused dependency-sensitive integration runs |

`real_tiles` is maintained as a legacy alias marker and is normalized to `real_images`
in collection hooks.

Note: `-m "e2e and real_images"` usually selects no tests because the current suite
separates those marker sets by design.

## Contract Gates

1. E2E tests are opt-in and skipped unless:

```bash
export RUN_FULL_INTEGRATION=1
```

2. Real-image tests are opt-in and skipped unless:

```bash
export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images
```

3. CI must not require private image assets.

## Canonical Commands

### 1) CI-parity local run (default)

```bash
micromamba run -n dataselector python -m pytest -q tests -k "not real_images"
```

### 2) Guard checks (always)

```bash
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_ci_nonblocking_allowlist.py
```

### 3) Full E2E gate (explicit)

```bash
export RUN_FULL_INTEGRATION=1
micromamba run -n dataselector python -m pytest -q -m e2e
```

### 4) Real-image run (local only)

```bash
export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images
micromamba run -n dataselector python -m pytest -q -m real_images
```

### 5) Real-image gate behavior check (without env var)

```bash
micromamba run -n dataselector python -m pytest -q tests/e2e/test_build_tiles_real.py -m real_images -rs
```

Expected: skipped with explicit `DATASELECTOR_IMAGE_DIR` requirement.

## Notes

- No private image assets are stored in the repository.
- If E2E tests skip unexpectedly, verify `RUN_FULL_INTEGRATION`.
- If real-image tests skip unexpectedly, verify `DATASELECTOR_IMAGE_DIR` exists.

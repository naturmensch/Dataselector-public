# Thesis Pipeline How-To (Authoritative)

This is the authoritative runbook for thesis-grade tile selection before
annotation.

## 1) Preconditions

1. Use the authoritative environment:

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector --help
```

2. Keep private images local-only (never commit image assets).
3. Keep spatial hard-cut contract (`ul_x/ul_y/lr_x/lr_y`), no legacy `N` path.

## 2) Required Runtime Gates

```bash
export RUN_FULL_INTEGRATION=1
export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images
```

## 3) Sampler Decision (Before Production Run)

Use the decision rule:

- `docs/status/thesis_sampler_decision_rule_2026-02-08.md`

This determines:

1. exploration sampler (`lhs` or `sobol`)
2. Optuna sampler (`TPESampler`, `QMCSampler`, `CmaEsSampler`)
3. fixed `n_initial_final` and run seed.

## 4) End-to-End Thesis Flow (Copy/Paste)

### 4.1 Build metadata from local images

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector build-tiles \
  --image-dir "$DATASELECTOR_IMAGE_DIR" \
  --out data/new_all_tiles.csv
```

### 4.2 Validate raster/CSV alignment

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector align-audit \
  --csv data/new_all_tiles.csv \
  --base-dir "$DATASELECTOR_IMAGE_DIR" \
  --out outputs/align_audit.json
```

### 4.3 Dry-run (safety check)

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --dry-run \
  --output-dir outputs/thesis_preflight
```

### 4.4 Production run A

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --output-dir outputs/thesis_run_A
```

### 4.5 Production run B (determinism check)

```bash
/opt/miniconda3/envs/dataselector/bin/python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --seed 42 \
  --output-dir outputs/thesis_run_B
```

## 5) Required Validation Gates

### 5.1 E2E and real-image tests

```bash
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q -m e2e -rs
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q -m real_images -rs
```

### 5.2 Guard test

```bash
/opt/miniconda3/envs/dataselector/bin/python -m pytest -q \
  tests/unit/test_no_legacy_script_references.py
```

### 5.3 Determinism gate

The annotation gate is only open when:

1. `outputs/thesis_run_A` and `outputs/thesis_run_B` both exist.
2. selected tile IDs are identical in A and B.
3. `run_metadata.json` exists in run outputs and includes:
   - commit SHA
   - command
   - seed
   - execution profile
4. sampled `image_path` entries resolve to real files.

## 6) Go/No-Go Before Annotation

Start annotation only if all are true:

1. Latest `main` core workflows are green.
2. Feature-gap scan reports no `missing_required` and no `implemented_but_misaligned`.
3. Gates in section 5 pass.
4. No open blocking PR/issue.

If a gate fails: open one focused fix branch per root cause. Do not use broad
skip/xfail or `continue-on-error` expansion.


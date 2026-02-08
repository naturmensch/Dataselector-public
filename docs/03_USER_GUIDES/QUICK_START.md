# Quick Start (Authoritative)

This quick start is aligned with the current package CLI.

For the full thesis workflow (including deterministic twin-run gate), use:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## 1) Environment

```bash
git clone <repository-url> Dataselector
cd Dataselector
mamba env create -f environment.yml
conda activate dataselector
python -m pip install -e .
```

## 2) Verify CLI

```bash
python -m dataselector --help
python -m dataselector check-geo
python -m dataselector check-env
```

## 3) Build metadata from images

```bash
python -m dataselector build-tiles \
  --image-dir data/images \
  --out data/new_all_tiles.csv
```

## 4) Run modern thesis flow

```bash
python -m dataselector thesis-sampler-suite --autoscale
python -m dataselector xxl
```

For a single end-to-end thesis run with reproducibility controls, see:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## 5) Useful checks

```bash
python -m dataselector align-audit --csv data/new_all_tiles.csv --base-dir data/images
python -m pytest -q tests/unit/test_no_legacy_script_references.py
```


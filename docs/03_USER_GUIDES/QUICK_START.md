# Quick Start (Authoritative)

This quick start is aligned with the current package CLI.

For the full thesis workflow (including deterministic twin-run gate), use:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## 1) Environment

```bash
git clone <repository-url> Dataselector
cd Dataselector
micromamba create -f environment.yml -n dataselector
micromamba run -n dataselector python -m pip install -e .
```

## 2) Verify CLI

```bash
micromamba run -n dataselector python -m dataselector --help
micromamba run -n dataselector python -m dataselector check-geo
micromamba run -n dataselector python -m dataselector check-env
```

## 3) Build metadata from images

```bash
micromamba run -n dataselector python -m dataselector build-tiles \
  --image-dir data/images \
  --out data/new_all_tiles.csv
```

## 4) Run modern thesis flow

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

For a single end-to-end thesis run with reproducibility controls, see:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## 5) Useful checks

```bash
micromamba run -n dataselector python -m dataselector align-audit --csv data/new_all_tiles.csv --base-dir data/images
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
```

# End-to-End Checklist (Raw Data -> Training Start)

This is the shortest operational checklist for the full thesis flow:

1. local raw data and metadata prepared,
2. canonical Dataselector run executed,
3. optional width calibration with two-click tool completed,
4. handoff artifacts created and verified,
5. server transfer + training-repo checks passed,
6. training/split execution can start.

Use this page for fast execution. Use
[THESIS_PIPELINE_HOWTO.md](THESIS_PIPELINE_HOWTO.md) for full details.

## 0) Preconditions

```bash
micromamba run -n dataselector python -m dataselector --help
```

Required runtime variables:

```bash
export RUN_FULL_INTEGRATION=1
export DATASELECTOR_IMAGE_DIR=/abs/path/to/private/images
```

## 1) Raw Data Readiness

Checklist:

1. Metadata CSV is available and consistent.
2. Raw tile images exist under local private image storage.
3. Required sidecars/auxiliary files are available where needed.

Optional tile index build:

```bash
micromamba run -n dataselector python -m dataselector build-tiles
```

## 2) Run Canonical Selection Pipeline

Recommended canonical run:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/thesis_orchestrated_$(date -u +%Y%m%dT%H%M%SZ)
```

Direct validated-snapshot path (if snapshot already exists):

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml
```

## 3) Optional Width Calibration (Two-Click Tool)

Run this if you need explicit width-policy evidence before final handoff.

```bash
micromamba run -n dataselector python -m dataselector build-width-calibration-roads-source
micromamba run -n dataselector python -m dataselector sync-width-calibration-source
micromamba run -n dataselector python -m dataselector prepare-width-calibration
micromamba run -n dataselector python -m dataselector measure-width-calibration
micromamba run -n dataselector python -m dataselector summarize-width-calibration
```

Operational rules:

1. Only accepted measurements enter policy evidence.
2. Re-run `sync-width-calibration-source` after external geometry edits.

## 4) Build and Verify Handoff Artifacts

Patch-based handoff (recommended for current thesis dataset):

```bash
bash scripts/handoff_check.sh prepare-patches \
  --run-dir outputs/runs/<run_id> \
  --out handoff/<selection_id> \
  --patch-id-file config/patch_filters/<subset>.txt

bash scripts/handoff_check.sh verify-patches \
  --handoff-dir handoff/<selection_id>
```

Tile-based legacy handoff:

```bash
bash scripts/handoff_check.sh prepare \
  --run-dir outputs/runs/<run_id> \
  --out handoff/<selection_id>

bash scripts/handoff_check.sh verify-local \
  --handoff-dir handoff/<selection_id>
```

## 5) Git/GitHub Reproducibility Checkpoint

Before server transfer, ensure authoritative repo state is synchronized.

```bash
git status
git add docs/ config/ dataselector/ scripts/ tests/
git commit -m "docs: update end-to-end thesis flow guidance"
git push origin main
```

Do not commit generated run artifacts from `outputs/`.

## 6) Server-/UniCloud-Bereitstellung und Verify im Trainings-Repo

Operativer Serverstandard:

1. Server-Checkout aktualisieren (`git pull`), damit Code, getrackter Handoff
   und `phase5_final_width_contract.json` verfuegbar sind.
2. Git LFS initialisieren/ziehen (`git lfs install`, `git lfs pull`), damit
   getrackte Handoff-Quicklooks vorhanden sind.
3. Finalen Maskenordner als externes Datenartefakt am erwarteten Serverpfad
   bereitstellen (`data/patch_masks_final_width_calibration_20260418T195314Z/`).

In `masterarbeit-strassenerkennung` ausfuehren:

```bash
python scripts/setup/validate_phase5_final_width_handoff.py \
  --contract /path/to/handoff/<selection_id>/phase5_final_width_contract.json \
  --masks-dir /path/to/final_width_masks \
  --integration-dir data/integration/<integration_id> \
  --require-local-artifacts

bash scripts/setup/handoff_check.sh verify-server-patches \
  --handoff-dir /path/to/handoff/<selection_id> \
  --masks-dir /path/to/final_width_masks

bash scripts/setup/handoff_check.sh materialize-patches \
  --handoff-dir /path/to/handoff/<selection_id> \
  --masks-dir /path/to/final_width_masks \
  --out-root data/integration \
  --split-policy use_handoff \
  --selection-id <integration_id>

bash scripts/setup/handoff_check.sh verify-server \
  --handoff-dir /path/to/handoff/<selection_id> \
  --raw-tiles-dir /path/to/raw/Tiles \
  --masks-dir /path/to/masks
```

Start training/splits only after validator + verify + materialize are green.

## 7) Minimal Governance Gates

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m dataselector docs-link-check
```

## Related

1. [THESIS_PIPELINE_HOWTO.md](THESIS_PIPELINE_HOWTO.md)
2. [PIPELINES.md](PIPELINES.md)
3. [../06_REFERENCE/CLI_COMMAND_CATALOG.md](../06_REFERENCE/CLI_COMMAND_CATALOG.md)
4. [../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md](../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md)

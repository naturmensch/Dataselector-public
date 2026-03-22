# Dataselector

**Algorithmic data selection for the Karte des Deutschen Reiches (KDR100).**

Dataselector freezes a reproducible, reviewable selection contract for
annotating historical map tiles. The repository is centered on the thesis-grade
workflow: selection, provenance, validation, freeze artifacts, and optional
post-freeze handoff packaging for the downstream training repository.

## What this repository owns

Dataselector owns the **selection contract**:

- candidate pool handling and policy-driven tile exclusions
- parameter-resolved, validated thesis selection
- freeze artifacts (`selection_core.csv`, `selection_final_with_cases.csv`,
  `selection_contract.json`)
- reporting, provenance, CRS audit
- optional post-freeze tile/patch handoff packaging

The downstream training repository owns the **evaluation contract**:

- authoritative train/val/test strategy for model training
- the actual segmentation models
- cross-validation and final model comparisons

## Methodological boundary (Thesis Freeze)

1. Selection is architecture-neutral / model-agnostic and optimizes a
   diversity/coverage proxy.
2. The current thesis freeze is a `frozen dataset`; downstream model
   comparisons happen after the freeze.
3. No direct model-metric optimization (SegFormer/MapSAM/UNet++).
4. `alpha_visual` is an optimized parameter, not a hard dominance constraint.
5. A visual-biased or model-aware selection strategy is a separate ablation
   path and requires a new freeze.

## Canonical workflow

Canonical runtime:

```bash
micromamba run -n dataselector python -m dataselector <command>
```

Canonical production path:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

Direct validated-snapshot path:

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml
```

Important operational defaults:

- active config: `config/pipeline_config.yaml`
- canonical run root: `outputs/runs/`
- thesis default: `selection_authority = snapshot_primary`
- optional Phase 5 packaging is **off by default**

Optional integrated Phase 5 packaging:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id> \
  --build-handoffs \
  --patches-per-tile 2 \
  --patch-include-case false
```

Phase 5 is **post-freeze operational packaging**, not reselection.

## Quick start

```bash
make env-create
micromamba run -n dataselector python -m pip install -e .
micromamba run -n dataselector python -m pip install -r requirements-cpu.txt
```

Fast governance gates:

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector pytest -q tests/unit/test_authoritative_docs_consistency.py
```

Long canonical pipeline gate:

```bash
micromamba run -n dataselector pytest -q tests/test_thesis_pipeline.py
```

## Repository status and quality posture

The repository is intended to present a release-grade thesis workflow:

- scientific freeze via `selection_*` and `selection_contract.json`
- explicit CRS provenance in the `thesis_repro` path
- thin wrappers, scientific core logic inside the package
- policy/doc governance enforced by pytest
- optional tile and patch handoff packaging

Intentionally **not versioned**:

- generated runs under `outputs/`
- handoff bundles under `handoff/`
- local QGIS exports
- private image corpora

## Project layout

```text
Dataselector/
├── dataselector/                 # canonical Python package
│   ├── data/                     # metadata, CRS, tile policy
│   ├── pipeline/                 # cache, experiment, run helpers
│   ├── runtime/                  # run metadata / error reporting
│   ├── selection/                # selection logic
│   └── workflows/                # thesis-orchestrate / thesis-pipeline / handoff
├── config/                       # active runtime and policy config
├── docs/                         # active methodology and operations docs
├── scripts/                      # thin wrappers / operational helpers
├── tests/                        # governance, unit, integration, and pipeline tests
└── outputs/                      # generated run artifacts (not versioned)
```

## Key documents

- [Thesis Pipeline How-To](docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
- [Config Policy](docs/CONFIG_POLICY.md)
- [Parameter Policy Ledger](docs/PARAMETER_POLICY_LEDGER.md)
- [Methodology](docs/METHODOLOGY.md)
- [Thesis Method Contract](docs/THESIS_METHOD_CONTRACT.md)
- [Min-Distance Evidence Addendum](docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md)
- [N-Samples Evidence Addendum](docs/N_SAMPLES_EVIDENCE_ADDENDUM.md)
- [Test Suite Curation](docs/TEST_SUITE_CURATION.md)
- [Repository Surface Curation](docs/REPO_SURFACE_CURATION.md)

## Advanced / historical workflows

`thesis-sampler-suite`, `xxl`, `xxl-monitor`, and selected monitor/recovery
helpers remain in the repository for supplementary or advanced operational
work. They are **not** the canonical thesis release path anymore; in the
README-level surface they are secondary-active to historical and should not
dominate the default entry story.

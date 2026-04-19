# Dataselector

**Algorithmische Datenselektion fuer die Karte des Deutschen Reiches (KDR100).**

Dataselector friert einen nachvollziehbaren, reproduzierbaren Auswahlvertrag fuer
die Annotation historischer Karten ein. Das Repo ist auf den
Masterarbeits-Workflow zugeschnitten: Selektion, Provenance, Freeze,
Validierung und optionales Handoff-Paket fuer die nachgelagerte
Trainingspipeline.

## Wofuer dieses Repo verantwortlich ist

Dataselector besitzt den **Selection Contract**:

- Kandidatenpool und Policy-gebundene Tile-Exclusions
- parameteraufgeloeste, validierte Thesis-Selektion
- Freeze-Artefakte (`selection_core.csv`, `selection_final_with_cases.csv`,
  `selection_contract.json`)
- Reporting, Provenance, CRS-Audit
- optionales post-freeze Packaging fuer Tile-/Patch-Handoffs

Das nachgelagerte Trainings-Repo besitzt den **Evaluation Contract**:

- train/val/test-Strategie fuer Modelltraining
- eigentliche Segmentierungsmodelle
- Cross-Validation und finale Modellvergleiche

## Methodischer Rahmen (Thesis Freeze)

1. Die Selektion ist architektur-neutral / model-agnostic und baut auf einem
   Diversity/Coverage-Proxy auf.
2. Der aktuelle Freeze ist ein `frozen dataset`; Modellvergleiche erfolgen
   nachgelagert.
3. No direct model-metric optimization (SegFormer/MapSAM/UNet++).
4. `alpha_visual` ist ein optimierter Parameter, aber keine harte
   Dominanzbedingung.
5. Visual-biased oder model-aware Selektion ist ein separater Ablationspfad
   mit neuem Freeze.

## Kanonischer Workflow

Die kanonische Runtime ist immer:

```bash
micromamba run -n dataselector python -m dataselector <command>
```

Kanonischer Produktionspfad:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>
```

Direkter Snapshot-Pfad:

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml
```

Wichtige operative Defaults:

- aktive Config: `config/pipeline_config.yaml`
- kanonischer Run-Root: `outputs/runs/`
- Thesis-Default: `selection_authority = snapshot_primary`
- optionales Phase-5-Packaging ist standardmaessig **aus**

Optionales Phase 5 Packaging:

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id> \
  --build-handoffs \
  --patches-per-tile 2 \
  --patch-include-case false
```

Phase 5 ist **post-freeze operational packaging**, nicht reselection.

## Schnellstart

```bash
make env-create
micromamba run -n dataselector python -m pip install -e .
micromamba run -n dataselector python -m pip install -r requirements-cpu.txt
```

Schnelle Governance-Gates:

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py
```

Langer kanonischer Pipeline-Gate:

```bash
micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py
```

## Repo-Status und Qualitaetsbild

Das Repo ist auf einen reproduzierbaren Showcase-Stand ausgelegt:

- wissenschaftlicher Freeze ueber `selection_*` und `selection_contract.json`
- explizite CRS-Provenance im `thesis_repro`-Pfad
- duenne Wrapper, wissenschaftliche Kernlogik im Paket
- Doc-/Policy-Governance per Pytest abgesichert
- optionale Handoffs fuer Tiles und Patches

Bewusst **nicht** versioniert:

- generierte Runs unter `outputs/`
- Handoff-Bundles unter `handoff/`
- lokale QGIS-Exporte
- private Bilddaten

## Projektstruktur

```text
Dataselector/
├── dataselector/                 # kanonische Python-Logik
│   ├── data/                     # Metadaten, CRS, Tile-Policy
│   ├── pipeline/                 # Cache, Experimente, Run-Utilities
│   ├── runtime/                  # Run-Metadata / Error Reporting
│   ├── selection/                # Selektionslogik
│   └── workflows/                # thesis-orchestrate / thesis-pipeline / handoff
├── config/                       # aktive Policies und Runtime-Config
├── docs/                         # aktive Methodik- und Betriebsdoku
├── scripts/                      # duenne Wrapper / operative Helfer
├── tests/                        # Governance-, Unit-, Integrations- und Pipeline-Tests
└── outputs/                      # generierte Run-Artefakte (nicht versioniert)
```

## Wichtige Dokumente

- [Thesis Pipeline How-To](docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md)
- [Config Policy](docs/08_GOVERNANCE/CONFIG_POLICY.md)
- [Parameter Policy Ledger](docs/08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md)
- [Methodology](docs/08_GOVERNANCE/METHODOLOGY.md)
- [Thesis Method Contract](docs/08_GOVERNANCE/THESIS_METHOD_CONTRACT.md)
- [Min-Distance Evidence Addendum](docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md)
- [N-Samples Evidence Addendum](docs/N_SAMPLES_EVIDENCE_ADDENDUM.md)
- [Test Suite Curation](docs/08_GOVERNANCE/TEST_SUITE_CURATION.md)
- [Repository Surface Curation](docs/08_GOVERNANCE/REPO_SURFACE_CURATION.md)

## Advanced / historical workflows

`thesis-sampler-suite`, `generate-monitor` und ausgewaehlte Analyse-/Export-
Helfer bleiben fuer supplementary oder advanced work im Repo. Die fruehere
`xxl`-/`xxl-monitor`-/Resume-Welt ist archiviert und **kein** aktiver Teil des
Release-Surface mehr.

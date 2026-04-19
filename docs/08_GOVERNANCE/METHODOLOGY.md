# Methodology (Current Thesis Path)

This document summarizes the active methodology for thesis runs.

## Canonical Execution

1. Runtime: `micromamba run -n dataselector <command>`
2. Production path: `python -m dataselector thesis-pipeline`
3. Output root: `outputs/runs/<timestamped_run>/`

## Responsibility Boundary

1. Dataselector owns the selection contract, the frozen dataset artifacts, and
   the associated provenance/reporting boundary.
2. The downstream training repository owns the authoritative train/val/test
   split strategy, model training, and final segmentation evaluation.
3. Optional integrated Phase 5 packaging is post-freeze operational output
   only; it must not mutate the scientific freeze artifacts.
4. Once annotation uses the Phase 5 patches, the annotated Phase 5 patch
   dataset is frozen: `selected_patches.csv`, `patch_id`, patch bounds /
   quicklook extents, patch-mask assignment, and the upstream
   `patch_split_manifest.json`.
5. Downstream `split_authority = masterarbeit_strassenerkennung_cv` means the
   training repository may manage the materialized training contract it
   actually runs, but for primary thesis results it must not silently replace
   the frozen upstream patch split regime except as fallback or sensitivity
   analysis.

## Scientific Resolution Model

Critical parameters are resolved centrally at pipeline start and must be:

1. `computed` with provenance, or
2. explicit `policy`/`manual` values with rationale.

Resolver flags:

1. `--compute-params`
2. `--use-params <snapshot.yaml>`
3. `--snapshot-config`
4. `--no-auto-continue`
5. `--force` (auditable exception path)

## Sampler and Snapshot Contracts

1. Sampler resolution order:
   - config policy
   - artifact
   - controlled determination run
2. No implicit hardcoded sampler fallback in production path.
3. Snapshot validation is mandatory; mismatch fails unless `--force`.
4. Snapshot hashes:
   - `parameters_hash`
   - `snapshot_content_sha256`

## Validation and Evidence

1. Validation stage runs as part of `thesis-pipeline`.
2. Inferenzielle Unsicherheitsquantifizierung läuft standardmäßig über
   `bootstrap_candidates` (nicht über nominale Seed-Replays).
3. Run metadata captures:
   - resolved sampler + source
   - snapshot path + hashes
   - resolved parameter provenance
   - CRS provenance status + audit path
4. Parameter classes and evidence references are tracked in:
   - `../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md`

## CRS Provenance (Thesis-Repro)

1. `thesis_repro` requires explicit source CRS provenance from sidecars/raster
   metadata; heuristic CRS inference is not accepted as final runtime evidence.
2. Heuristic CRS inference remains a technical fallback only outside strict
   thesis execution and for diagnostic audit generation.
3. Thesis runs must emit `data_quality/crs_provenance_audit.csv`.
4. If explicit CRS is missing or inconsistent with the observed coordinate
   regime, the strict thesis run fails before scientific optimization starts.

## Core+Case Contract

1. Primäre Thesis-Metriken basieren auf `selection_core.csv`.
2. Fallbeispiele (z. B. Hamburg) werden getrennt in `selection_case.csv`
   dokumentiert.
3. Operative Gesamtmenge wird in `selection_final_with_cases.csv` geführt.
4. Der Selektionsvertrag wird in `selection_contract.json` abgelegt.
5. Aktueller Policy-Default ist `selection.case_tile_names: ["Hamburg"]`
   bei `selection.case_exclude_from_core: true`.
6. Hamburg seeded-vs-unseeded Vergleiche sind nur supplementäre Evidenz und
   ersetzen keine Core-only Primärmetriken.
7. Dataset authority: `selection_core.csv`, `selection_final_with_cases.csv`,
   und `selection_contract.json` definieren den eingefrorenen Thesis-Datensatz.
8. Parameter authority: validierter Snapshot und Parameter-Resolution-Artefakte
   definieren den aufgelösten Parameterkontext.
9. Thesis-Default ist `selection.selection_authority: snapshot_primary`:
   Die Core-Selektion wird direkt aus Snapshot-Parametern materialisiert.
10. Legacy-Kompatibilität bleibt über
    `selection.selection_authority: materialized_csv_primary` erhalten.
11. `selection.objective_authority: unified_normalized` erzwingt dieselbe
    Objective-Definition für Exploration und Autoscale.
12. `selection_source` und `selection_source_file` dokumentieren im
    `selection_contract.json`, aus welcher Quelle der eingefrorene Datensatz
    materialisiert wurde.
13. Für kanonische v2-Runs ist `Selection Reconciliation = aligned` der
    erwartete Standard; `documented_difference` ist ein Legacy-Ausnahmefall.
14. Die Freeze-Selektion ist architektur-neutral / model-agnostic.
15. Der Freeze ist ein `frozen dataset`; der Modellvergleich folgt
    nachgelagert.
16. No direct model-metric optimization (SegFormer/MapSAM/UNet++).
17. `alpha_visual` ist ein optimierter Parameter, aber keine harte
    Dominanzbedingung.
18. Visual-biased oder model-aware Selektion ist ein separater
    Ablationspfad mit neuem Freeze.

## Related Authoritative Documents

1. `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
2. `docs/CONFIG_POLICY.md`
3. `../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md`
4. `../../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md`

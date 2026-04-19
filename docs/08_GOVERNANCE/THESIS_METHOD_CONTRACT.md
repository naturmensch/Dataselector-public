# Thesis Method Contract (Core+Case, Bootstrap UQ)

Dieses Dokument definiert den aktiven Methodik-Vertrag für thesis-relevante
Runs im Dataselector.

## 1. Sampling Design: Core+Case

1. Primäre wissenschaftliche Aussagen basieren auf der **Core-Selektion**.
2. Case-Tiles (z. B. Hamburg) sind **separat** dokumentiert und werden erst
   nach der Core-Selektion angehängt.
3. Bei `selection.case_exclude_from_core=true` dürfen Case-Tiles nicht zur
   Core-Selektion beitragen.
4. Aktueller Default in `config/pipeline_config.yaml` ist
   `selection.case_tile_names: ["Hamburg"]` (als Case, nicht als Core-Anker).
5. Hamburg-Effekte aus seeded-vs-unseeded Vergleichen gelten nur als
   ergänzende Diagnostik, nicht als Primärquelle für Thesis-Hauptclaims.
6. Dataset authority: `selection_core.csv`, `selection_final_with_cases.csv`,
   und `selection_contract.json` definieren den eingefrorenen Thesis-Datensatz.
7. Parameter authority: validierter Snapshot und Parameter-Resolution-Artefakte
   definieren den aufgelösten Parameterkontext.
8. Thesis-Default: `selection.selection_authority = snapshot_primary`.
9. Legacy-Kompatibilität: `selection.selection_authority = materialized_csv_primary`.
10. Objective-Default: `selection.objective_authority = unified_normalized`.
11. `selection_source` und `selection_source_file` in `selection_contract.json`
    dokumentieren die materialisierte Auswahlquelle des eingefrorenen Datensatzes.
12. Freeze-Selektion ist architektur-neutral / model-agnostic.
13. Freeze-Zielfunktion bleibt selektionsintern; Modellvergleich folgt
    nachgelagert auf dem `frozen dataset`.
14. No direct model-metric optimization (SegFormer/MapSAM/UNet++).
15. `alpha_visual` ist optimiert, aber keine harte Dominanzbedingung.
16. Visual-biased oder model-aware Selektion ist ein separater
    Ablationspfad mit neuem Freeze.

Verpflichtende Run-Artefakte:

1. `selection_core.csv`
2. `selection_case.csv`
3. `selection_final_with_cases.csv`
4. `selection_contract.json`

## 1a. Operational Packaging Boundary

1. Optional integrated Phase 5 packaging (`--build-handoffs`) is a
   post-freeze operational step only.
2. Once annotation uses the Phase 5 patches, the annotated Phase 5 patch
   dataset is frozen.
3. This frozen patch contract includes `selected_patches.csv`, `patch_id`,
   patch bounds / quicklook extents, patch-mask assignment, and the upstream
   `patch_split_manifest.json`.
4. Default integrated patch scope is `core-only`
   (`--patch-include-case false`).
5. Annotation-plan and handoff artifacts may be written after the freeze, but
   they must not mutate snapshot files, `selection_*`,
   `selection_contract.json`, or the annotated Phase 5 patch dataset.
6. Downstream training remains authoritative for evaluation splits
   (`split_authority = masterarbeit_strassenerkennung_cv`).
7. Downstream `split_authority` means the training repo may materialize and
   manage the materialized training contract it actually runs, but it must not
   redefine the annotated Phase 5 patch dataset.
8. For primary thesis results, the frozen upstream patch split regime remains
   the methodological reference. Alternative downstream-local split generation
   is a technical fallback or sensitivity analysis only, not a silent
   replacement of the main protocol.

## 2. Unsicherheitsquantifizierung (UQ)

1. Inferenzielle UQ erfolgt über
   `validation.replicate_mode=bootstrap_candidates`.
2. `seed_replay` ist ein Determinismus-/Replay-Check, kein inferenzieller
   Ersatz für unabhängige Replikation.
3. Validation-Outputs müssen den aktiven Modus explizit ausweisen.

Verpflichtende Validation-Artefakte:

1. `validation/validation_method_contract.md`
2. `validation/validation_results_bootstrap.csv` (bei Bootstrap-Modus)
3. `validation/validation_summary_stats.csv`

## 3. Tile-Exclusion und temporaler Scope

1. Tile-Exclusions sind policy-gesteuert via
   `config/tile_exclusion_policy.yaml`.
2. Temporale Scope-Entscheidungen müssen auditierbar sein.
3. Exclusion-Provenance wird in Run-Metadata dokumentiert
   (`tile_exclusions_*`, `tile_exclusion_policy_sha256`).
4. `KDR_155b` bleibt als räumliche Duplicate-Standort-Repräsentation aus dem
   Kandidatenpool ausgeschlossen.
5. Tiles außerhalb des über die Policy-Konstante
   `kdr_core_publication_frame = 1878-1945` definierten KDR-Kernzeitraums
   bleiben im Kandidatenpool, müssen aber als retained temporal outliers in
   Metadata/Reports kritisch markiert werden (`tile_flagged_*`), weil ihre
   Jahreslage außerhalb des Thesis-Kernfensters liegt. Im aktuellen Datensatz
   betrifft das unter anderem `KDR_039` (1980) und `KDR_521` (1985).

Erwartetes Audit-Artefakt:

1. `data_quality/year_scope_audit.csv`
2. `data_quality/crs_provenance_audit.csv`

## 3a. CRS-Provenance und Distanzraum

1. `thesis_repro` verlangt explizite Source-CRS-Provenance aus
   `*.aux.xml`-Sidecars oder aus der Rasterdatei selbst.
2. Heuristische CRS-Erkennung ist nur ein technischer Fallback außerhalb
   strikter Thesis-Ausführung und darf nicht als kanonische Evidenz eines
   Thesis-Runs enden.
3. Das Run-Metadata muss `metadata_crs` plus
   `crs_provenance_audit_path` ausweisen.
4. Wenn explizite CRS fehlt oder die explizite CRS nicht zum beobachteten
   Koordinatenregime passt, ist der Thesis-Run ein Hard-Fail.

## 4. Reporting Contract

1. `THESIS_PIPELINE_REPORT.md` trennt Core und Case explizit.
2. Primärmetriken sind als **Core-only** markiert.
3. Methodische Claims verweisen auf konkrete Run-Artefakte.
4. Reports müssen Dataset authority und Parameter authority explizit benennen.
5. Für kanonische Thesis-v2-Runs wird `Selection Reconciliation = aligned`
   erwartet; `documented_difference` ist nur für explizite Legacy-Runs zulässig.

Erwartete Zusatzartefakte:

1. `THESIS_METHOD_AUDIT.md`
2. `THESIS_KEY_CLAIMS.csv`

## 5. Autoritative Quellen

1. `../../08_GOVERNANCE/METHODOLOGY.md`
2. `docs/CONFIG_POLICY.md`
3. `../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md`
4. `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

# Thesis Phase-5 Run Protocol (Canonical)

Last updated: 2026-04-25

Former location: `docs/09_INTERNAL/status/width_calibration_run_chain_2026-04-19.md`.


## 0) Ziel dieses Dokuments

Dieses Protokoll dokumentiert den aktuellen Thesis-Pfad als durchgaengige,
wissenschaftlich nachvollziehbare Prozesskette.

Lueckenlos bedeutet hier:

1. Jeder Schritt beschreibt, was gemacht wurde.
2. Jeder Schritt beschreibt, wie er ausgefuehrt wurde.
3. Jeder Schritt beschreibt, warum er methodisch noetig ist.
4. Jeder Schritt beschreibt, woran Erfolg erkannt wird.
5. Jeder Schritt nennt Belegartefakte und ordnet sie inhaltlich ein.

Wichtig:

1. Es werden nur erfolgreiche, aufeinander aufbauende Schritte dokumentiert.
2. Fehlgeschlagene Nebenlaeufe sind nicht Teil des Hauptprotokolls.
3. QGIS/QField-Strassenannotation und Width-Messung sind getrennte Prozesse.

## 1) Aktueller Stand (2026-04-25)

1. Dataselector-Freeze ist abgeschlossen.
2. Patch-Handoff-Vertrag ist abgeschlossen.
3. Strassenannotation auf Patch-Basis ist die Ground-Truth-Grundlage.
4. Width Calibration ist abgeschlossen.
5. Finale width-basierte Patch-Masken sind erzeugt.
6. Finale Downstream-Integration fuer 54 Patches ist materialisiert.
7. Readiness-Gate ist bestanden.
8. Die serverseitige Pilotphase der Phase-5-Kampagne ist abgeschlossen.
9. Fixe Hauptbudgets sind aus der Pilotphase abgeleitet: `unetpp=200`, `segformer=200`, `mapsam=150`.
10. `results/pilot_budget_summary.csv` und `results/pilot_budget_decisions.md` liegen im Trainingsrepo vor.
11. Bei knappem Serverplatz duerfen Pilot-Roots nach dem Budgetfreeze auf `checkpoint_best.pt` sowie `checkpoint_epoch_{050,100,150,200}.pt` ausgeduennt werden.
12. Die 9-Run-Hauptkampagne auf Folds `1/2/4` ist angelaufen; vier valide Hauptlaeufe liegen bereits vor (`unetpp full_1024_no_aug`, `unetpp crop_512_no_aug`, `unetpp crop_512_aug`, `segformer full_1024_no_aug`).
13. Fuer die restliche Hauptkampagne ist jetzt ein archivierungsgekoppelter Orchestrator im Trainingsrepo der operative Standard: validieren, nach OneDrive archivieren, per `rclone check` verifizieren, erst dann lokal loeschen.
14. Restore archivierter Haupt-Roots ist Teil des methodischen Betriebsmodells, weil spaetere `final`-, `iou_best`- und `apls_best`-Re-Evaluationen echte Checkpoints benoetigen.
15. Der zusaetzliche klassische Morphologie-Track wird getrennt als
    pilot-kalibrierte deterministische Bildverarbeitungs-Baseline gefuehrt:
    Compact-Sweep nur auf Folds `0/3`, eingefrorene Main-Evaluation auf
    Folds `1/2/4`.

Statusmatrix:

| Prozessschritt | Status | Primaerer Beleg | Naechster Handlungsbedarf |
| --- | --- | --- | --- |
| Freeze | abgeschlossen | `selection_contract.json` | keiner im Dataselector |
| Patch-Handoff | abgeschlossen | `patch_handoff_manifest.json` | keiner im Dataselector |
| Annotation | abgeschlossen als Ground-Truth-Grundlage | Patch-Handoff + Maskenanforderungen | keine Handoff-Umdefinition |
| Width Calibration | abgeschlossen | `width_calibration_summary.csv` | keine neue Messrunde |
| Finale Maskenerzeugung | abgeschlossen | `width_calibration_final_mask_manifest.json` | keinen Debug-Maskensatz verwenden |
| Downstream Materialize | final abgeschlossen | `phase5_dataset_manifest.json` | produktiven Vertrag verwenden |
| Readiness | bestanden | `phase5_samples.csv` + Split-/Dataset-Manifest | bei Trainingsstart kurz erneut pruefen |
| Pilotkampagne | abgeschlossen | serverseitige `cv_summary_*` + `phase5_threshold_sweep_*` Artefakte | keine neue Pilotmatrix starten |
| Budgetfreeze | abgeschlossen | `pilot_budget_summary.csv` + `pilot_budget_decisions.md` im Trainingsrepo | Budgets unveraendert in die Hauptkampagne uebernehmen |
| Hauptkampagne | angelaufen / aktiv | Hauptlauf-Registry, neue `cv_summary_*`-Artefakte und archivierte Haupt-Roots | restliche Hauptruns unter Archiv-Orchestrierung abschliessen |
| Klassischer Morphologie-Track | offen / separater Vergleichstrack | `classical_pipeline_selection_*`, `cv_metrics_morphology_*` im Trainingsrepo | Compact-Pilotkalibrierung auf `0/3`, danach frozen Main-Evaluation auf `1/2/4` |

## 2) Methodischer Vertrag (Warum die Kette so aufgebaut ist)

Methodikquellen:

1. [docs/08_GOVERNANCE/THESIS_METHOD_CONTRACT.md](../../08_GOVERNANCE/THESIS_METHOD_CONTRACT.md)
2. [docs/08_GOVERNANCE/METHODOLOGY.md](../../08_GOVERNANCE/METHODOLOGY.md)
3. [docs/08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md](../../08_GOVERNANCE/PARAMETER_POLICY_LEDGER.md)

Methodische Kernaussagen:

1. Die Selektion wird als gefrorener Datensatzvertrag festgelegt
   (dataset authority).
2. Annotation und Width sind nachgelagerte Veredelungsschritte auf genau diesem
   Freeze-Datensatz.
3. Downstream darf Trainingsartefakte materialisieren, aber nicht stillschweigend
   den eingefrorenen Patch-Datensatz umdefinieren.
4. Der Freeze ist model-agnostisch und kein direktes Optimieren auf
   SegFormer/MapSAM/UNet++-Metriken.

Warum das wissenschaftlich wichtig ist:

1. Reproduzierbarkeit: derselbe eingefrorene Datensatz ist spaeter erneut
   rekonstruierbar.
2. Trennung von Verantwortungen: Dataselector selektiert, Training-Repo
   evaluiert.
3. Vermeidung von Bias: kein stilles Nachjustieren der Stichprobe waehrend oder
   nach Training.

## 3) Schritt A: Upstream-Selektion der Tiles (Freeze)

### A.1 Was wurde gemacht?

1. Ein autoritativer Thesis-Freeze-Run wurde ausgefuehrt.
2. Ergebnis ist eine Core+Case-Selektion als feste Datengrundlage.

### A.2 Wie wurde es gemacht?

1. Dataselector erstellt eine Kandidatenbasis und filtert sie nach aktiver
   Exclusion-Policy.
2. Die eigentliche Auswahl erfolgt als gewichtete Multi-Kriterien-Selektion
   (visuell, raeumlich, zeitlich) mit harter Distanzpolicy.
3. Core und Case werden getrennt gefuehrt, dann als finaler Datensatzvertrag
   dokumentiert.

Belegartefakte:

1. [outputs/runs/thesis_orchestrate_20260313T200624Z](../../../outputs/runs/thesis_orchestrate_20260313T200624Z)
2. [outputs/runs/thesis_orchestrate_20260313T200624Z/selection_contract.json](../../../outputs/runs/thesis_orchestrate_20260313T200624Z/selection_contract.json)
3. [outputs/runs/thesis_orchestrate_20260313T200624Z/selection_core.csv](../../../outputs/runs/thesis_orchestrate_20260313T200624Z/selection_core.csv)
4. [outputs/runs/thesis_orchestrate_20260313T200624Z/selection_case.csv](../../../outputs/runs/thesis_orchestrate_20260313T200624Z/selection_case.csv)
5. [outputs/runs/thesis_orchestrate_20260313T200624Z/selection_final_with_cases.csv](../../../outputs/runs/thesis_orchestrate_20260313T200624Z/selection_final_with_cases.csv)

### A.3 Warum wurde es so gemacht?

1. Nicht alle Karten zu annotieren reduziert Aufwand, ohne den Anspruch auf
   methodische Diversitaet aufzugeben.
2. Das Core+Case-Design erlaubt eine stabile Primarstichprobe plus separaten
   Fallkontext (Hamburg).
3. Der eingefrorene Vertrag sichert, dass spaetere Schritte denselben
   Datensatz nutzen.

### A.4 Woran wird Erfolg erkannt?

1. Core, Case und Final-Dateien liegen konsistent vor.
2. `selection_contract.json` dokumentiert die materialisierte Quelle und den
   Freeze-Kontext.
3. Die Freeze-Metriken sind nachvollziehbar dokumentiert.

Dokumentierter Freeze-Stand:

1. Core selection: 27 Tiles.
2. Case selection: 1 Tile (Hamburg).
3. Gesamt: 28 Tiles.

## 4) Schritt B: Von Freeze-Tiles zu annotierbaren Patches

### B.1 Was wurde gemacht?

1. Aus der Freeze-Auswahl wurde ein patch-basierter Handoff-Vertrag erzeugt.
2. Dieser Vertrag definiert die annotierbaren Arbeitseinheiten fuer den
   Phase-5-Datensatz.

### B.2 Wie wurde es gemacht?

1. Dataselector verpackt patch-level Artefakte in einem standardisierten
   Handoff-Format.
2. Dabei werden Patch-Metadaten, Quicklooks, Split-Provenance und
   Maskenanforderungen dokumentiert.
3. Das Ergebnis wird im Patch-Handoff-Manifest und begleitenden CSV/JSON
   Artefakten festgehalten.

Belegartefakte:

1. [handoff/thesis_orchestrate_20260313T200624Z_patches_core/patch_handoff_manifest.json](../../../handoff/thesis_orchestrate_20260313T200624Z_patches_core/patch_handoff_manifest.json)

Relevante Vertragswerte:

1. `selection_id`: thesis_orchestrate_20260313T200624Z_1e251e4af7c3
2. `patch_selection_count`: 54
3. `split_authority`: masterarbeit_strassenerkennung_cv
4. `patch_split_manifest_sha256`: e6abd4d7cc96289a8cb47126bc220d798e9e281af98324d8f584d3ed56d26cb0

### B.3 Warum wurde es so gemacht?

1. Patch-Einheiten sind die operative Annotationseinheit.
2. Der Handoff erzwingt ein maschinenpruefbares Schema statt informeller
   Dateisammlungen.
3. Durch den Vertrag bleiben Auswahl und spaetere Trainingsnutzung sauber
   gekoppelt.

### B.4 Woran wird Erfolg erkannt?

1. Pflichtdateien sind vorhanden und schema-konsistent.
2. Patch-Anzahl und Split-Provenance sind mit dem Freeze konsistent.
3. Quicklook- und Georeferenzanforderungen sind erfuellt.

## 5) Schritt C: Strassenannotation in QGIS/QField (Ground Truth)

### C.1 Was wurde gemacht?

1. Die Patch-Einheiten wurden in QGIS/QField als manuelle
   Strassen-Ground-Truth annotiert.
2. Ergebnis sind Patch-Masken, die spaeter fuer Training materialisiert werden.

### C.2 Wie wurde es gemacht?

1. Annotatoren arbeiten auf den patch-basierten Quicklooks.
2. Strassen werden manuell als Ground-Truth in den vorgesehenen
   Patch-Arbeitseinheiten digitalisiert.
3. Der resultierende Maskenbestand wird in der durch den Handoff geforderten
   Struktur bereitgestellt.

Operative Anschlussstellen:

1. Handoff verlangt `patch_mask_requirements.csv`-konforme Maskenbereitstellung.
2. Server-Verify prueft Existenz und Konsistenz dieser Masken gegen die
   Patch-ID-Liste.

### C.3 Warum wurde es so gemacht?

1. Historische Karten enthalten komplexe Symbole; Ground Truth braucht
   fachlich kontrollierte manuelle Annotation.
2. Patch-basierte Annotation ermoeglicht kontrollierbare Qualitaet und
   reproduzierbare Zuordnung von Patch zu Maske.
3. Der Handoff-Vertrag verhindert, dass spaeter Masken und Patch-IDs
   auseinanderlaufen.

### C.4 Woran wird Erfolg erkannt?

1. Alle erforderlichen Patch-Masken sind fuer den Handoff vorhanden.
2. Verify kann Patch-IDs gegen Maskendateien konsistent pruefen.
3. Der annotierte Phase-5-Patchdatensatz bleibt als eingefrorener Vertrag
   stabil.

Wichtige methodische Abgrenzung:

1. Dieser Schritt erzeugt Ground-Truth-Masken.
2. Dieser Schritt ist nicht die Width-Messung.

## 6) Schritt D: Width Calibration (separater Messprozess)

### D.1 Run-Authority und Inputs

1. Fuer die Strassenklassen wurden Breitenmessungen erhoben.
2. Daraus wurden classwise Endbreiten (`final_width_px`) abgeleitet.
3. Der finale belegte Run ist
   `outputs/runs/width_calibration_20260418T195314Z`.
4. Fruehere oder archivierte Width-Runs sind nicht Teil dieser kanonischen
   Hauptkette.

Belegartefakte:

1. [outputs/runs/width_calibration_20260418T195314Z/width_calibration_manifest.json](../../../outputs/runs/width_calibration_20260418T195314Z/width_calibration_manifest.json)
2. [outputs/runs/width_calibration_20260418T195314Z/width_calibration_tasks.csv](../../../outputs/runs/width_calibration_20260418T195314Z/width_calibration_tasks.csv)
3. [outputs/runs/width_calibration_20260418T195314Z/width_calibration_measurements.csv](../../../outputs/runs/width_calibration_20260418T195314Z/width_calibration_measurements.csv)
4. [outputs/runs/width_calibration_20260418T195314Z/summary/width_calibration_summary.csv](../../../outputs/runs/width_calibration_20260418T195314Z/summary/width_calibration_summary.csv)
5. [outputs/runs/width_calibration_20260418T195314Z/sensitivity/width_calibration_sensitivity.csv](../../../outputs/runs/width_calibration_20260418T195314Z/sensitivity/width_calibration_sensitivity.csv)

Konkrete Manifestwerte des finalen Runs:

1. `workflow_version`: `phase5_width_calibration_v2`
2. Run-Manifest erzeugt: `2026-04-18T19:53:26Z`
3. Summary erzeugt: `2026-04-18T20:55:35Z`
4. `handoff_dir`:
   `<dataselector-repo>/handoff/thesis_orchestrate_20260313T200624Z_patches_core`
5. `roads_gpkg`:
   `<dataselector-repo>/handoff/local_sources/phase5_roads_merged.gpkg`
6. `roads_gpkg_sha256`:
   `41ccdc984261871d46dcd37984f280443ad5d5328606e1ec9457a872b447536d`
7. `seed`: `42`
8. `crop_size_px`: `128`
9. `candidate_count`: `24977`
10. `primary_task_count`: `1257`
11. `repeat_task_count`: `256`
12. `hamburg_excluded_at_task_generation`: `true`

Lokale Roads-Rekonstruktion fuer die finale Maskenerzeugung:

1. Der im finalen Width-Run dokumentierte Byte-SHA
   `41ccdc984261871d46dcd37984f280443ad5d5328606e1ec9457a872b447536d`
   war am 2026-04-19 lokal nicht mehr als Datei auffindbar.
2. Der repo-lokale `handoff/local_sources/phase5_roads_merged.gpkg` war durch
   spaetere Test-/Hilfslaeufe nicht mehr identisch mit dem Messstand.
3. Wiederherstellung erfolgte aus dem Snapshot
   `handoff/local_sources/snapshots/20260418T192057Z`.
4. Der neu gebaute lokale Roads-Stand hat den SHA
   `cde29bc0631fb13489f38f4a25405bb13a622c3d26b4a441e89b9e305f860b44`.
5. Validierung gegen den finalen Width-Run:
   Re-Prepare mit Seed `42`, `crop_size_px=128` und proportionaler Policy
   erzeugte exakt denselben `width_calibration_tasks.csv`-SHA
   `56db81d8fd7dbe6b5fdac9efae1b0952544ea9aeb538d0039d020cd0f70691e8`.
6. Damit ist die finale Maskenerzeugung ueber identische Task-Generierung
   fachlich an den finalen Width-Run gekoppelt; byte-identische
   `roads_gpkg_sha256=41cc...` ist nicht mehr verfuegbar.

### D.2 Messdesign und Qualitaetslogik

Wie:

1. Zuerst wird eine deterministische Aufgabenliste vorbereitet.
2. Die Messung erfolgt interaktiv per Zwei-Klick-Bedienung auf einem
   vorbereiteten Crop.
3. Ungueltige Messstellen werden mit kodierten Reject-Gruenden erfasst.
4. Danach werden pro Klasse robuste Kennzahlen (Median, Streuung,
   Reliabilitaetsflags) zusammengefasst.

Warum:

1. Training benoetigt konsistente, klassenabhaengige Breiten fuer
   maskenbildende Schritte.
2. Die getrennte Messung verhindert, dass Width-Annahmen implizit oder
   unkontrolliert gesetzt werden.
3. Deterministische Queue + kodierte Rejects machen den Prozess auditierbar.

Referenzmethodik und Bedienlogik:

1. [docs/03_USER_GUIDES/PHASE5_WIDTH_CALIBRATION_METHOD.md](../../03_USER_GUIDES/PHASE5_WIDTH_CALIBRATION_METHOD.md)

Dokumentierte Reject-Codes:

1. `crossing`
2. `label_overlap`
3. `endpoint`
4. `tight_curve`
5. `blur_damage`
6. `ambiguous_symbol`
7. `crop_too_small`
8. `click_error`
9. `other`

### D.3 Ergebnisse

Erfolgskriterien:

1. Mess- und Summary-Artefakte sind vollstaendig vorhanden.
2. Pro Klasse sind finale Breiten und Qualitaetsflags dokumentiert.
3. Der einzelne verworfene Messpunkt ist explizit und nachvollziehbar belegt.

Dokumentierte Runzahlen:

1. Messzeilen: 1513
2. Akzeptiert: 1512
3. Verworfen: 1

Dokumentierte Ergebnisbreiten:

| Klasse | valide Primaermessungen | Median px | `final_width_px` | Flag-Hinweis |
| --- | ---: | ---: | ---: | --- |
| 0 | 38 | 14.964471 | 15 | high variance |
| 1 | 246 | 6.890223 | 7 | high variance |
| 2 | 11 | 5.052880 | 5 | low reliability |
| 3 | 3 | 5.942019 | 6 | low evidence |
| 4 | 20 | 9.674027 | 10 | high variance |
| 5 | 817 | 4.061600 | 4 | none |
| 6 | 112 | 3.711605 | 4 | none |
| 8 | 3 | 6.334313 | 6 | low evidence |
| 9 | 6 | 9.536708 | 10 | high variance |

### D.4 Reject-Fall

Beleg fuer den verworfenen Messpunkt:

1. [outputs/runs/width_calibration_20260418T195314Z/width_calibration_tasks.csv](../../../outputs/runs/width_calibration_20260418T195314Z/width_calibration_tasks.csv)
2. [outputs/runs/width_calibration_20260418T195314Z/width_calibration_measurements.csv](../../../outputs/runs/width_calibration_20260418T195314Z/width_calibration_measurements.csv)

Inhaltliche Einordnung des Reject-Falls:

1. `task_id`: `task_00494`
2. Klasse: `5`
3. Patch: `KDR_082_p2`
4. Source Feature: `row_004720`
5. Candidate Anchor: `anchor04`
6. Messung: `measure_01513`
7. `keep`: `0`
8. `reject_reason`: `other`

## 7) Schritt E: Finale Maskenerzeugung aus Width Calibration

### E.1 Was wurde gemacht?

1. Die pre-finale Debug-/Smoke-Maskenkette mit `fixed_width_px=10` wurde nicht
   als finaler Thesis-Maskensatz uebernommen.
2. Stattdessen wurden 54 finale GeoTIFF-Patchmasken mit den classwise
   `final_width_px` aus der Width-Summary erzeugt.
3. Der finale Maskensatz liegt getrennt vom alten Debug-Pfad im Trainingsrepo.

Pfad-Authority fuer diesen Stand:

1. Dataselector-Quellhandoff:
   `<dataselector-repo>/handoff/thesis_orchestrate_20260313T200624Z_patches_core`
2. Downstream-importierter Handoff:
   `<training-repo>/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413`
3. Finale Summary:
   `<dataselector-repo>/outputs/runs/width_calibration_20260418T195314Z/summary/width_calibration_summary.csv`
4. Wiederhergestellte Roads-GPKG:
   `<dataselector-repo>/handoff/local_sources/phase5_roads_merged.gpkg`
5. Finaler Maskenpfad:
   `<training-repo>/data/patch_masks_final_width_calibration_20260418T195314Z`
6. Finaler Handoff-Contract:
   `<training-repo>/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413/phase5_final_width_contract.json`

### E.2 Wie wurde es gemacht?

Repo-lokalen Roads-Stand aus dem naechsten validen Snapshot wiederherstellen:

```bash
micromamba run -n dataselector python -m dataselector build-width-calibration-roads-source \
  --cut-roads-gpkg handoff/local_sources/snapshots/20260418T192057Z/cut_fixed_geometry_roads.gpkg \
  --tracer4-gpkg handoff/local_sources/snapshots/20260418T192057Z/4_fixed.gpkg \
  --tracer5-gpkg handoff/local_sources/snapshots/20260418T192057Z/5_fixed.gpkg \
  --cut-roads-layer cut_fixed_geometry_roads \
  --tracer4-layer 4_fixed \
  --tracer5-layer 5_fixed \
  --dest-layer phase5_roads_merged
```

Finale Masken rendern:

```bash
DS_ROOT=<dataselector-repo>
TRAIN_REPO=<training-repo>
HANDOFF_DIR="$TRAIN_REPO/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413"
ROADS_GPKG="$DS_ROOT/handoff/local_sources/phase5_roads_merged.gpkg"
SUMMARY_CSV="$DS_ROOT/outputs/runs/width_calibration_20260418T195314Z/summary/width_calibration_summary.csv"
FINAL_MASKS_DIR="$TRAIN_REPO/data/patch_masks_final_width_calibration_20260418T195314Z"

micromamba run -n dataselector python -m dataselector render-width-calibration-final-masks \
  --handoff-dir "$HANDOFF_DIR" \
  --roads-gpkg "$ROADS_GPKG" \
  --roads-layer phase5_roads_merged \
  --summary-csv "$SUMMARY_CSV" \
  --out-dir "$FINAL_MASKS_DIR" \
  --expected-roads-gpkg-sha256 cde29bc0631fb13489f38f4a25405bb13a622c3d26b4a441e89b9e305f860b44 \
  --expected-summary-csv-sha256 53848f6e845a9fad24180cb60baa7b04817a330897514a5769f42e8fe4c7774c
```

### E.3 Warum wurde es so gemacht?

1. Die Trainingsmasken muessen die gemessenen Klassenbreiten verwenden; ein
   fixer Debugwert waere fuer den finalen Thesis-Vertrag nicht valide.
2. Der separate Final-Pfad verhindert, dass alte `data/patch_masks/`-Artefakte
   stillschweigend als final umgedeutet werden.
3. Die SHA-Guards verhindern versehentliches Rendern mit falscher Summary oder
   falschem Roads-Stand.
4. Der finale Width-Zustand wird als Handoff-Erweiterung gefuehrt; Contract und
   Maskenmanifest bilden die pruefbare Grenze zwischen Dataselector-Kontext und
   Trainingsrepo.

### E.4 Woran wird Erfolg erkannt?

1. Genau 54 `*_mask.tif`-Dateien liegen im finalen Maskenordner.
2. Das Manifest ist vorhanden:
   `width_calibration_final_mask_manifest.json`.
3. `debug_only=false`, `test_only=false`.
4. `rendering_mode=final_width_px`.
5. Keine `fixed_width_px`-Authority im finalen Manifest.

Finale Maskenwerte:

| Klasse | `final_width_px` |
| --- | ---: |
| 0 | 15 |
| 1 | 7 |
| 2 | 5 |
| 3 | 6 |
| 4 | 10 |
| 5 | 4 |
| 6 | 4 |
| 8 | 6 |
| 9 | 10 |

Belegwerte:

1. `mask_count`: `54`
2. `summary_csv_sha256`:
   `53848f6e845a9fad24180cb60baa7b04817a330897514a5769f42e8fe4c7774c`
3. `roads_gpkg_sha256`:
   `cde29bc0631fb13489f38f4a25405bb13a622c3d26b4a441e89b9e305f860b44`
4. Semantische Kopplung an den finalen Width-Run:
   `width_calibration_tasks.csv` reproduziert mit SHA
   `56db81d8fd7dbe6b5fdac9efae1b0952544ea9aeb538d0039d020cd0f70691e8`.
5. `phase5_final_width_contract.json` fixiert den finalen Maskensatz am
   Handoff und trennt ihn explizit vom pre-finalen `data/patch_masks/`-Pfad.
6. Dieser Handoff-Contract ist upstream Authority; Downstream-Meta-SHAs werden
   erst nach `materialize-patches` im materialisierten Contract erzeugt.

## 8) Schritt F: Finaler Downstream Verify + Materialize

### F.1 Was wurde gemacht?

1. Der Downstream-Handoff wurde gegen den finalen Maskenpfad verifiziert.
2. Ein neuer materialisierter Trainingsvertrag wurde mit expliziter finaler
   `selection_id` erzeugt.
3. Der fruehere Materialize-Stand
   `thesis_orchestrate_20260313T200624Z_1e251e4af7c3` bleibt pre-final, weil
   er auf `data/patch_masks/` und damit auf dem Debug-/Smoke-Maskenpfad beruhte.

Finale `selection_id`:

```text
thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418
```

### F.2 Wie wurde es gemacht?

```bash
TRAIN_REPO=<training-repo>
HANDOFF_DIR="$TRAIN_REPO/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413"
MASKS_DIR="$TRAIN_REPO/data/patch_masks_final_width_calibration_20260418T195314Z"
INTEGRATION_ID=thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418
INTEGRATION_DIR="$TRAIN_REPO/data/integration/$INTEGRATION_ID"
```

Verify:

```bash
cd "$TRAIN_REPO"
bash scripts/setup/handoff_check.sh verify-server-patches \
  --handoff-dir "$HANDOFF_DIR" \
  --masks-dir "$MASKS_DIR"
```

Materialize:

```bash
cd "$TRAIN_REPO"
bash scripts/setup/handoff_check.sh materialize-patches \
  --handoff-dir "$HANDOFF_DIR" \
  --masks-dir "$MASKS_DIR" \
  --out-root data/integration \
  --split-policy use_handoff \
  --selection-id "$INTEGRATION_ID"
```

### F.3 Warum wurde es so gemacht?

1. Verify beantwortet die Frage, ob finaler Handoff und finaler Maskensatz
   technisch und fachlich konsistent sind.
2. Materialize erzeugt den lokalen Trainingsvertrag, den der Trainer direkt
   konsumiert.
3. Die neue `selection_id` trennt pre-finalen Debug-Maskenstand und finalen
   Width-Maskenstand auditierbar.
4. `use_handoff` erzwingt Upstream-Split-Provenance fuer methodische
   Vergleichbarkeit.

### F.4 Woran wird Erfolg erkannt?

1. Verify-Status: `ok`
2. `patch_count`: `54`
3. `handoff_split_status`: `complete`
4. Materialize-Status: `ok`
5. `split_source`: `handoff_imported`
6. `split_policy_requested`: `use_handoff`
7. `split_authority`: `masterarbeit_strassenerkennung_cv`
8. `phase5_dataset_manifest.json` zeigt auf
   `<training-repo>/data/patch_masks_final_width_calibration_20260418T195314Z`.
9. `phase5_final_width_contract.source.json` liegt als Upstream-Snapshot in
   `meta/`; `phase5_final_width_contract.json` ist dort der materialisierte
   Downstream-Contract mit den lokalen Meta-SHAs.

Belegartefakte:

1. [masterarbeit-strassenerkennung/data/patch_masks_final_width_calibration_20260418T195314Z/width_calibration_final_mask_manifest.json](../../../../masterarbeit-strassenerkennung/data/patch_masks_final_width_calibration_20260418T195314Z/width_calibration_final_mask_manifest.json)
2. [masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_dataset_manifest.json](../../../../masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_dataset_manifest.json)
3. [masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_samples.csv](../../../../masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_samples.csv)
4. [masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_local_split_manifest.json](../../../../masterarbeit-strassenerkennung/data/integration/thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418/meta/phase5_local_split_manifest.json)
5. [masterarbeit-strassenerkennung/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413/phase5_final_width_contract.json](../../../../masterarbeit-strassenerkennung/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413/phase5_final_width_contract.json)

Final materialisierte Vertragswerte:

1. `patch_count`: `54`
2. `phase5_samples.csv`: 54 Datenzeilen
3. Gruppen/Tiles: 27
4. Fold-Verteilung: Fold 0 = 12, Fold 1 = 12, Fold 2 = 10, Fold 3 = 10,
   Fold 4 = 10
5. `source_handoff_dir`:
   `<training-repo>/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413`
6. `masks_dir`:
   `<training-repo>/data/patch_masks_final_width_calibration_20260418T195314Z`
7. `phase5_final_width_contract.source.json` und
   `phase5_final_width_contract.json`: in der Integration unter `meta/`
   vorhanden.

## 8.5) Server-/UniCloud-Bereitstellung ueber Git und Git LFS

Operativer Standard fuer den Zielserver:

1. Git-Checkout liefert den Code sowie getrackte Handoff-Artefakte.
2. Git LFS liefert die getrackten Handoff-Quicklooks und die finalen
   width-basierten Patch-Masken.
3. `phase5_final_width_contract.json` im Handoff ist upstream Authority; der
   materialisierte Downstream-Contract wird erst in `meta/` erzeugt.
4. Der finale Maskenordner
   `data/patch_masks_final_width_calibration_20260418T195314Z/` ist ein
   Git-LFS-getrackter Bestandteil des Trainingsrepos und muss im Checkout
   verfuegbar sein.

Methodische Klarstellung:

1. Es wird kein konkretes Kopierwerkzeug vorgeschrieben.
2. Der akzeptierte Zielzustand wird nicht ueber das Transferwerkzeug, sondern
   ueber Validator, Manifest-SHAs und `verify-server-patches` nachgewiesen.
3. Das wissenschaftliche Authority-Modell bleibt unveraendert: Handoff,
   Contract, finale Masken und materialisierte Integration bleiben getrennte,
   pruefbare Artefakte.

## 8.6) Evidenz-Wiederherstellung und Loeschschutz (Run-Herleitung)

Ziel:

1. Die Herleitung der fuer den aktiven Thesis-Run materialisierten Parameter
   bleibt nachvollziehbar, auch wenn historische/ausgeblendete Bereiche spaeter
   bereinigt werden.
2. Die methodische Trennung bleibt explizit:
   - Primary Evidence: run-spezifische Authority-Artefakte
   - Secondary Evidence: historische Kontextartefakte

Primary Evidence (run-spezifisch, authority-nah):

1. `outputs/runs/thesis_orchestrate_20260313T200624Z/parameter_resolution/optuna_autoscale_summary_20260313.csv`
2. `outputs/runs/thesis_orchestrate_20260313T200624Z/parameter_resolution/optuna_autoscale_report_20260313.md`
3. `outputs/runs/thesis_orchestrate_20260313T200624Z/parameter_resolution/sampler_resolution/selected_sampler.json`
4. `outputs/runs/thesis_orchestrate_20260313T200624Z/final_config_20260313T201819Z.yaml`
5. `outputs/runs/thesis_orchestrate_20260313T200624Z/validation/validation_method_contract.md`
6. `outputs/runs/thesis_orchestrate_20260313T200624Z/validation/validation_summary_stats.csv`
7. `outputs/runs/thesis_orchestrate_20260313T200624Z/THESIS_PIPELINE_REPORT.md`
8. `outputs/runs/thesis_orchestrate_20260313T200624Z/THESIS_METHOD_AUDIT.md`

Secondary Evidence (historischer Kontext, nicht authority):

1. `archive_local/old_runs/20260116_T164624_hamburg_full_2000/results/bootstrap_final_selection_summary.csv`
2. `archive_local/old_runs/sampler_comparison_20260116_T215803/comparison_summary.csv`
3. `archive_local/old_runs/20260117_T205336_hamburg_xxl_final_5000trials/results/bootstrap_results_summary.csv`

Zielzonen fuer dauerhafte Ablage:

1. Aktive evidenzfuehrende Zone:
   `docs/06_REFERENCE/thesis_decision_evidence/`
2. Archivwellen mit Restore-Manifest:
   `docs/07_ARCHIVE/<wave>/MANIFEST.md`

Loeschschutz-Regel:

1. Keine Archivierung oder Entfernung von governance-/evidenzrelevanten Dateien
   ohne SHA256-basiertes `MANIFEST.md` mit Restore-Prozedur.
2. Quartalsweise Integrity-Pruefung aller Archive-Manifeste ist verpflichtend.

## 9) Operatives Runbook ab Server-Bereitstellung bis Training

### 9.1 Konkrete Parameter des finalen Laufs

```bash
TRAIN_REPO=/home/s-seprag/masterarbeit-strassenerkennung
HANDOFF_DIR="$TRAIN_REPO/data/handoff/thesis_orchestrate_20260313T200624Z_patches_qlr_no_hamburg_20260413"
MASKS_DIR="$TRAIN_REPO/data/patch_masks_final_width_calibration_20260418T195314Z"
INTEGRATION_ID=thesis_orchestrate_20260313T200624Z_1e251e4af7c3_final_width_20260418
INTEGRATION_DIR="$TRAIN_REPO/data/integration/$INTEGRATION_ID"
ENV_PREFIX="$HOME/.conda_envs/masterarbeit-strassenerkennung-gpu"
```

Einordnung:

1. `HANDOFF_DIR` ist die operative Trainingsrepo-Kopie fuer Verify und Materialize.
2. `MASKS_DIR` enthaelt die 54 finalen width-basierten Patch-Masken.
3. `INTEGRATION_DIR` ist der finale materialisierte Trainingsvertrag.
4. `ENV_PREFIX` ist das produktive GPU-Environment auf AppHub/UniCloud.

### 9.2 N0 Server-Checkout aktualisieren

```bash
cd "$TRAIN_REPO"
git pull
git log --oneline -3
```

Erfolg:

1. Aktueller Stand des Trainings-Repos ist eingecheckt.
2. `data/handoff/.../phase5_final_width_contract.json` ist im Checkout vorhanden.
3. Der Checkout enthaelt mindestens den aktuellen Phase-5-Eval-/Mode-Stand.

Hard stop:

1. Bei Merge-/Checkout-Konflikten keinen Trainingslauf starten.

### 9.3 N1 Git LFS initialisieren und ziehen

```bash
cd "$TRAIN_REPO"
micromamba run -p "$ENV_PREFIX" git lfs install
micromamba run -p "$ENV_PREFIX" git lfs pull
```

Erfolg:

1. Getrackte Handoff-Quicklooks sind lokal verfuegbar.
2. `micromamba run -p "$ENV_PREFIX" which git` zeigt auf das Ziel-Environment und nicht auf ein System-`git` ohne LFS.
3. Der finale Maskenordner ist nicht nur als LFS-Pointer, sondern als reale GeoTIFF-Dateien ausgecheckt.

Hard stop:

1. Bei fehlenden LFS-Objekten keine Downstream-Verify starten.

### 9.3.5 N1.5 Server-Environment nach Code-Update auffrischen

```bash
cd "$TRAIN_REPO"
micromamba run -p "$ENV_PREFIX" python -m pip install -e . --no-deps

micromamba run -p "$ENV_PREFIX" \
  python -m pytest -q \
    tests/unit/test_phase5_loader.py \
    tests/unit/test_phase5_eval_and_trainer.py \
    tests/unit/test_phase5_training_mode.py \
    tests/unit/test_run_pipeline_phase5_wrapper.py \
    tests/unit/test_augmentations.py
```

Erfolg:

1. Editable Install zeigt auf den aktuellen Checkout.
2. Die Phase-5-Augmentation-/Loader-Tests laufen im Server-Environment.

Hard stop:

1. Bei fehlendem Albumentations/Torch/CUDA-Environment kein produktives Training starten.
2. Bei Testfehlern zuerst Server-Environment oder Checkout korrigieren.

### 9.4 N2 Finalen Maskenordner am Serverpfad bereitstellen

Was:

1. `"$MASKS_DIR"` muss als Git-LFS-getrackter Pfad im Checkout vorhanden sein.

Wie:

1. Bereitstellung ueber Git-Checkout plus `git lfs install` und `git lfs pull`.
2. Es wird bewusst kein konkretes Kopierwerkzeug vorgeschrieben.

Erfolg:

1. Genau 54 finale Masken liegen unter `"$MASKS_DIR"`.
2. `width_calibration_final_mask_manifest.json` liegt im Maskenordner.

Hard stop:

1. Wenn der Pfad fehlt oder auf `data/patch_masks/` zeigt, ist der Vertrag nicht final.

### 9.5 N2.5 Finalen Width-Handoff vor Materialize validieren

```bash
cd "$TRAIN_REPO"
micromamba run -p "$ENV_PREFIX" \
  python scripts/setup/validate_phase5_final_width_handoff.py \
   --contract "$HANDOFF_DIR/phase5_final_width_contract.json" \
   --masks-dir "$MASKS_DIR"
```

Erfolg:

1. Upstream-Handoff-Contract und finaler Maskensatz sind konsistent validiert.
2. Der Check ist strikt fail-fast bei SHA-/Semantik-Mismatch im upstream Contract oder Maskenmanifest.

Hinweis:

1. Vor `materialize-patches` wird bewusst noch nicht gegen `"$INTEGRATION_DIR"` validiert.
2. Falls dort eine alte Integration liegt, koennte sie sonst einen erwartbaren `phase5_dataset_manifest_sha256 mismatch` oder `dataset manifest selection_id mismatch` ausloesen.

Hard stop:

1. Bei Upstream-Validator-Fehlern keine weitere Materialisierung.

### 9.6 N2.6 Verify im Trainings-Repo

```bash
cd "$TRAIN_REPO"
bash scripts/setup/handoff_check.sh verify-server-patches \
   --handoff-dir "$HANDOFF_DIR" \
   --masks-dir "$MASKS_DIR"
```

Erfolg:

1. Verify-Status: `ok`
2. `patch_count`: `54`
3. `handoff_split_status`: `complete`

Hard stop:

1. Bei Verify-Fehlern keinen Materialize- oder Trainingsstart.

### 9.7 N2.7 Materialize mit Upstream-Split-Authority

```bash
cd "$TRAIN_REPO"
bash scripts/setup/handoff_check.sh materialize-patches \
   --handoff-dir "$HANDOFF_DIR" \
   --masks-dir "$MASKS_DIR" \
   --out-root data/integration \
   --split-policy use_handoff \
   --selection-id "$INTEGRATION_ID"
```

Erfolg:

1. Materialize-Status: `ok`
2. `split_source`: `handoff_imported`
3. `split_policy_requested`: `use_handoff`

Hard stop:

1. Bei Materialize-Fehlern keinen Trainingsstart.

### 9.7.5 N2.8 Finalen Width-Handoff nach Materialize strikt validieren

```bash
cd "$TRAIN_REPO"
micromamba run -p "$ENV_PREFIX" \
  python scripts/setup/validate_phase5_final_width_handoff.py \
    --contract "$HANDOFF_DIR/phase5_final_width_contract.json" \
    --masks-dir "$MASKS_DIR" \
    --integration-dir "$INTEGRATION_DIR" \
    --require-local-artifacts
```

Erfolg:

1. Handoff/Contract/Masken und materialisierte Integration sind konsistent validiert.
2. Der Check gibt `status=ok` und `integration_status=validated` aus.

Hard stop:

1. Bei `phase5_dataset_manifest_sha256 mismatch`, `phase5_samples_sha256 mismatch`, `phase5_local_split_manifest_sha256 mismatch` oder `dataset manifest selection_id mismatch` keinen Trainingsstart.
2. In diesem Fall zuerst neu materialisieren und den Validator erneut laufen lassen.

### 9.8 N3 Readiness-Gate (bestanden, vor Training kurz wiederholen)

```bash
test -f "$INTEGRATION_DIR/meta/phase5_samples.csv" \
  && test -f "$INTEGRATION_DIR/meta/phase5_local_split_manifest.json" \
  && test -f "$INTEGRATION_DIR/meta/phase5_dataset_manifest.json"
```

Erfolg:

1. Alle drei Dateien vorhanden.
2. Erwartete Vertragswerte sichtbar: `patch_count=54`, 27 Gruppen/Tiles, Fold-Verteilung 12/12/10/10/10.
3. `phase5_dataset_manifest.json` zeigt auf den finalen `MASKS_DIR`.

Hard stop:

1. Bei unvollstaendigem Gate kein Trainingsstart.
2. Wenn `masks_dir` wieder auf `data/patch_masks` zeigt, ist der Vertrag nicht final.

### 9.8.5 N3.5 Evidence-Integrity-Gate (vor Trainingsstart verpflichtend)

Was:

1. Vor produktivem Training muss die methodische Herleitungs-Evidenz vollstaendig referenzierbar bleiben.

Wie (minimaler Gate-Check):

```bash
DS_ROOT=<dataselector-repo>

test -f "$DS_ROOT/outputs/runs/thesis_orchestrate_20260313T200624Z/parameter_resolution/optuna_autoscale_summary_20260313.csv" \
  && test -f "$DS_ROOT/outputs/runs/thesis_orchestrate_20260313T200624Z/parameter_resolution/sampler_resolution/selected_sampler.json" \
  && test -f "$DS_ROOT/outputs/runs/thesis_orchestrate_20260313T200624Z/final_config_20260313T201819Z.yaml" \
  && test -f "$DS_ROOT/outputs/runs/thesis_orchestrate_20260313T200624Z/validation/validation_summary_stats.csv"
```

Erfolg:

1. Alle Primary-Evidence-Dateien vorhanden.
2. Herleitung von `n_samples`, Sampler, Snapshot und Validierungskennzahlen ist fuer Audit/Review direkt nachweisbar.

Hard stop:

1. Bei fehlender Primary-Evidence kein produktiver Trainingsstart.
2. Zuerst Evidenz in die dauerhafte Evidenzzone kuratieren und erst dann Trainingslauf starten.

### 9.9 N4 Aktueller Kampagnenstand (2026-04-24)

Aktueller operativer Stand im Trainingsrepo:

1. Die serverseitige Pilotphase fuer `full_1024_no_aug` ist abgeschlossen.
2. Erfolgreich vorliegend sind sechs Pilot-Trainingslaeufe:
   - `unetpp`, Fold `0`, `epochs=200`
   - `unetpp`, Fold `3`, `epochs=200`
   - `segformer`, Fold `0`, `epochs=200`
   - `segformer`, Fold `3`, `epochs=200`
   - `mapsam`, Fold `0`, `epochs=200`
   - `mapsam`, Fold `3`, `epochs=200`
3. Zu jedem dieser Laeufe liegt genau ein diagnostischer Threshold-Sweep vor mit `checkpoint_kind=all` und Thresholds `0.3/0.4/0.5/0.6`.
4. Die feste modellbezogene Budgetentscheidung ist abgeschlossen:
   - `unetpp = 200`
   - `segformer = 200`
   - `mapsam = 150`
5. Naechster aktiver Operativschritt ist die 9-Run-Hauptkampagne auf Folds `1/2/4`.
6. Der klassische Morphologie-Track ist ein separater, nicht lernender
   Vergleichstrack und wird nicht zur neuronalen Budgetentscheidung genutzt.

Methodische Festlegung:

1. Pilotfolds `0` und `3` bleiben diagnostisch und gehen nicht in die Haupttabelle oder in die globale Threshold-Kalibration ein.
2. Die Haupttabelle basiert ausschliesslich auf den neun Hauptruns, finalen Checkpoints und einem globalen Threshold.
3. `iou_best` und `apls_best` bleiben Zusatzanalysen nach eingefrorener Haupttabelle.
4. Fuer den klassischen Track duerfen Folds `0/3` nur zur Auswahl des
   eingefrorenen Morphologie-Kandidaten genutzt werden; die berichtete
   Main-Evaluation erfolgt ausschliesslich auf `1/2/4`.

### 9.10 N5 Pflicht-Smokes vor oder parallel zum Kampagnenlauf

Ziel:

1. Vor produktiven Langlaeufen muessen die nachgelagerten Interfaces nachweisbar zum Phase-5-Schema passen.

Pflicht-Smokes:

1. Timestamped checkpoint root smoke
   - Erwarteter Root: `models/checkpoints/<model>/<mode>_<timestamp>/phase5_fold_<fold>`
   - `cv_summary_*.json` muss `checkpoint_root` enthalten
   - im Root muessen mindestens `checkpoint_best.pt` und `checkpoint_epoch_010.pt` liegen
2. Ranking schema smoke
   - `rank_models.py` muss mit einer repräsentativen reevaluated Phase-5-CSV ohne manuelle Nachbearbeitung laufen
   - insbesondere muss das unpraefixierte `apls`/`iou`-Schema akzeptiert werden

Akzeptanz:

1. Beide Smokes sind erfolgreich oder es wurden ausschliesslich kleine Interface-/Output-Pfad-Angleichungen vorgenommen.
2. Fold-Zuordnung, Budgetregel, Threshold-Regel und Rankinglogik bleiben unveraendert.

### 9.11 N6 Pilotphase zur Budgetableitung

Pilotmatrix:

1. Modelle: `unetpp`, `segformer`, `mapsam`
2. Modus: `full_1024_no_aug`
3. Folds: `0` und `3`
4. Epochenbudget je Pilotlauf: `200`
5. Trainingsdefaults:
   - `dataset_mode=phase5_patches`
   - `phase5_split_policy=use_handoff`
   - `--no-augmentation`
   - kein `--subpatch-size`
   - `batch_size=2`
   - `num_workers=4`
   - `patience=0`
   - `device=cuda`
   - `--no-wandb`
   - fuer MapSAM dieselben SAM-/DoRA-Parameter wie spaeter in der Hauptkampagne

Diagnostische Sweep-Regel:

1. Pro Pilotlauf genau ein Sweep mit `scripts/evaluation/sweep_phase5_thresholds.py`
2. Parameter:
   - `--checkpoint-kind all`
   - `--thresholds 0.3 0.4 0.5 0.6`
   - `--folds <pilotfold>`
3. In die Budgetanalyse gehen ausschliesslich Zeilen mit:
   - `checkpoint_kind=epoch`
   - `epoch in {50, 100, 150, 200}`

Aktueller Status:

1. Alle sechs Pilot-Sweeps liegen serverseitig vor.
2. Die Pilotmatrix ist damit operativ abgeschlossen.

### 9.12 N7 Budgetentscheidung und Budgetfreeze

Entscheidungsregel pro Modell:

1. `A_best` ist das Maximum von `apls_mean_over_thresholds`.
2. Eligible Budgets erfuellen: `A_best - apls_mean_over_thresholds <= 0.01`.
3. Auswahl innerhalb der eligible Menge:
   - hoechstes `iou_mean_over_thresholds`
   - bei weiterem Gleichstand das kleinere Budget

Dokumentierte Pilotentscheidung:

| Modell | gewaehltes Hauptbudget | methodische Einordnung |
| --- | ---: | --- |
| `unetpp` | 200 | innerhalb der APLS-Eligible-Menge; IoU-Tie-Break bevorzugt 200 |
| `segformer` | 200 | innerhalb der APLS-Eligible-Menge; IoU-Tie-Break bevorzugt 200 |
| `mapsam` | 150 | einziges eligible Budget; 200 faellt klar ab |

Pflichtartefakte im Trainingsrepo:

1. `results/pilot_budget_summary.csv`
2. `results/pilot_budget_decisions.md`

Budgetfreeze-Regel:

1. Diese drei Budgets werden unveraendert in die Hauptkampagne uebernommen.
2. Keine nachtraegliche budgetweise Modellbevorzugung innerhalb der Hauptkampagne.
3. Nach abgeschlossenem Budgetfreeze bleiben als kanonische Pilot-Belegartefakte mindestens erhalten:
   - `results/pilot_budget_summary.csv`
   - `results/pilot_budget_decisions.md`
   - die 6 Pilot-`cv_summary_*.json`
   - die 6 Pilot-Sweep-CSVs
   - in den Pilot-Roots mindestens `checkpoint_best.pt` sowie `checkpoint_epoch_050.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_200.pt`

### 9.13 N8 Hauptkampagne fuer das Thesis-Hauptergebnis

Hauptmatrix:

1. Hauptfolds: `1`, `2`, `4`
2. Modi:
   - `full_1024_no_aug`
   - `crop_512_no_aug`
   - `crop_512_aug`
3. Modellbudgets:
   - `unetpp`: `200` in allen drei Modi
   - `segformer`: `200` in allen drei Modi
   - `mapsam`: `150` in allen drei Modi

Trainingsregel:

1. Start mit `batch_size=2`.
2. Falls der erste Lauf eines Modells OOMt, dieses Modell konsistent auf `batch_size=1` umstellen und alle restlichen Modi desselben Modells mit `batch_size=1` fahren.
3. Keine Batch-Size-Mischung innerhalb eines Modells nach Stabilisierung.
4. Ein Run, der nicht vollstaendig auf Folds `1/2/4` durchlaeuft, gilt als fehlgeschlagen und wird vollstaendig neu gestartet.

Ausfuehrungsregel:

1. `run_pipeline_phase5_cv.sh` darf nur dann verwendet werden, wenn `--folds 1 2 4`, modellspezifische Epochenbudgets, `--subpatch-size` und Augmentierungssettings nachweislich unveraendert durchgereicht werden.
2. Fuer die aktuelle Kampagne ist direkter Aufruf von `scripts/training/train_cv.py` der sichere Standard, insbesondere wegen explizitem `--no-wandb`, Fold-Override und Modellbudget.
3. Produktive Langlaeufe sollen in `results/run_logs/` mit timestamped Logdatei mitgeschrieben werden.

Operative Speicherregel bei begrenztem Serverplatz:

1. Pilot-Roots duerfen nach abgeschlossenem Budgetfreeze ausgeduennt werden, solange die in N7 genannten Minimalartefakte erhalten bleiben.
2. Historische 10-Epochen-Smokes sind nicht Teil der kanonischen Kampagnen-Evidenz.
3. Haupt-Checkpoint-Roots duerfen vor Abschluss der finalen Re-Evaluationsschritte nicht ersatzlos geloescht werden.
4. Wenn Serverplatz knapp ist, muessen Haupt-Roots vor einer Loeschung zuerst extern archiviert werden; lokal unverzichtbar bleiben mindestens `cv_summary_*.json`, `cv_metrics_*.csv`, Sweep-CSVs und Run-Logs.
5. Operativer Standard ist jetzt ein archivierungsgekoppelter Hauptkampagnen-Orchestrator im Trainingsrepo: Run validieren, Checkpoint-Root und kleine Artefakte nach `onedrive:Masterarbeit/phase5/main_campaign/` kopieren, mit `rclone check` verifizieren, danach lokalen Haupt-Root entfernen.
6. Es darf hoechstens ein fertig trainierter, noch nicht geloeschter Haupt-Root parallel zu einem aktiven Trainingslauf existieren; faellt der freie Platz unter den definierten Watermark, blockiert der Orchestrator bis Archivierung und Loeschung abgeschlossen sind.
7. Spaetere `final`-, `iou_best`- und `apls_best`-Re-Evaluationen duerfen auf archivierten Runs basieren, aber nur nach explizitem Restore des vollstaendigen Checkpoint-Layouts.

### 9.14 N9 Hauptkalibration, Haupttabelle und Zusatzanalysen

Hauptkalibration:

1. Fuer jeden der 9 Hauptrun-Roots einen Sweep mit:
   - `--checkpoint-kind final`
   - `--thresholds 0.2 0.3 0.4 0.5 0.6 0.7`
   - `--folds 1 2 4`
2. Fuer `crop_*` zusaetzlich `--subpatch-size 512`.
3. Fuer MapSAM dieselben SAM-/DoRA-Argumente wie im Training.
4. Alle 9 Sweep-CSVs gehen gemeinsam in `select_global_threshold.py`.
5. Auswahlregel:
   - maximales `mean(APLS)`
   - Tie innerhalb `0.01` via `mean(IoU)`
   - dann Naehe zu `0.5`

Haupttabelle:

1. Die einzige Quelle der Haupttabelle sind die 9 re-evaluated Final-CSV-Dateien.
2. Pro Hauptrun:
   - `reevaluate_phase5_checkpoints.py --checkpoint-kind final`
   - `--threshold-calibration-json results/phase5_threshold_calibration.json`
   - `--folds 1 2 4`
3. Ranking pro Modus mit `rank_models.py`, Regel: APLS primaer, IoU Tie-Break, `apls_tie_eps=0.01`.

Zusatzanalysen:

1. Erst nach eingefrorener Haupttabelle dieselben 9 Hauptruns zusaetzlich mit `--checkpoint-kind iou_best` und `--checkpoint-kind apls_best` reevaluieren.
2. Auch diese Zusatzanalysen verwenden denselben eingefrorenen globalen Threshold.
3. Sie fliessen nicht in den Hauptvergleich ein.

Empfohlene Reihenfolge pro Hauptrun auf platzbegrenzten Servern:

1. Training fertig laufen lassen und den neuen `checkpoint_root` aus `cv_summary_*.json` notieren.
2. Den zugehoerigen Final-Threshold-Sweep fuer genau diesen Root erzeugen und die Sweep-CSV lokal behalten.
3. Nach Abschluss aller 9 Hauptruns den globalen Threshold bestimmen.
4. Danach `final`, `iou_best` und `apls_best` mit dem eingefrorenen Threshold reevaluieren.
5. Erst wenn diese Re-Evaluationspflicht fuer einen Hauptrun erfuellt oder der Root extern archiviert ist, darf der grosse Haupt-Root lokal entfernt werden.

### 9.14.5 N9.5 Klassischer Morphologie-Track

Ziel:

1. Einen thesis-faehigen klassischen Vergleichstrack bereitstellen, ohne die
   neuronale Pilotmatrix oder die Main-Folds methodisch zu vermischen.
2. Der Track wird als pilot-kalibrierte deterministische klassische
   Bildverarbeitungs-Baseline beschrieben, nicht als vollstaendig optimiertes
   klassisches Verfahren.

Pilot-Kalibrierung:

```bash
cd "$TRAIN_REPO"
micromamba run -p "$ENV_PREFIX" \
  python scripts/evaluation/evaluate_morphological_baseline.py \
  --integration-dir "data/integration/$INTEGRATION_ID" \
  --mode full_1024_no_aug \
  --folds 0,3 \
  --pipeline-id all \
  --candidate-set compact \
  --select-best \
  --write-patch-metrics \
  --out-dir results
```

Regel:

1. Der Compact-Sweep enthaelt `40` Kandidaten: `8` je klassischer Familie
   (`otsu_clahe`, `adaptive_gaussian`, `blackhat_otsu`,
   `sato_dark_ridges`, `canny_bridge`).
2. Auswahlregel: hoechstes mittleres APLS, Tie innerhalb `0.01` via IoU,
   danach Dice, danach stabile Familie-/Kandidaten-Reihenfolge.
3. Die Selection-JSON speichert den globalen Gewinner, die besten Kandidaten
   je Familie, konkrete Parameter, `selection_folds` und
   `selection_patch_count`.

Eingefrorene Main-Evaluation:

```bash
export MORPH_SEL=results/classical_pipeline_selection_full_1024_no_aug_<timestamp>.json

micromamba run -p "$ENV_PREFIX" \
  python scripts/evaluation/evaluate_morphological_baseline.py \
  --integration-dir "data/integration/$INTEGRATION_ID" \
  --mode full_1024_no_aug \
  --folds 1,2,4 \
  --pipeline-selection-json "$MORPH_SEL" \
  --write-patch-metrics \
  --out-dir results
```

Hard stop:

1. Die Main-Evaluation darf nicht auf Folds laufen, die in
   `selection_folds` der Selection-JSON stehen.
2. Keine manuelle Nachjustierung von Morphologie-Parametern auf Folds `1/2/4`.
3. Alte Einzelparameter-Flags fuer Morphologie sind nicht Teil des
   autoritativen Thesis-Workflows.

### 9.15 N10 Abschlusskriterien fuer thesis-ready Kampagnenergebnisse

Ein Ergebnisblock ist thesis-ready, wenn alle Punkte erfuellt sind:

1. Preflight-Gates erfolgreich:
   - timestamped checkpoint root smoke
   - ranking schema smoke mit repräsentativer reevaluated Phase-5-CSV
2. Pilotphase abgeschlossen:
   - 6 Pilot-`cv_summary_*.json`
   - 6 diagnostische Pilot-Sweep-CSVs
   - `pilot_budget_summary.csv`
   - `pilot_budget_decisions.md`
3. Hauptkampagne abgeschlossen:
   - 9 neue `cv_summary_*.json`
   - 9 timestamped checkpoint roots
   - alle Hauptruns nur auf `1/2/4`
4. Hauptauswertung abgeschlossen:
   - 9 final-threshold sweep-CSVs
   - `results/phase5_threshold_calibration.json`
   - 9 reevaluated final-CSVs
   - 3 modusspezifische Ranking-Artefakte
5. Klassischer Morphologie-Track abgeschlossen:
   - `cv_metrics_morphology_*classical_candidates*.csv`
   - `classical_pipeline_selection_full_1024_no_aug_*.json`
   - finale `cv_metrics_morphology_full_1024_no_aug_0_final_<pipeline_id>_*.csv` auf Folds `1/2/4`
6. Methodische Trennung bleibt explizit:
   - Pilotfolds `0/3` tauchen nicht in Haupttabelle oder globaler Threshold-Kalibration auf
   - Morphologie-Pilotfolds `0/3` tauchen nicht in der eingefrorenen Morphologie-Main-Evaluation auf
   - `iou_best` und `apls_best` bleiben Zusatzanalyse

## 10) Vorlage fuer die weitere Protokollfortschreibung

Bei jedem neuen Schritt verpflichtend eintragen:

1. Datum/Uhrzeit
2. Schritt-ID
3. Ziel des Schritts
4. Ausgefuehrtes Kommando
5. Ergebnis (Pass/Fail)
6. Belegdateien
7. Entscheidung/Nachfolgeaktion

## 11) Kurzfazit

1. Die erfolgreiche Kette von Freeze ueber Annotation und Width bis zur final materialisierten Downstream-Integration ist abgeschlossen und belegt.
2. Der finale Trainingsvertrag nutzt die width-basierten Masken aus `final_width_px`, nicht den frueheren Debug-/Smoke-Maskensatz.
3. Die Pilotkampagne der neuen Phase-5-Hauptauswertung ist abgeschlossen; die Hauptbudgets sind eingefroren (`unetpp=200`, `segformer=200`, `mapsam=150`).
4. Der aktuelle Gate-Zustand ist: Readiness bestanden, Pilotbudgetdiagnostik abgeschlossen, Hauptkampagne auf Folds `1/2/4` aktiv; vier valide Hauptlaeufe liegen vor, die restlichen Runs laufen unter archivierungsgekoppelter Orchestrierung weiter.
5. Die Haupttabelle ist methodisch auf finale Checkpoints, einen globalen Threshold und APLS-primaeres Ranking mit IoU-Tie-Break festgelegt.
6. Der klassische Morphologie-Track ist als separater pilot-kalibrierter
   Baseline-Track festgelegt: Compact-Sweep auf `0/3`, frozen Evaluation auf
   `1/2/4`.

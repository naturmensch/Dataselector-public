# ExperimentManager: Professionelle Versionierung und Reproducibility

## Überblick

Das neue **ExperimentManager**-System bietet:

1. **Automatische Versionierung**: Jeder Run bekommt ein datiertes, eindeutiges Verzeichnis
2. **Vollständige Provenance**: Git-Info, Python-Version, Packages, Umgebung
3. **Inkrementelle Ergebnisse**: Trials werden während des Runs gespeichert (nicht nur am Ende)
4. **Hierarchische Struktur**: Klare Trennung zwischen Config, Results, Logs, Artifacts
5. **Manifest**: `manifest.json` mit kompletten Metadaten für jede Run
6. **Status-Tracking**: Welche Stages abgeschlossen sind, Fehlerstatus

Runtime policy:

1. Canonical invocation is `micromamba run -n dataselector <command>`.
2. `scripts/exec_in_env.sh` remains available as compatibility wrapper.
3. For thesis production orchestration prefer:
   `micromamba run -n dataselector python -m dataselector thesis-orchestrate`.

Scientific hardening policy (Phase 4H v5):

1. Runs must capture feature cache identity and model provenance in metadata.
2. Contract evidence is run-relative by default; `repo:` evidence is explicit.
3. Force overrides require an explicit reason and are always logged.
4. Diagnostic autoscale stages are never allowed to define production parameters.
5. Spatial min-distance references are derived from strict metric CRS coordinates
   (EPSG:25832 target), not raw display projection units.
6. Leakage-safe split artifacts (`distance_policy.json`, `split_manifest.json`,
   `leakage_audit.csv`) are part of the run-level reproducibility chain.
7. Fair model comparison requires identical `split_manifest_sha256` across model runs.
8. `n_samples` staging defaults are globally resolved in autoscale core from
   effective candidate count (`N_eff`) using the corridor policy in
   `config/pipeline_config.yaml`.
9. Final `n_samples` is selected by minimal-feasible plateau rule and persisted in
   `parameter_resolution/optuna_autoscale_selected_n_samples.txt`.

## Struktur

```
outputs/runs/
├── 20260116_T160213_hamburg_optuna_n2000/
│   ├── manifest.json                          ← Komplette Metadaten des Runs
│   ├── config/
│   │   ├── config_optuna.yaml                 (Optuna-Parameter)
│   │   └── config_best_selection.yaml         (Beste gefundene Gewichte)
│   ├── results/
│   │   ├── trials.csv                         (Alle Trials, inkrementell geschrieben)
│   │   ├── best_trial.json
│   │   ├── candidate_set.csv
│   │   └── convergence.csv                    (Optional: nachgelagert)
│   ├── logs/
│   │   ├── experiment.log                     (DEBUG + INFO, alles)
│   │   └── status.log                         (INFO+, für Monitoring)
│   ├── artifacts/
│   │   ├── convergence_plot.png
│   │   ├── feature_correlation.csv
│   │   └── ... (beliebige Zwischen-Outputs)
│   └── monitor/
│       └── snapshots/
│           ├── snapshot_20260116T160213.txt
│           └── ...
│
├── 20260115_T234520_baseline_optuna_500/      (Frühere Runs)
└── 20260115_T120000_fine_sweep/               (Verschiedene Stages)
```

## Verwendung

### Option 1: Direkt über CLI (EMPFOHLEN)

```bash
micromamba run -n dataselector python -m dataselector optuna-optimize \
  --n-trials 2000 \
  --n-candidates 800 \
  --exp-name hamburg_optuna_n2000 \
  --sampler tpe
```

**Automatisch erstellt:**
```
outputs/runs/20260116_T160213_hamburg_optuna_n2000/
  ├── manifest.json (mit Git-Commit, Python-Version, Packages)
  ├── config/config_optuna.yaml (alle Parameter)
  ├── results/trials.csv (inkrementell während Lauf)
  ├── logs/experiment.log (vollständiger Output)
  └── logs/status.log (für Live-Monitoring)
```

### Option 2: In Custom-Scripts verwenden

```python
from dataselector.pipeline.experiment_manager import ExperimentManager

# Experiment initialisieren
em = ExperimentManager(
    name="custom_analysis",
    description="My custom tile selection experiment",
    metadata={"thesis_chapter": "4", "version": "v2"}
)

# Konfiguration speichern
em.save_config("selection", {
    "n_samples": 50,
    "min_distance": 40,
    "weights": {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
})

# Ergebnisse speichern
em.save_results("tiles", selected_tiles_df, format="csv")

# Artifacts archivieren
em.save_artifact("plots/convergence.png", "convergence_plot", category="plot")

# Stages tracken
em.mark_stage_complete("exploration", summary={"n_samples": 100})

# Logging
em.log("Processing complete")
em.log("Something went wrong", level="error")

# Finalisieren
em.save_manifest()
em.mark_complete(success=True)

# Summary ausgeben
print(em.summary())
```

## Key Features

### 1. Inkrementelle CSV-Schreibvorgänge

```python
from dataselector.pipeline.incremental_results import IncrementalCSVWriter, TrialBuffer

# Writer für große Dateien
writer = IncrementalCSVWriter(
    "trials.csv",
    fieldnames=["trial_id", "value", "params"],
    buffer_size=100  # Alle 100 Rows schreiben
)

# Trials hinzufügen
for trial in all_trials:
    writer.append({"trial_id": trial.id, "value": trial.value, ...})

writer.close()  # Flush any remaining
```

**Vorteil:** Wenn der Prozess abbricht, sind die bisherigen Ergebnisse schon gespeichert!

### 2. Provenance Tracking

Das Manifest erfasst automatisch:
- Git-Commit und -Branch
- Dirty-Status (ungespeicherte Änderungen)
- Python-Version
- Installierte Packages (optuna, pytorch, pandas, etc.)
- Hostname, Zeitstempel
- Komplette Befehlszeile (in config/)

### 3. Stage-Basiertes Tracking

```python
em.mark_stage_complete(
    "optuna",
    summary={
        "n_trials_completed": 2000,
        "best_value": 77.7,
        "best_trial": 640,
        "duration_sec": 3600
    }
)
```

Daraus entsteht in `manifest.json`:
```json
{
  "experiment": {
    "stages": {
      "optuna": {
        "status": "complete",
        "timestamp": "2026-01-16T17:30:00Z",
        "summary": { "n_trials_completed": 2000, ... }
      }
    }
  }
}
```

## Für die Pipeline

### Pipeline Integration (Option B)

Die Pipeline (`micromamba run -n dataselector python -m dataselector adaptive-pipeline`) initialisiert **ein zentrales Run-Verzeichnis** (via `ExperimentManager`) und gibt dessen Pfad an Sub-Stage-Module weiter.

- Die Pipeline erstellt das Run-Verzeichnis und exportiert dessen Pfad in die Umgebungsvariable `EXPERIMENT_RUN_DIR`.
- Stage-Skripte (Exploration, Fine Sweep, Optuna, Bootstrap) erkennen `EXPERIMENT_RUN_DIR` automatisch und **hängen sich an** den vorhandenen Run via `ExperimentManager.from_existing(run_dir)` an.
- Jeder Stage speichert Konfiguration, Ergebnisse und Artefakte direkt unter `outputs/runs/<run_id>/` und ruft `em.mark_stage_complete(<stage>, summary=...)` auf.

Beispiel-CLI (Pipeline):

```bash
micromamba run -n dataselector python -m dataselector adaptive-pipeline \
  --n-lhs 50 --n-trials 200 --n-candidates 500 \
  --experiment-name adaptive_full
```

**Vorteile:** einheitliches Manifest, reproduzierbare Runs, per-stage Status und bessere Observability.

### Run mit Hamburg:
```bash
micromamba run -n dataselector python -m dataselector optuna-optimize \
  --n-trials 2000 \
  --n-candidates 800 \
  --exp-name hamburg_n2000 \
  --sampler tpe
```

**Resultat:** 
- ✅ Alle 2000 Trials sind WÄHREND des Runs in `results/trials.csv` gespeichert
- ✅ Wenn Prozess abstürzt bei Trial 1429, sind diese 1429 schon on Disk
- ✅ `manifest.json` hat komplette Reproducibility-Info
- ✅ `logs/experiment.log` hat jeden Schritt

### Alte statische Dateien (deprecated):
```
outputs/optuna_results.csv           ← NICHT MEHR NUTZEN
outputs/optuna_candidate_set.csv    ← NICHT MEHR NUTZEN
outputs/pipeline_config.optuna.yaml ← NICHT MEHR NUTZEN
```

Diese bleiben nur für Rückwärtskompatibilität, sollten aber ignoriert werden.

## Monitoring während des Runs

**Hinweis:** Falls `results/trials.csv` nach einem abgeschlossenen Lauf fehlt (z. B. durch einen unterbrochenen Schreibvorgang), nutze die CLI-Auswertung über `micromamba run -n dataselector python -m dataselector generate-monitor --run-dir <run_dir>` und prüfe anschließend, ob der Run mit `thesis-orchestrate` oder `thesis-pipeline` erneut erzeugt werden muss.

Resume / Restart:

- Nutze `python -m dataselector thesis-orchestrate` für einen erneuten orchestrierten Lauf.
- Nutze `python -m dataselector generate-monitor --run-dir <run_dir>` für die zusammenfassende Laufauswertung.


```bash
# Live monitor
tail -F outputs/runs/20260116_T160213_hamburg_optuna_n2000/logs/status.log

# Oder mit watch
watch -n 5 'tail -20 outputs/runs/20260116_T160213_hamburg_optuna_n2000/logs/status.log'

# Oder Python für schönere Ausgabe
python - <<'EOF'
import json
from pathlib import Path

manifest_path = Path("outputs/runs").glob("*hamburg*/manifest.json").__next__()
manifest = json.load(open(manifest_path))
print(f"Run: {manifest['experiment']['run_id']}")
print(f"Status: {manifest['experiment']['status']}")
for stage, info in manifest['experiment'].get('stages', {}).items():
    print(f"  {stage}: {info['status']}")
EOF
```

## Migration: Alt → Neu

**Alte Weise:**
```bash
python -m dataselector optuna-optimize --n-trials 2000
# Speichert alles statisch in outputs/
# Bei neuem Run: Daten werden überschrieben ❌
```

**Neue Weise:**
```bash
python -m dataselector optuna-optimize --n-trials 2000 --exp-name hamburg_n2000
# Speichert versioniert in outputs/runs/20260116_T160213_hamburg_n2000/
# Alle alten Runs bleiben erhalten ✅
```

## Status heute (16. Januar 2026)

✅ **Implementiert:**
- ExperimentManager-Klasse (dataselector/pipeline/experiment_manager.py)
- IncrementalCSVWriter, TrialBuffer (dataselector/pipeline/incremental_results.py)
- optuna_optimize.py umgestellt auf neue Struktur
- Inkrementelle Trial-Speicherung während Optuna läuft
- Provenance-Tracking (Git, Packages, Environment)

🔄 **Nächste Schritte:**
- adaptive-pipeline weiter ausbauen
- Monitor-Integration
- Test-Run mit vollständiger Pipeline

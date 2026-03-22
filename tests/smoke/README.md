# Smoke Tests — Quick README ✅

Kurz: Diese Anleitung erklärt, wie man schnelle Smoke‑Runs startet und Logs direkt überwacht.

## Ziel
- Kurzlauf (smoke) ausführen, Logs live ansehen und auf offensichtliche Fehler prüfen.

## 1) Kurzer Smoke‑Run starten
```bash
ts=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="outputs/runs/thesis_smoke_${ts}"
log="outputs/logs/thesis_smoke_${ts}.log"
mkdir -p outputs/runs outputs/logs

# canonical quick preflight
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup \
  micromamba run -n dataselector -- \
  python -m dataselector thesis-pipeline \
  --execution-profile thesis_repro \
  --dry-run \
  --output-dir "$run_dir" \
  > "$log" 2>&1 & echo $!
```
Merke dir die ausgegebene PID (oder suche sie mit `pgrep -f "python -m dataselector thesis-pipeline"`).

## 2) Schnell überwachen (empfohlen)
- Folgen des Logs:
```bash
tail -n 100 -f "$log"
```
- Monitor-Zusammenfassung für einen **echten** Thesis-Run erzeugen:
```bash
python -m dataselector generate-monitor --run-dir outputs/runs/<run_id>
```

## 3) Was prüfen (Quick‑Checks)
- In Logs nachsehen auf: `FAILED`, `Traceback`, `Exception`, `ERROR` oder `FAILED (see`.
- Schrittweise Marker: `=== [<label>] Starting` und `Completed:` zeigen Fortschritt an.
- Nach Abschluss: prüfe `outputs/runs/run_<TIMESTAMP>/` auf Artefakte:
  - `pipeline_config.used.yaml` (eingesetzte Konfiguration)
  - `run_metadata.json`
  - bei echten Nicht-Dry-Runs zusätzlich `THESIS_PIPELINE_REPORT.md` und weitere Run-Artefakte

## 4) Wenn etwas schief läuft
- Abbruch: `kill <PID>` (ggf. `kill -9 <PID>` als letzter Ausweg).
- Speicher/CPU‑Engpässe prüfen mit `top -p <PID>` oder `ps -p <PID> -o pid,%cpu,%mem,etime,cmd`.
- Bei wiederholten Fehlern: speichere relevante Logabschnitte und öffne ein Issue mit den Logs und Befehls-Args.

## 5) Smoke‑Test Checklist (kurz)
- [ ] Run gestartet (PID erhalten)
- [ ] Logs zeigen keine `Traceback` / `FAILED` in den ersten ~2min
- [ ] `outputs/runs/thesis_smoke_<TIMESTAMP>/pipeline_config.used.yaml` vorhanden
- [ ] Für echte Runs kann `generate-monitor --run-dir outputs/runs/<run_id>` einen Run-Bericht erzeugen

---
Wenn du möchtest, kann ich noch eine CI‑Job‑Skizze für automatisierte Smoke‑Runs hinzufügen (z. B. GitHub Actions).

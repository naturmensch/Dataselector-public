# Smoke Tests — Quick README ✅

Kurz: Diese Anleitung erklärt, wie man schnelle Smoke‑Runs startet und Logs direkt überwacht.

## Ziel
- Kurzlauf (smoke) ausführen, Logs live ansehen und auf offensichtliche Fehler prüfen.

## 1) Kurzer Smoke‑Run starten
```bash
# moderate limits for quick smoke
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python -m dataselector xxl --smoke > outputs/experiments/run_smoke.log 2>&1 & echo $!
```
Merke dir die ausgegebene PID (oder suche sie mit `pgrep -f "python -m dataselector xxl"`).

## 2) Schnell überwachen (empfohlen)
- Folgen des Logs:
```bash
tail -n 100 -f outputs/experiments/run_smoke.log
```
- Monitor-Zusammenfassung erzeugen:
```bash
python -m dataselector generate-monitor
```

## 3) Was prüfen (Quick‑Checks)
- In Logs nachsehen auf: `FAILED`, `Traceback`, `Exception`, `ERROR` oder `FAILED (see`.
- Schrittweise Marker: `=== [<label>] Starting` und `Completed:` zeigen Fortschritt an.
- Nach Abschluss: prüfe `outputs/experiments/run_<TIMESTAMP>/` auf Artefakte:
  - `pipeline_config.used.yaml` (eingesetzte Konfiguration)
  - `optuna_results.csv` (falls Optuna lief)
  - `bootstrap_results.csv`, `final_selection/*`, `gen_report`-Ausgaben

## 4) Wenn etwas schief läuft
- Abbruch: `kill <PID>` (ggf. `kill -9 <PID>` als letzter Ausweg).
- Speicher/CPU‑Engpässe prüfen mit `top -p <PID>` oder `ps -p <PID> -o pid,%cpu,%mem,etime,cmd`.
- Bei wiederholten Fehlern: speichere relevante Logabschnitte und öffne ein Issue mit den Logs und Befehls-Args.

## 5) Smoke‑Test Checklist (kurz)
- [ ] Run gestartet (PID erhalten)
- [ ] Logs zeigen keine `Traceback` / `FAILED` in den ersten ~2min
- [ ] `outputs/experiments/run_<TIMESTAMP>/pipeline_config.used.yaml` vorhanden
- [ ] Finaler Bericht (`gen_report`) wird erstellt oder es gibt Hinweise im Run‑Log

---
Wenn du möchtest, kann ich noch eine CI‑Job‑Skizze für automatisierte Smoke‑Runs hinzufügen (z. B. GitHub Actions).

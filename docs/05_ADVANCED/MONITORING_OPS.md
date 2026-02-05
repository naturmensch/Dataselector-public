# 🚀 MONITORING & DEPLOYMENT OPERATIONS

**Dokument:** `docs/05_ADVANCED/MONITORING_OPS.md`  
**Zielgruppe:** DevOps, System Operators, XXL Pipeline Monitoring  
**Status:** Production Ready

---

## 📊 Systemd Monitor Service Setup (XXL Full Run)

### Automated XXL Pipeline Monitoring via systemd User Service

**Zweck:** Führe XXL Full-Run-Pipeline automatisch aus und überwache sie im Hintergrund

### Installation (5 Minuten)

```bash
# Copy systemd service files
cp contrib/xxl-monitor.service ~/.config/systemd/user/
cp contrib/xxl-monitor.timer ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload

# Enable timer (nightly execution)
systemctl --user enable xxl-monitor.timer
systemctl --user start xxl-monitor.timer

# Check status
systemctl --user status xxl-monitor.timer
```

### Timer Schedule

**Default:** Täglich um 02:00 Uhr nachts

**Anpassen:**
```bash
systemctl --user edit xxl-monitor.timer
# Edit OnCalendar= line, e.g., `OnCalendar=*-*-* 22:30:00` (22:30 Uhr)
```

### Monitoring

```bash
# View recent logs
journalctl --user -u xxl-monitor.service -n 50

# Follow live logs
journalctl --user -u xxl-monitor.service -f

# Check timer
systemctl --user list-timers xxl-monitor.timer
```

---

## 🔄 Resume & Recovery from Failures

### Monitor Resume Modes

**Syntax:**
```bash
python scripts/xxl_full_run_monitor.py \
  --restart [last|<run_id>|none] \
  [--force-restart] \
  [--dry-run-restart]
```

### Resume Options

| Mode | Verhalten | Use Case |
|------|-----------|----------|
| `--restart last` | Fortsetzen vom letzten Checkpoint | Network-Ausfall, kurze Unterbrechung |
| `--restart <run_id>` | Bestimmte Run-ID wiederherstellen | Multi-Experiment Recovery |
| `--restart none` | Von Scratch starten | Sauberer Neustart |
| `--force-restart` | Überschreibe Locks | Bei Deadlock |
| `--dry-run-restart` | Simuliere Recovery ohne echte Ausführung | Testing |

### Beispiele

```bash
# Nach Interrupt fortsetzen
python scripts/xxl_full_run_monitor.py --restart last

# Bestimmte Run-ID recovern
python scripts/xxl_full_run_monitor.py --restart phase2_reproducibility_20260202_143022

# Trockenlauf für Testing
python scripts/xxl_full_run_monitor.py --dry-run-restart --restart last
```

### Optuna DB Reconstruction

Die Pipeline erstellt automatisch Backups vor Recovery:

```bash
# Backups
outputs/xxl/optuna_study.db.bak_resume_<timestamp>
```

**Bei Integrität-Fehlern:**

```bash
# Check DB integrity
sqlite3 outputs/xxl/optuna_study.db "PRAGMA integrity_check;"

# Optuna DB Reconstruction deaktivieren
python scripts/xxl_full_run_monitor.py --no-reconstruct

# Alternativer Backup Recovery
cp outputs/xxl/optuna_study.db.bak_resume_<timestamp> \
   outputs/xxl/optuna_study.db
```

---

## 📈 Experiment Tracking mit W&B (Weights & Biases)

### W&B Integration

**Status:** Vollständig integriert (23. Januar 2026)

### Neue Module

**`dataselector/wandb_logger.py`:**
```python
from dataselector.wandb_logger import WandBLogger

# Initialize (with graceful fallback if wandb unavailable)
logger = WandBLogger(project="kdr100-dataselector")

# Log config
logger.log_config({
    "alpha": 0.4,
    "beta": 0.3,
    "feature_extractor": "dinov2"
})

# Log trial results
logger.log_trial(trial_number=1, value=67.3, params={"alpha": 0.4})

# Log bootstrap metrics
logger.log_bootstrap(iteration=1, ci_width=2.4, coverage=67.2)
```

### W&B Quick Start

```bash
# Installation
pip install wandb

# Login (one-time)
wandb login
# Follow prompt, get API key from https://wandb.ai/authorize

# Run pipeline with W&B tracking
python scripts/run_thesis_pipeline.py \
  --wandb-project "kdr100-dataselector" \
  --wandb-entity "your-username"

# View results
# https://wandb.ai/your-username/kdr100-dataselector
```

### W&B Dashboard Features

| Feature | Nutzung |
|---------|---------|
| **Config Logging** | Alle Parameter trackbar |
| **Trial Results** | Optuna-Trials in Echtzeit |
| **Bootstrap Metrics** | Confidence Intervals visualisiert |
| **Comparison Charts** | Multi-Experiment Vergleich |
| **Artifact Storage** | CSV Results hochladen |

### W&B Konfiguration

**Config:** `config/wandb_config.yaml` (optional)

```yaml
wandb:
  enabled: true
  project: "kdr100-dataselector"
  entity: "your-workspace"  # or null for personal
  tags: ["thesis", "production"]
  notes: "Phase 1 LHS exploration"
```

### Graceful Fallback

Falls `wandb` nicht installiert ist, läuft die Pipeline trotzdem:

```python
try:
    import wandb
    # Logging enabled
except ImportError:
    # Continue without W&B
    pass
```

---

## ⚙️ Advanced Monitoring

### XXL Pipeline Phasen

**Phase 0–5 Status:**

```
outputs/xxl/
├── phase0_convergence/
│   ├── convergence_report.json
│   └── status.log
├── phase1_hamburg/
│   ├── hamburg_results.csv
│   └── status.log
├── phase2_reproducibility/
│   ├── seed_*.csv
│   └── consistency_check.json
├── phase3_statistics/
│   ├── bootstrap_results.csv
│   └── sensitivity_map.png
├── phase4_summary/
│   ├── final_summary.md
│   └── thesis_ready_report.json
└── phase5_bootstrap_uq/
    ├── final_results.csv
    ├── confidence_intervals.json
    └── monitor.log
```

### Health Checks

```bash
# Check last 100 lines of monitor log
tail -100 outputs/xxl/phase5_bootstrap_uq/monitor.log

# Verify all phases completed
for phase in 0 1 2 3 4 5; do
  [ -f outputs/xxl/phase${phase}_*/status.log ] && echo "Phase $phase: OK" || echo "Phase $phase: MISSING"
done

# Check Optuna DB
sqlite3 outputs/xxl/optuna_study.db "SELECT COUNT(*) as trial_count FROM trials;"
```

---

## 🚨 Troubleshooting

### Monitor Service wird nicht gestartet

```bash
# Check service status
systemctl --user status xxl-monitor.service

# View detailed logs
journalctl --user -u xxl-monitor.service -x

# Manually test
python scripts/xxl_full_run_monitor.py --dry-run-restart
```

### "wandb not installed" Error

```bash
pip install wandb
```

### Monitor Resume Failure

```bash
# Check Optuna DB integrity
sqlite3 outputs/xxl/optuna_study.db "PRAGMA integrity_check;"

# Restore from backup
cp outputs/xxl/optuna_study.db.bak_resume_<TIMESTAMP> outputs/xxl/optuna_study.db

# Retry recovery
python scripts/xxl_full_run_monitor.py --restart last --force-restart
```

### Missing Features

```bash
# Regenerate features
python -c "from dataselector.feature_extraction import extract_features; extract_features()"

# Or run full pipeline from start
python scripts/run_thesis_pipeline.py --no-checkpoint
```

---

## 📚 Weiterführende Ressourcen

- **Quick Start:** [../03_USER_GUIDES/QUICK_START.md](../03_USER_GUIDES/QUICK_START.md)
- **Pipelines:** [../03_USER_GUIDES/PIPELINES.md](../03_USER_GUIDES/PIPELINES.md)
- **Developer Setup:** [../04_DEVELOPER/DEVELOPER_SETUP.md](../04_DEVELOPER/DEVELOPER_SETUP.md)
- **Advanced Tuning:** [ADVANCED_TUNING.md](ADVANCED_TUNING.md)

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

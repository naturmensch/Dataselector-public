# 🚀 QUICK START: Datenselektion für KDR100

**Dokument:** `docs/03_USER_GUIDES/QUICK_START.md`  
**Zielgruppe:** Neue Benutzer, Thesis-Autoren  
**Dauer:** ~15 Minuten zum Setup + ~10 Minuten erste Pipeline  
**Status:** Production Ready

---

## ⚡ Installation (5 Minuten)

```bash
# Repository klonen
git clone <repository-url> Dataselector
cd Dataselector

# Virtual Environment (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows

# Dependencies (inkl. scipy für LHS)
pip install -r requirements.txt

# Optional: GPU Support (CUDA 11.8+)
pip install -r requirements-gpu.txt
```

---

## 🔧 Konfiguration (5 Minuten)

**Standard-Config:** `config/pipeline_config.yaml`

```yaml
# Minimal Setup
feature_extractor: "dinov2"      # oder "resnet50"
min_distance_km: 40              # Räumlicher Constraint
n_clusters: 8                    # K-Means Clustering
weights:
  alpha: 0.4                     # Visuelle Diversität
  beta: 0.3                      # Räumliche Diversität
  gamma: 0.3                     # Zeitliche Diversität
```

---

## ▶️ Erste Datenselektion (10 Minuten)

### Option 1: Komplette 4-Phasen-Pipeline

```bash
python scripts/run_thesis_pipeline.py
```

**Was passiert:**
1. **Phase 1** (~2 min): LHS-Parametererkundung (50 Samples)
2. **Phase 2/3** (~5 min): Bayesian Optimization mit Optuna
3. **Phase 4** (~3 min): Bootstrap Validierung (200 Resamples)

**Output:**
- `outputs/tuning_weights/pareto/pareto_solutions.csv` - Trade-off Plots
- `outputs/optuna_autoscale_best_YYYYMMDD.json` - Beste Parameter
- `outputs/validation_results.csv` - Confidence Intervals

### Option 2: Nur LHS-Exploration (schnell)

```bash
python scripts/tune_weights_and_run.py --n-samples 50
```

**Output:** 50 verschiedene Parameter-Kombinationen zum Visualisieren

### Option 3: Einzelne Tile-Gruppe

```bash
python scripts/run_single_tile.py --tile-id TILE_001 --config config/pipeline_config.yaml
```

---

## 📊 Ergebnisse Verstehen

| File | Inhalt | Nutzung |
|------|--------|---------|
| `pareto_solutions.csv` | α,β,γ vs. Coverage % | Trade-off Visualisierung |
| `optuna_autoscale_best_*.json` | Optimal parameters | Produktive Selektion |
| `validation_results.csv` | 95% CI Interval | Robustheit-Beweis |

### Visualisierung

```bash
# Pareto-Front plotten
jupyter notebook notebooks/pareto_plots.ipynb

# Validierungsergebnisse
python scripts/plot_validation_results.py --input outputs/validation_results.csv
```

---

## ✅ Häufige Anfragen

**F: Wie lange dauert ein kompletter Run?**  
A: ~10-15 Minuten auf Standard-Hardware. Mit GPU: ~5 Minuten.

**F: Kann ich Parameter anpassen?**  
A: Ja! Editiere `config/pipeline_config.yaml` oder nutze CLI-Flags:
```bash
python scripts/run_thesis_pipeline.py \
  --alpha 0.5 --beta 0.3 --gamma 0.2 \
  --min-distance 50 --n-clusters 10
```

**F: Was ist Bootstrap Uncertainty Quantification?**  
A: Siehe [UQ_VALIDATION.md](UQ_VALIDATION.md)

**F: Welcher Sampler ist am besten?**  
A: Siehe [SAMPLERS.md](SAMPLERS.md) für Vergleich (QMC vs TPE vs CMA-ES)

---

## 🔗 Weiterführende Ressourcen

- **Vollständige Pipelines:** [PIPELINES.md](PIPELINES.md)
- **Parameteroptimierung:** [../02_THEORY/methodology.md](../02_THEORY/methodology.md)
- **Entwickler-Setup:** [../04_DEVELOPER/DEVELOPER_SETUP.md](../04_DEVELOPER/DEVELOPER_SETUP.md)
- **Troubleshooting:** [../05_ADVANCED/ADVANCED_OPS.md](../05_ADVANCED/ADVANCED_OPS.md)

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

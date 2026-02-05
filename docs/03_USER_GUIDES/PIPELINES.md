# 📦 PIPELINES: Thesis & XXL Orchestrierung

**Dokument:** `docs/03_USER_GUIDES/PIPELINES.md`  
**Zielgruppe:** Thesis-Autoren, Pipeline-Entwickler  
**Status:** Production Ready

---

## 🎯 Pipeline-Übersicht

### Thesis Pipeline (4 Phasen)

**Zweck:** Wissenschaftliche Parameteroptimierung mit Robustheitsbeweis

```
┌────────────────────────────────────────────────────┐
│ PHASE 1: EXPLORATION (Latin Hypercube Sampling)   │
│ - 50 verschiedene α,β,γ Kombinationen              │
│ - Ziel: Parameterraum verstehen, Trade-offs zeigen│
│ - Output: pareto_solutions.csv                     │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│ PHASE 2/3: OPTIMIZATION (Bayesian + Autoscale)    │
│ - Optuna mit TPE/QMC/CMA-ES Sampler               │
│ - Adaptive Bounds für progressive Refinement       │
│ - Ziel: Mathematisches Optimum finden             │
│ - Output: optuna_autoscale_best_*.json            │
└────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────┐
│ PHASE 4: VALIDATION (Bootstrap + Sensitivität)    │
│ - 200 Resamples × 5 Distanzen × 5 Seeds          │
│ - Robustheitsbeweis & Confidence Intervals        │
│ - Output: validation_results.csv                  │
└────────────────────────────────────────────────────┘
```

**Ausführung:**

```bash
python scripts/run_thesis_pipeline.py \
  --alpha 0.4 --beta 0.3 --gamma 0.3 \
  --n-lhs 50 \
  --n-boot 200
```

**Dauer:** ~10-15 Minuten (Standard-Hardware)

**Output-Struktur:**
```
outputs/
  ├── tuning_weights/
  │   └── pareto/
  │       ├── pareto_solutions.csv      # Phase 1 Output
  │       └── pareto_plots.png          # Visualisierung
  ├── optuna_autoscale_best_*.json      # Phase 2/3 Output
  └── validation_results.csv             # Phase 4 Output
```

---

### XXL Pipeline (Phases 0–5)

**Zweck:** Vollständige experimentelle Thesis-Pipeline mit allen Validierungen

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 0: CONVERGENCE ANALYSIS                           │
│ - Prüfe Optuna Convergence über mehrere Seeds          │
│ - Validiere dass Optimierung stabil ist                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: HAMBURG OPTIMIZATION                           │
│ - Lokal-Optimierung für Hamburg-Region                 │
│ - Validiere räumliche Constraints in realistischer Stadt│
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: REPRODUCIBILITY TESTS                          │
│ - Multi-Seed Vergleich (5 Seeds)                       │
│ - Determinismus-Validierung                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: STATISTICS & ROBUSTNESS                        │
│ - Bootstrap Sampling                                    │
│ - Sensitivity Analysis                                  │
│ - Output: Confidence Intervals & Stability Reports      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 4: SUMMARY GENERATION                             │
│ - Aggregiere alle Ergebnisse                           │
│ - Thesis-ready Report                                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 5: BOOTSTRAP-BASED UQ                             │
│ - Final Uncertainty Quantification                      │
│ - Output: Finalisierte Confidence Intervals             │
└─────────────────────────────────────────────────────────┘
```

**Ausführung:**

```bash
# Vollständige XXL Pipeline (~45 Minuten)
python scripts/run_xxl_pipeline.py \
  --phases 0 1 2 3 4 5 \
  --seeds 42 43 44 45 46 \
  --n-boot 200
```

**Output-Struktur:**
```
outputs/xxl/
├── phase0_convergence/
├── phase1_hamburg/
├── phase2_reproducibility/
├── phase3_statistics/
├── phase4_summary/
└── phase5_bootstrap_uq/
    ├── final_results.csv
    ├── confidence_intervals.json
    └── stability_report.md
```

**Dauer:** ~45 Minuten (alle Phasen)

---

## 🔧 Sampler Vergleich

**Eingebundene Sampler:**
| Sampler | Methode | Einsatz | Performance |
|---------|---------|---------|-------------|
| **QMC (Sobol)** | Quasi-Monte Carlo | Default, deterministisch, Thesis | Konsistent |
| **TPE** | Tree-structured Parzen | Bayesian, empirisch beste Ergebnisse | +0.44% fitness |
| **CMA-ES** | Covariance Matrix Adaptation | Evolutionary, robust bei Constraints | Stabil |

**Sampler auswählen:**

```bash
# TPE (empfohlen für Bayesian Optimization)
python scripts/run_thesis_pipeline.py --sampler tpe

# QMC (deterministisch, reproducible)
python scripts/run_thesis_pipeline.py --sampler qmc

# CMA-ES (evolutionär, robust)
python scripts/run_thesis_pipeline.py --sampler cmaes

# Alle vergleichen (Sampler Suite)
python scripts/run_thesis_sampler_suite.py \
  --n-seeds 10 \
  --n-trials 1000
```

**Output:** `outputs/selected_sampler.json` mit Empfehlung

---

## 📊 Pipeline Visualisierung & Interpretation

### Pareto-Front Plots (Phase 1)

```bash
jupyter notebook notebooks/pareto_plots.ipynb
```

**Interpretation:**
- X-Achse: Visuelle Diversität (α)
- Y-Achse: Räumliche Diversität (β)
- Farbe: Coverage %
- Frontlinie: Nicht-dominiert (auswählen!)

### Validation Results (Phase 4)

```bash
python scripts/plot_validation_results.py --input outputs/validation_results.csv
```

**Metriken:**
- **95% CI Width:** Eng = stabil, Breit = sensitiv
- **Bootstrap Variance:** Niedrig = robust
- **Seed-Konsistenz:** Hoch = reproducible

---

## 🚨 Troubleshooting

**F: Pipeline bricht ab mit "No convergence"?**  
A: Reduziere `--n-trials` oder nutze `--sampler qmc` für stabilere Konvergenz.

**F: Zu viel RAM verbraucht?**  
A: Nutze `--batch-size 32` um Memory zu sparen.

**F: XXL Pipeline dauert zu lange?**  
A: Starten Sie nur spezifische Phasen:
```bash
python scripts/run_xxl_pipeline.py --phases 3 4 5  # Nur Stats & Summary
```

---

## 🔗 Weiterführende Ressourcen

- **Installation & Setup:** [QUICK_START.md](QUICK_START.md)
- **Uncertainty Quantification:** [UQ_VALIDATION.md](UQ_VALIDATION.md)
- **Sampler-Vergleich:** [SAMPLERS.md](SAMPLERS.md)
- **Methodologie:** [../02_THEORY/methodology.md](../02_THEORY/methodology.md)

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

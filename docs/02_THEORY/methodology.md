# 🔬 METHODOLOGIE: Parameter-Optimierung für KDR100-Datenselektion

**Dokument:** `methodology.md`  
**Erstellt:** 15. Januar 2026  
**Anwendungsbereich:** Thesis-Kapitel "Methodik der Datenselektion"

---

## 📖 Überblick: Der 4-Phasen-Hybrid-Ansatz

Die Parameteroptimierung erfolgt in **4 aufeinanderfolgenden Phasen**, die unterschiedliche wissenschaftliche Ziele verfolgen:

```
┌─────────────────────────────────────────────────────────────────┐
│         PHASE 1: EXPLORATION (LHS)                              │
│    Verstehe den Parameterraum & Trade-offs                      │
│    Methode: Latin Hypercube Sampling (50 Kombinationen)         │
│    Ziel: Pareto-Front für Thesis-Plots visualisieren            │
│    Output: pareto_solutions.csv, Visualisierungen               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│      PHASE 2/3: OPTIMIZATION (Optuna Autoscale)                 │
│    Finde das mathematische Optimum                              │
│    Methode: Bayesian Optimization mit TPE + Adaptive Bounds     │
│    Ziel: Best Parameters mit Konvergenz-Garantie                │
│    Output: optuna_autoscale_best_YYYYMMDD.json                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│       PHASE 4: VALIDATION (Bootstrap & Sensitivität)            │
│    Beweise Robustheit der gefundenen Parameter                  │
│    Methode: 200 Resamples × 5 Distanzen × 5 Seeds              │
│    Ziel: Confidence Intervals & Stability Analysis              │
│    Output: validation_results.csv, Stability Reports            │
└─────────────────────────────────────────────────────────────────┘
```

(Weitere Inhalte übernommen; siehe Originaldokument für vollständige Methodik und Formeln.)

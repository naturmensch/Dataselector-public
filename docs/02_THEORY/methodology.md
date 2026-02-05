# 🔬 METHODOLOGIE: Parameter-Optimierung für KDR100-Datenselektion

**Dokument:** `docs/02_THEORY/methodology.md`  
**Erstellt:** 15. Januar 2026  
**Status:** Production Ready (Post Phase-4 Migration)  
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

---

## 🎯 PHASE 1: EXPLORATION (LHS-Sweep)

### Wissenschaftliche Begründung

**Warum nicht einfach ein manuelles Grid?**

Manuelle Grids haben zwei Probleme:
1. **Willkürlichkeit:** Wie wählt man die Gridpunkte? (3×3 vs. 5×5?)
2. **Lücken:** Der Parameterraum bleibt ungleichmäßig abgedeckt

**Die Lösung: Latin Hypercube Sampling (LHS)**

LHS ist eine quasi-Monte-Carlo-Methode, die:
- ✅ Den Parameterraum **garantiert lückenlos abdeckt**
- ✅ **Gleichmäßig** in jeder Dimension stratifiziert
- ✅ **Deterministisch** mit Seed reproduzierbar
- ✅ Besser als zufälliges Sampling (höhere Effizienz)

### Konfiguration & Output

**Parameter:**
| Parameter | Wert | Begründung |
|-----------|------|-----------|
| **n_samples** | 50 | Balance: Breite Abdeckung ohne zu viele Läufe (~2 Std.) |
| **seed** | 42 | Reproduzierbarkeit & Konsistenz |
| **min_distance_km** | 28.0 | Datensatz-Median |

**Output:** `outputs/tuning_weights/pareto/pareto_solutions.csv`

---

## 🚀 PHASE 2/3: OPTIMIZATION (Optuna Autoscale)

### Wissenschaftliche Begründung

Nach Phase 1 nutzen wir **Bayesian Optimization (TPE)** intelligent:

- ❌ **Grid Search:** 2700+ Läufe (ineffizient)
- ✅ **Bayesian Optimization:** Konzentriert sich auf vielversprechende Regionen

### Adaptive Bounds & Staged Optimization

```
Stage 1: 50 Trials
├─ Finde grobe Richtung
└─ Reduziere Suchraum um 20%

Stage 2-4: Progressive Refinement
└─ Konvergiere zu Optimum
```

**Output:** `outputs/optuna_autoscale_best_YYYYMMDD.json`

---

## ✅ PHASE 4: VALIDATION (Bootstrap & Sensitivität)

### Bootstrap Resampling

```
Original: 673 Kartenkacheln
├─ Resample WITH REPLACEMENT: 200 Mal
├─ Führe Selection aus
└─ OUTPUT: Confidence Intervals (95% CI)
```

**Output:** `outputs/validation_results.csv`

---

## 🔄 Reproduzierbarkeit

### Seed Management

| Komponente | Seed |
|------------|------|
| LHS | 42 |
| Optuna | 42 |
| K-Means | 42 |
| Bootstrap | [42-46] |

---

## 🚀 Execution

```bash
# Volle Pipeline
python scripts/run_thesis_pipeline.py

# Einzelne Phasen
python scripts/tune_weights_and_run.py --n-samples 50  # Phase 1
python scripts/optuna_autoscale.py --stages 50 100 300 full  # Phase 2/3
python scripts/validate_pareto_candidates.py --seeds 42 43 44 45 46  # Phase 4
```

---

## 📚 Referenzen

- **LHS:** McKay et al. (1979)
- **TPE/Optuna:** Bergstra et al. (2011)
- **Bootstrap:** Efron & Tibshirani (1993)

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

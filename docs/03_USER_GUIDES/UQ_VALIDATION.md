# 🔬 UNCERTAINTY QUANTIFICATION & VALIDATION

**Dokument:** `docs/03_USER_GUIDES/UQ_VALIDATION.md`  
**Zielgruppe:** Thesis-Autoren, Forscher  
**Status:** Production Ready

---

## 📊 Bootstrap Uncertainty Quantification

### Was ist Bootstrap?

**Bootstrap** ist eine nicht-parametrische Methode zur Schätzung von Konfidenzintervallen:

```
Originaldaten (673 Kartenkacheln)
    ↓
Resample WITH REPLACEMENT (200 Mal)
    ↓
Für jeden Resample: Führe Datenselektion aus
    ↓
Sammle 200 Ergebnisse
    ↓
Berechne 95% Confidence Interval [Q2.5, Q97.5]
```

### Bootstrap-Konfiguration

| Parameter | Default | Bedeutung |
|-----------|---------|-----------|
| `--n-boot` | 200 | Anzahl Resamples |
| `--boot-seeds` | [42-46] | Random Seeds für Reproduzierbarkeit |
| `--ci-level` | 0.95 | Confidence Level (95%) |

### Ausführung

```bash
# Phase 4: Bootstrap Validierung aus Thesis Pipeline
python scripts/run_thesis_pipeline.py --n-boot 200

# Oder standalone
python scripts/validate_pareto_candidates.py \
  --n-boot 200 \
  --seeds 42 43 44 45 46 \
  --input outputs/optuna_autoscale_best_*.json
```

### Output Interpretation

**File:** `outputs/validation_results.csv`

```
Metric,Value,CI_Lower,CI_Upper,Interpretation
Coverage_Mean,67.3,66.1,68.5,67.3% der Kacheln selektiert (95% CI: 66.1–68.5%)
Diversity_Visual,0.92,0.90,0.93,Hohe visuelle Diversität (stabil)
Diversity_Spatial,0.88,0.86,0.90,Gute räumliche Streuung
Diversity_Temporal,0.85,0.83,0.87,Moderater zeitlicher Abstand
Bootstrap_Variance,0.043,0.041,0.045,Niedrig → Robuste Parameter
Seed_Consistency,0.97,0.96,0.98,Sehr hoch → Reproducible
```

**Interpretation:**
- ✅ **Enge CI:** Parameter sind stabil & reproducible
- ⚠️ **Breite CI:** Ergebnisse sensitiv gegen Randomness
- ✅ **Hohe Seed-Konsistenz:** System ist deterministisch

---

## 🧪 Sampler Vergleich (QMC vs TPE vs CMA-ES)

### Warum 3 Sampler?

| Sampler | Stärke | Schwäche | Einsatz |
|---------|--------|---------|---------|
| **QMC (Sobol)** | Deterministisch, reproducible | Weniger adaptiv | Thesis (Default) |
| **TPE** | Bayesian, beste empirische Ergebnisse | Stochastisch | Produktiv |
| **CMA-ES** | Robust bei Constraints | Evolutionär, langsamere Konvergenz | Fallback |

### Sampler-Suite Experiment

```bash
# Vergleiche alle 3 Sampler über 10 Seeds
python scripts/run_thesis_sampler_suite.py \
  --n-seeds 10 \
  --n-trials 1000 \
  --output outputs/sampler_comparison.csv
```

**Output:** `outputs/selected_sampler.json`

```json
{
  "recommended_sampler": "tpe",
  "reason": "Best fitness (+0.44% vs QMC baseline)",
  "convergence_speed": {
    "qmc": 750,
    "tpe": 620,
    "cmaes": 890
  },
  "stability_ranking": ["tpe", "qmc", "cmaes"],
  "seed_consistency": {
    "qmc": 0.99,
    "tpe": 0.96,
    "cmaes": 0.93
  }
}
```

### Interpretation

| Metrik | Zielwert | Bedeutung |
|--------|----------|-----------|
| **Fitness** | Höher ist besser | Coverage & Diversität |
| **Konvergenz-Trials** | Niedrig ist besser | Wie schnell findet Optuna Optimum |
| **Seed-Konsistenz** | >0.9 | Wie reproducible ist das Ergebnis? |

---

## 🔄 Reproducibility-Validierung

### Multi-Seed Konsistenz-Test

```bash
# Phase 2: Reproducibility Tests (XXL Pipeline)
python scripts/run_xxl_pipeline.py --phases 2 \
  --seeds 42 43 44 45 46
```

**Was wird getestet:**
1. Verschiedene Random Seeds
2. Identische Inputdaten
3. Erwartung: Identische Outputs (deterministisch)

**Output:** `outputs/xxl/phase2_reproducibility/`

```
results_seed_42.csv
results_seed_43.csv
results_seed_44.csv
results_seed_45.csv
results_seed_46.csv
consistency_report.md
```

**Konsistenz-Report:**
```markdown
# Reproducibility Validation Report

## Seed-Konsistenz
| Seed | Coverage | CI_Width | Status |
|------|----------|----------|--------|
| 42   | 67.2%    | 2.4%     | ✓ |
| 43   | 67.1%    | 2.3%     | ✓ |
| 44   | 67.3%    | 2.5%     | ✓ |
| 45   | 67.0%    | 2.4%     | ✓ |
| 46   | 67.2%    | 2.3%     | ✓ |

**Avg Std Dev:** 0.1% → HIGHLY REPRODUCIBLE ✓
```

---

## 📈 Sensitivitätsanalyse

### Parameter-Sensitivität

```bash
# Teste wie sensitiv das System auf Parameter-Variationen ist
python scripts/sensitivity_analysis.py \
  --base-config config/pipeline_config.yaml \
  --alpha-range 0.2:0.6:0.05 \
  --beta-range 0.2:0.6:0.05 \
  --gamma-range 0.2:0.6:0.05
```

**Output:** Wärmekarte von Coverage vs. (α, β, γ)

```
Coverage Sensitivity Map:
        β=0.2  β=0.3  β=0.4  β=0.5  β=0.6
α=0.2   62%    64%    65%    65%    64%
α=0.3   65%    67%    68%    68%    67%
α=0.4   67%    69%    70%    69%    68%
α=0.5   66%    68%    69%    68%    67%
α=0.6   64%    66%    67%    66%    65%

Peak: α=0.4, β=0.4 → 70% Coverage
Sensitivity: Moderat (±3% um Optimum)
```

---

## ✅ Validierungs-Checkliste für Thesis

```
□ Bootstrap 200 Resamples mit 5 Seeds durchführen
□ 95% CI Width < 3% (Stabilität)
□ Seed-Konsistenz > 0.95 (Reproducibility)
□ Sampler-Vergleich zeigt TPE oder QMC empfohlen
□ Reproducibility-Tests alle Phasen 2 bestanden
□ Sensitivitätsanalyse zeigt moderate Abhängigkeit
□ Alle Reports in outputs/ vorhanden
□ Confidence Intervals in Thesis-Kapitel dokumentiert
```

---

## 🚨 Häufige Probleme & Lösungen

**F: "Bootstrap variance sehr hoch (>0.1)"?**  
A: Parameter könnten suboptimal sein. Führe Phase 2/3 (Optimization) neu aus.

**F: "Seed-Konsistenz niedrig (<0.9)"?**  
A: Möglicherweise numerische Instabilität. Prüfe Feature Extractor Version (DINOv2 vs ResNet50).

**F: "TPE vs QMC stark unterschiedlich"?**  
A: Normal! TPE ist stochastisch. Nutze `--sampler qmc` für reproduzierbare Thesis.

---

## 🔗 Weiterführende Ressourcen

- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Pipelines:** [PIPELINES.md](PIPELINES.md)
- **Bootstrap Mathematik:** [../02_THEORY/methodology.md](../02_THEORY/methodology.md)
- **Statistik Hintergrund:** [../SCIENTIFIC_BACKGROUND.md](../SCIENTIFIC_BACKGROUND.md)

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

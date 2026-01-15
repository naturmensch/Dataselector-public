# Deprecated Scripts

Diese Scripts sind veraltet und durch die neue harmonisierte Pipeline mit **Latin Hypercube Sampling (LHS)** ersetzt worden. Sie werden für historische Referenzen aufbewahrt.

## Archivierte Dateien

### `optimize_realistic_params.py`
**Deprecated am:** 15. Januar 2026
**Ersetzt durch:** `scripts/run_adaptive_pipeline.py` (nutzt LHS statt manuelles Grid)
**Grund:** 
- Verwendet veraltete manuelle Parameter-Kombinationen (9×3 Grid)
- Nicht wissenschaftlich fundiert (Lücken in Coverage)
- Keine adaptiven Bounds
- Ersetzt durch LHS: `max(27, √n_tiles)` adaptive Samples

**Original-Zweck:** Optimierter Parameter-Test mit n_samples, gamma_temporal und min_distance_km Variation

---

### `test_temporal_sensitivity.py`
**Deprecated am:** 15. Januar 2026
**Ersetzt durch:** `scripts/run_fine_sweep.py` mit multi-criteria Gewichten aus config
**Grund:**
- Testet isoliert nur temporal_weight
- Nicht Teil der harmonisierten Pipeline
- Config-Werte (α=0.40, β=0.30, γ=0.30) bereits durch Optuna validiert

**Original-Zweck:** Temporal Weight Sensitivity Test mit constraint-integrierter Methode

---

### `performance_test_full_selection.py`
**Deprecated am:** 15. Januar 2026
**Ersetzt durch:** `scripts/final_selection.py` mit config-getriebenen Parametern
**Grund:**
- Hardcoded Parameter (alpha=0.60, beta=0.15, gamma=0.25) - veraltet
- min_distance_km=0.0 entspricht nicht der validierten Strategie (40km optimal)
- Kein Bootstrap, keine Robustheit-Analyse

**Original-Zweck:** Multi-Criteria Performance-Test für Full Dataset (n_samples=673)

---

### `run_coarse_sweep.py` ⚠️ **LEGACY**
**Status:** Noch aktiv, aber **nicht empfohlen** für neue Experimente
**Ersetzt durch:** `scripts/tune_weights_and_run.py` (nutzt LHS statt manuelles Grid)
**Grund:**
- Manuelles 9×3=27 Grid (hardcoded) statt adaptive LHS
- Lücken in Parameter-Coverage (nicht gleichmäßig verteilt)
- Nicht skalierbar mit Datensatzgröße

**Migration:**
```bash
# ALT (Legacy Grid):
python scripts/run_coarse_sweep.py

# NEU (Adaptive LHS):
python scripts/tune_weights_and_run.py --n-lhs 27  # oder adaptiv: max(27, √n_tiles)
```

---

## Migration Guide

**Statt `optimize_realistic_params.py` oder `run_coarse_sweep.py`:**
```bash
# MODERN: LHS-basierte Exploration (adaptiv, gleichmäßig)
python scripts/run_adaptive_pipeline.py --yes

# Nur Exploration (ohne Fine/Optuna):
python scripts/tune_weights_and_run.py --n-lhs 27
```

**Statt `test_temporal_sensitivity.py`:**
```bash
# Multi-Criteria Gewichte aus config (bereits optimiert: α=0.40, β=0.30, γ=0.30)
python scripts/run_fine_sweep.py
```

**Statt `performance_test_full_selection.py`:**
```bash
# Full-Dataset Selection mit validiertem min_distance und Bootstrap
python scripts/final_selection.py --n-samples 673 --min-distance-km 40.0
python scripts/bootstrap_pareto_candidates.py --pareto outputs/fine_sweep/pareto_solutions.csv --n-boot 200
```

---

## Warum LHS statt manuelles Grid?

| Methode | Coverage | Wissenschaftlich | Adaptiv | Empfohlen |
|---------|----------|------------------|---------|-----------|
| **LHS** (Latin Hypercube) | ✅ Gleichmäßig | ✅ Ja (Quasi-Monte-Carlo) | ✅ `max(27, √n)` | ✅ **JA** |
| **Manual Grid** (Coarse) | ⚠️ Lücken | ❌ Ad-hoc | ❌ Fest: 27 | ❌ **Legacy** |

**Adaptive Pipeline (empfohlen):**
```bash
# Komplette Cascade: LHS → Fine → Optuna → Bootstrap
python scripts/run_adaptive_pipeline.py --yes
```

---

## Hinweise
- Diese Scripts werden NICHT mehr gewartet
- Für neue Experimente immer die **LHS-basierte** adaptive Pipeline verwenden
- Bei Fragen zur Migration siehe `docs/REPRODUCIBILITY.md` und `docs/Pipeline_260115.md`

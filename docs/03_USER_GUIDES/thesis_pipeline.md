# Historical/Legacy (Not Normative)

This page is retained for historical context and still contains script-era
examples.

Use the authoritative replacement:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

# 🚀 Thesis Pipeline: Quick Start Guide

Schnelleinstieg für die **4-Phasen Parameter-Optimierung**:

## Installation

```bash
# Install dependencies (including scipy for LHS)
pip install -r requirements.txt
```

## Volle Pipeline ausführen

```bash
./scripts/exec_in_env.sh --env dataselector -- python scripts/run_thesis_pipeline.py
```

Dies führt automatisch alle 4 Phasen aus (~10 Minuten Gesamtdauer):

| Phase | Methode | Output | Zeit |
|-------|---------|--------|------|
| **1. Exploration** | LHS (adaptiv: ~50 Samples*) | Trade-off Plots | ~2 min |
| **2/3. Optimization** | Bayesian (Optuna) | Best Parameters | ~5 min |
| **4. Validation** | Bootstrap (200×5×5) | Confidence Intervals | ~3 min |

*n_lhs wird automatisch berechnet: `max(50, 2×√n_tiles)` für optimale Parameterraum-Abdeckung

(Weitere Implementation- und Fehlersuch-Hinweise siehe Originaldokument.)

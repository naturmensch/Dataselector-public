# 🔧 ADVANCED TUNING & TROUBLESHOOTING

**Dokument:** `docs/05_ADVANCED/ADVANCED_TUNING.md`  
**Zielgruppe:** Data Scientists, Optimierer, Advanced Users  
**Status:** Production Ready

---

## 🎯 Optuna Deep Dive: Sampler & Parameter Tuning

### Sampler-Vergleich: TPE vs QMC vs CMA-ES

| Sampler | Konvergenz | Stabilität | Reproducibility | Use Case |
|---------|-----------|-----------|-----------------|----------|
| **TPE** | Schnell (600 trials) | Moderat | Stochastisch (Seed-variabel) | Production |
| **QMC (Sobol)** | Moderat (750 trials) | Höher | Vollständig reproducible | Thesis |
| **CMA-ES** | Langsam (890 trials) | Robust | Moderat | Constraints-heavy |

### Sampler Selection (CLI)

```bash
# TPE (Bayesian - empirisch am besten)
python scripts/run_thesis_pipeline.py --sampler tpe

# QMC (Quasi-Monte Carlo - reproducible)
python scripts/run_thesis_pipeline.py --sampler qmc

# CMA-ES (Evolutionary - robust)
python scripts/run_thesis_pipeline.py --sampler cmaes
```

### Staging & Adaptive Bounds

**Thesisscripte:** `scripts/optuna_autoscale.py`

```bash
# Progressive Refinement mit Staging
python scripts/optuna_autoscale.py \
  --stages 50 100 300 full \
  --sampler tpe \
  --adaptive-bounds
```

**Stagingprozess:**

```
Stage 1: 50 Trials
├─ Erkunde Parameterraum breit
└─ Berechne "vielversprechende Region"

Stage 2: 100 Trials
├─ Zoome in um Top 20% Kandidaten
└─ Reduziere Suchraum um 20%

Stage 3: 300 Trials
├─ Feinoptimierung in konvergierter Region
└─ Konvergiere zu lokalem Optimum

Stage 4: Full (adaptive)
├─ Weitere Trials bis Konvergenz
└─ Optuna stoppt automatisch (no improvement)
```

**Output:** `outputs/optuna_autoscale_best_*.json`

### Optuna Trial Inspection

```bash
# View top 10 trials
python scripts/inspect_optuna_study.py \
  --db outputs/optuna_study.db \
  --top 10 \
  --metric coverage

# Export trials to CSV
python scripts/export_optuna_trials.py \
  --db outputs/optuna_study.db \
  --output trials_export.csv

# Visualize convergence
python scripts/plot_optuna_convergence.py \
  --db outputs/optuna_study.db \
  --output convergence_plot.png
```

---

## 🧪 Scientific Validation & Methods

### Validation-Framework

Die Datenselektion wird auf drei Ebenen validiert:

#### 1. **Numerische Validierung** (Correctness)
- Feature Extractor Output-Dimensionalität
- K-Means Cluster-Anzahl
- Constraint-Enforcement (Spatial Distance)

#### 2. **Statistische Validierung** (Robustheit)
- Bootstrap Confidence Intervals
- Seed-Konsistenz über 5 Seeds
- Sensitivity Analysis (Parameterraum)

#### 3. **Wissenschaftliche Validierung** (Domain-Justification)
- Pareto-Front Trade-offs (Visuell/Räumlich/Zeitlich)
- Geographic Plausibility (Hamburg Region Test)
- Historical Map Coverage Analysis

### Validation Commands

```bash
# Run all validation phases
python scripts/validate_pipeline.py \
  --phases numeric statistical scientific

# Bootstrap validation (200 resamples)
python scripts/validate_pareto_candidates.py \
  --n-boot 200 \
  --seeds 42 43 44 45 46

# Sensitivity analysis
python scripts/sensitivity_analysis.py \
  --base-config config/pipeline_config.yaml \
  --param-ranges config/sensitivity_ranges.yaml
```

### Output Interpretation

**File:** `outputs/validation_results.csv`

```
Parameter,Value,CI_Lower,CI_Upper,Valid?
Coverage_Percent,67.3,66.1,68.5,✓
Spatial_Diversity,0.88,0.86,0.90,✓
Visual_Diversity,0.92,0.90,0.93,✓
Bootstrap_Variance,0.043,0.041,0.045,✓
```

**Akzeptanz-Kriterien:**
- ✅ Bootstrap Variance < 0.1
- ✅ Seed-Konsistenz > 0.95
- ✅ Coverage CI Width < 3%
- ✅ Spatial Constraint Enforcement 100%

---

## 🚨 Troubleshooting & FAQ

### "No convergence" Error

```bash
# Problem: Optuna findet nicht schnell genug Optimum

# Solution 1: Nutze QMC für stabilere Konvergenz
python scripts/run_thesis_pipeline.py --sampler qmc

# Solution 2: Reduziere n_trials
python scripts/run_thesis_pipeline.py \
  --n-trials 300 \
  --patience 30  # Stop wenn kein Progress

# Solution 3: Fix feature extractor issue
# Check DINOv2/ResNet50 compatibility
python -c "from dataselector.feature_extraction import DINOv2Extractor; print(DINOv2Extractor())"
```

### Memory Issues ("CUDA Out of Memory")

```bash
# Problem: GPU RAM voll (Features zu große)

# Solution 1: Reduce batch size
python scripts/run_thesis_pipeline.py \
  --batch-size 16  # from 32

# Solution 2: Use CPU only
export CUDA_VISIBLE_DEVICES=""

# Solution 3: Clear cache
rm -rf outputs/features/cache/

# Solution 4: Lower resolution
python scripts/run_thesis_pipeline.py \
  --image-resolution 448  # from 518
```

### Reproducibility Issues

```bash
# Problem: Same input, different output

# Solution 1: Check seed settings
grep seed config/pipeline_config.yaml

# Solution 2: Validate DINOv2 determinism
python scripts/test_determinism.py --feature-extractor dinov2

# Solution 3: Check version pinning
pip list | grep -E "torch|timm|optuna"

# Solution 4: Use QMC sampler
python scripts/run_thesis_pipeline.py --sampler qmc
```

### W&B Logging Failures

```bash
# Problem: wandb module missing or error

# Solution 1: Install wandb
pip install wandb

# Solution 2: Login required
wandb login  # Get API key from https://wandb.ai/authorize

# Solution 3: Offline mode
export WANDB_MODE=offline

# Solution 4: Check network
curl https://wandb.ai/api/v1/status
```

### Missing Feature Files

```bash
# Problem: "Feature cache not found"

# Solution 1: Regenerate features
python scripts/extract_features_batch.py \
  --input data/new_all_tiles.csv \
  --output outputs/features/

# Solution 2: Check data path
python -c "from dataselector.utils import get_data_path; print(get_data_path())"

# Solution 3: Use feature extraction API
from dataselector.feature_extraction import DINOv2Extractor
extractor = DINOv2Extractor()
features = extractor.extract_batch(image_paths, cache_dir="outputs/features/")
```

### Optuna DB Corruption

```bash
# Problem: "Database is locked" or integrity errors

# Solution 1: Check integrity
sqlite3 outputs/optuna_study.db "PRAGMA integrity_check;"

# Solution 2: Restore from backup
ls -lt outputs/optuna_study.db.bak_*
cp outputs/optuna_study.db.bak_<RECENT> outputs/optuna_study.db

# Solution 3: Force clean slate
rm outputs/optuna_study.db
python scripts/optuna_autoscale.py --restart none

# Solution 4: Vacuum database
sqlite3 outputs/optuna_study.db "VACUUM;"
```

---

## 🔬 Advanced Performance Tuning

### Feature Extraction Optimization

```python
from dataselector.feature_extraction import DINOv2Extractor

# Faster extraction (lower resolution)
extractor = DINOv2Extractor(
    model_name="dinov2_vits14",
    resolution=448,  # from 518 (default)
    batch_size=64    # from 32 (if GPU allows)
)

# Better quality (higher resolution)
extractor = DINOv2Extractor(
    model_name="dinov2_vitl14",  # Larger model
    resolution=518,
    batch_size=16
)
```

### Clustering Parameter Tuning

```bash
# Test different cluster counts
for n in 4 8 12 16; do
  python scripts/run_thesis_pipeline.py \
    --n-clusters $n \
    --output outputs/test_n_clusters_${n}/
done

# Evaluate silhouette scores
python scripts/evaluate_clustering.py \
  --output-dirs outputs/test_n_clusters_*/
```

### Spatial Constraint Relaxation

```bash
# Progressive relaxation wenn hard constraint too strict
python scripts/run_thesis_pipeline.py \
  --min-distance-km 40 \  # Start
  --fallback-relax 0.2    # Relax 20% if no solution

# Manual adjustment
python scripts/run_thesis_pipeline.py \
  --min-distance-km 30    # Reduce from 40
```

---

## 📊 Debugging Workflow

### Verbose Logging

```bash
# Enable debug logging
export DEBUG=1
python scripts/run_thesis_pipeline.py --verbose

# or with structured logging
python scripts/run_thesis_pipeline.py \
  --log-level DEBUG \
  --log-file outputs/debug.log
```

### Interactive Debugger

```python
# Insert breakpoint in script
python -m pdb scripts/run_thesis_pipeline.py

# Common commands:
# (Pdb) n           # next line
# (Pdb) s           # step into
# (Pdb) c           # continue
# (Pdb) p variable  # print variable
```

### Profiling

```bash
# Profile CPU/Memory
python -m cProfile -s cumtime scripts/run_thesis_pipeline.py 2>&1 | head -20

# GPU Memory Profile
python -c "
import torch
from dataselector.feature_extraction import DINOv2Extractor
torch.cuda.reset_peak_memory_stats()
extractor = DINOv2Extractor()
print(f'GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
"
```

---

## 📚 Weiterführende Ressourcen

- **UQ & Validation:** [../03_USER_GUIDES/UQ_VALIDATION.md](../03_USER_GUIDES/UQ_VALIDATION.md)
- **Monitoring:** [MONITORING_OPS.md](MONITORING_OPS.md)
- **Optuna Docs:** https://optuna.readthedocs.io/
- **DINOv2 Paper:** https://arxiv.org/abs/2304.07193

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

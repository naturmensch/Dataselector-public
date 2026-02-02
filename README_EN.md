# KDR100 Data Selection

**Algorithmic data selection for the "Karte des Deutschen Reiches" (KDR100) using Optuna-based hyperparameter optimization and deep-learning features.**

Scientifically rigorous method to objectively select training examples from 676 heterogeneous map tiles.

---

## 🚀 Quick Start

**For the full production pipeline (Sampler Suite + XXL Run + Bootstrap):**

**Option A: One-shot automation (RECOMMENDED):**

```bash
# Activate environment (see Installation below)
conda activate dataselector

# One-shot: orchestrates EVERYTHING automatically (Sampler Suite + XXL Phases 0-5)
bash scripts/run_complete_thesis_pipeline.sh

# Or with a custom environment:
bash scripts/run_complete_thesis_pipeline.sh --env my-env

# Or with an explicit sampler override (optional):
bash scripts/run_complete_thesis_pipeline.sh --sampler cmaes  # cmaes, qmc, or tpe
```

**Option B: Manual steps (if you want to inspect intermediate results):**

```bash
conda activate dataselector

# STEP 1: Thesis Sampler Suite (auto: 10 seeds, 1000 trials, Hamburg + KDR100)
python scripts/run_thesis_sampler_suite.py

# STEP 2: XXL Pipeline with integrated Bootstrap (Phases 0-5)
# Phase 0: Convergence analysis (auto-detects best sampler from Suite)
# Phase 1-4: Optimization (Hamburg + Reproducibility + Statistics + Summary)
# Phase 5: Bootstrap UQ (500 resamples)
python scripts/xxl_KDR146_run_thesis_complete.py
# Or override sampler explicitly:
python scripts/xxl_KDR146_run_thesis_complete.py --optuna-sampler cmaes  # qmc, tpe, or cmaes
```

### ✨ Current status (2026-01-23)

- ✅ **Sampler argument fix:** `--optuna-sampler` now propagates correctly through all phase functions
- ✅ **Environment:** NumPy pinned to 2.3.x (Numba-compatible); PyTorch CPU available
- ✅ **XXL Pipeline:** Phases 0–5 fully orchestrated
- ✅ **Dry-run:** Phase 1–2 validated successfully with QMC sampler

**Estimated durations:**
- Sampler Suite: 8–12 hours (10 seeds × 3 samplers × 1000 trials)
- XXL Pipeline + Bootstrap: 3.5–6 hours
- **Total:** ~12–18 hours

**Two-command, thesis‑ready workflow:**
- Sampler Suite (10 seeds, 1000 trials per sampler) → `outputs/selected_sampler.json`
- XXL Pipeline (Phase 0–5) with Bootstrap UQ → `outputs/runs/...`

---

## 📋 Overview

### Core features

- ✅ **Multi-seed sampler comparison:** QMC vs TPE vs CMA-ES (10 seeds, 1000 trials per sampler)
- ✅ **XXL Pipeline:** 5-phase optimization (Convergence → Hamburg → Reproducibility → Statistics → Summary + Bootstrap)
- ✅ **Convergence analysis:** automatic computation of suggested trial counts (Phase 0)
- ✅ **Reproducibility:** seed-based validation on Hamburg and KDR100 datasets
- ✅ **Uncertainty Quantification:** Bootstrap resampling with confidence intervals
- ✅ **Scientific outputs:** automatic reports, plots, and stability metrics (Jaccard index)

### Tech stack

| Component | Tools |
|---|---|
| Language | Python 3.11 |
| GPU / DL | PyTorch, torchvision (DINOv2 / ResNet50) |
| Optimization | Optuna (QMC/TPE/CMA-ES), apricot-select |
| Data | pandas, numpy, geopandas |
| Dimensionality reduction / clustering | UMAP, scikit-learn |
| Geospatial | geopandas, pyproj, shapely |
| Environment / testing | mamba/conda, pytest |

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/username/Dataselector.git
cd Dataselector
```

### 2. Create environment (recommended: mamba/conda)

```bash
# Fast (mamba recommended)
mamba env create -f environment.yml -n dataselector
conda activate dataselector

# Or use helper script
./scripts/create_env.sh dataselector 3.11
```

### 3. Install dependencies

```bash
# CPU-only setup
pip install -r requirements-cpu.txt

# Or with CUDA/GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest -v  # full test suite
pytest --lf  # only last failures
```

**Known compatibility note:** NumPy / numba: numba requires NumPy <= 2.3.x. `environment.yml` pins NumPy accordingly.

---

## 🎯 Workflows

### A. Production XXL Run (RECOMMENDED)

Full scientific workflow with automatic sampler selection.

#### Step 1: Thesis Sampler Suite (optional when run separately)

```bash
# 10 seeds, 1000 trials per sampler, datasets: hamburg + kdr100
python scripts/run_thesis_sampler_suite.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --n-trials 1000 \
  --datasets hamburg kdr100 \
  --samplers qmc tpe cmaes
```

**Output:** `outputs/selected_sampler.json` + plots + CSV reports

**Estimated duration:** 8–12 hours (1000 trials)

#### Step 2: XXL Pipeline with auto parameters (recommended)

```bash
python scripts/xxl_KDR146_run_thesis_complete.py
```

**Phases (0–5) summary:**

- Phase 0: Pre-flight convergence validation → computes `n_trials` from cached convergence baseline
- Phase 1: Hamburg run (e.g., 440 trials) to find best hyperparameters
- Phase 2: Reproducibility (2 seeds) validation
- Phase 3: Statistics aggregation
- Phase 4: Thesis-ready summary/report generation
- Phase 5: Bootstrap UQ (e.g., 500 resamples)

**Automatic parameter computation:** Phase 0 determines `n_trials = 5 × convergence_baseline` and auto-detects the best sampler via `outputs/selected_sampler.json` unless overridden by `--optuna-sampler`.

---

### B. XXL Run with Monitor (recommended)

```bash
# Run pre-hook sampler suite and the full XXL orchestrator with logging and monitoring
python scripts/xxl_full_run_monitor.py

# Dry-run mode (plan only):
python scripts/xxl_full_run_monitor.py --child-dry-run
```

Monitor features:
- Pre- and Post- Hooks
- Timestamped logs
- Trials.csv reconstruction if interrupted
- Monitor report when run completes

---

### C. Adaptive / Quick tests

```bash
# Quick smoke test with adaptive pipeline
python scripts/run_adaptive_pipeline.py --yes \
  --n-lhs 5 \
  --n-trials 10 \
  --n-boot 5 \
  --skip-optuna
```

**Use cases:** smoke tests, development, parameter tuning

---

### D. Useful single scripts

- `scripts/compare_samplers_multi_seed.py` — fast multi-seed sampler comparison
- `scripts/tune_weights_and_run.py` — Phase 1: LHS / Sobol weight sweep + Pareto front computation
- `scripts/run_fine_sweep.py` — Phase 2: fine sweep around Pareto region
- `scripts/optuna_optimize.py` — low-level Optuna runner called by higher-level scripts
- `scripts/bootstrap_pareto_candidates.py` — Phase 5: bootstrap UQ on Pareto selections
- `scripts/generate_experiment_report.py` — per-run report generation (Markdown + plots)
- `scripts/exec_in_env.sh` — helper to run commands inside conda/mamba environment

---

## 📂 Project layout

```
Dataselector/
├── src/                                  # Python package
│   ├── io.py                            # data loading / feature extraction
│   ├── metrics.py                       # metric computations
│   ├── metadata_processor.py            # CSV/DBF processing
│   ├── clustering.py                    # UMAP + K-means
│   ├── diversity_selector.py            # facility-location selection logic
│   ├── experiment_manager.py            # run/version management
│   └── visualizer.py                    # plotting helpers
├── scripts/
│   ├── run_thesis_sampler_suite.py     # Sampler suite orchestrator (thesis-grade)
│   ├── xxl_KDR146_run_thesis_complete.py # XXL orchestrator (Phase 0-5)
│   ├── xxl_full_run_monitor.py         # Monitor + pre/post hooks + logging
│   ├── run_adaptive_pipeline.py        # Adaptive pipeline (LHS-based)
│   ├── compare_samplers_multi_seed.py  # Multi-seed sampler comparer
│   ├── optuna_optimize.py              # low-level Optuna experiment runner
│   └── [other helper scripts...]
├── tests/                                # pytest suite
├── notebooks/
│   └── 01_data_exploration.ipynb
├── config/
│   └── pipeline_config.yaml
├── data/
│   ├── new_all_tiles.csv               # 676 KDR100 tiles metadata
│   └── images/
├── outputs/
│   ├── runs/
│   └── selected_sampler.json
└── environment.yml
```

---

## ⚙️ Configuration

Main config: `config/pipeline_config.yaml`

```yaml
selection:
  n_samples: 34                # approximate target selection size (~5%)
  min_distance_km: 50.0        # spatial minimum distance constraint

optimization:
  n_trials: 440                # default optuna trials
  n_candidates: 676            # all KDR100 tiles (100%)
```

---

If you'd like, I can also update the repository README to replace the German file or add a note linking to this English version.
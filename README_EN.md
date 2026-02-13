# KDR100 Data Selection

**Algorithmic data selection for the "Karte des Deutschen Reiches" (KDR100) using Optuna-based hyperparameter optimization and deep-learning features.**

Scientifically rigorous method to objectively select training examples from 676 heterogeneous map tiles.

---

## 🚀 Quick Start

**For the full production pipeline (Sampler Suite + XXL Run + Bootstrap):**

Canonical invocation rule: use `micromamba run -n dataselector <command>`
for all local runs and checks.

If examples below omit the prefix, run them inside an activated `dataselector`
environment or prepend the canonical micromamba invocation.

**Option A: CLI orchestration (RECOMMENDED):**

```bash
# STEP 1: (optional) Thesis Sampler Suite — benchmark samplers (writes outputs/selected_sampler.json)
micromamba run -n dataselector python -m dataselector thesis-sampler-suite --autoscale

# STEP 2: Canonical thesis orchestration (production path — uses pinned sampler and policy defaults)
micromamba run -n dataselector python -m dataselector thesis-orchestrate

> Tip: when invoking the `dataselector` env, avoid inserting an extra `--` between `micromamba run -n dataselector` and the command. Use `micromamba run -n dataselector python -m dataselector <command>` (example above). Some shells/micromamba versions treat `--` differently and it can break argument parsing.

# Optional: run thesis-pipeline directly from a validated snapshot
micromamba run -n dataselector python -m dataselector thesis-pipeline --use-params outputs/runs/<run_id>/final_config.yaml
```

**Option B: Manual steps (if you want to inspect intermediate results):**

```bash
# STEP 1: Thesis Sampler Suite (auto: 10 seeds, 1000 trials, Hamburg + KDR100)
micromamba run -n dataselector python -m dataselector thesis-sampler-suite

# STEP 2: XXL Pipeline with integrated Bootstrap (Phases 0-5)
# Phase 0: Convergence analysis (auto-detects best sampler from Suite)
# Phase 1-4: Optimization (Hamburg + Reproducibility + Statistics + Summary)
# Phase 5: Bootstrap UQ (500 resamples)
micromamba run -n dataselector python -m dataselector xxl
# Or override sampler explicitly:
micromamba run -n dataselector python -m dataselector xxl --best-sampler cmaes  # qmc, tpe, or cmaes
```

### ✨ Current status (2026-01-23)

- ✅ **Sampler argument fix:** `--best-sampler` now propagates correctly through orchestration
- ✅ **Environment:** NumPy pinned to 2.3.x (Numba-compatible); PyTorch CPU available
- ✅ **XXL Pipeline:** Phases 0–5 fully orchestrated
- ✅ **Dry-run:** Phase 1–2 validated successfully with QMC sampler

**Estimated durations:**
- Sampler Suite: 8–12 hours (10 seeds × 3 samplers × 1000 trials)
- XXL Pipeline + Bootstrap: 3.5–6 hours
- **Total:** ~12–18 hours

**Two-command, thesis‑ready workflow:**
- Sampler Suite (10 seeds, 1000 trials per sampler) → `outputs/selected_sampler.json`
- Thesis orchestration (`thesis-orchestrate` / `thesis-pipeline`) with production defaults (n_trials = 370, sampler = tpe) → `outputs/runs/...`

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
| Environment / testing | micromamba (canonical) + exec_in_env (compatibility), pytest |

---

## 📦 Installation

### 1. Clone repository

```bash
git clone https://github.com/username/Dataselector.git
cd Dataselector
```

### 2. Create environment (recommended: micromamba)

```bash
# Canonical runtime path
micromamba create -n dataselector -f environment.yml -y
micromamba run -n dataselector python -V

# Optional compatibility wrapper (delegates to micromamba/conda)
./scripts/exec_in_env.sh --env dataselector --create --yes -- python -V
```

### 3. Install dependencies

```bash
# CPU-only setup
micromamba run -n dataselector pip install -r requirements-cpu.txt

# Or with CUDA/GPU support
micromamba run -n dataselector pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
micromamba run -n dataselector pip install -r requirements.txt
```

### 4. Run tests

```bash
micromamba run -n dataselector pytest -v  # full test suite
micromamba run -n dataselector pytest --lf  # only last failures
```

**Known compatibility note:** NumPy / numba: numba requires NumPy <= 2.3.x. `environment.yml` pins NumPy accordingly.

---

## 🎯 Workflows

### A. Production XXL Run (RECOMMENDED)

Full scientific workflow with automatic sampler selection.

#### Step 1: Thesis Sampler Suite (optional when run separately)

```bash
# 10 seeds, 1000 trials per sampler, datasets: hamburg + kdr100
python -m dataselector thesis-sampler-suite \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --n-trials 1000 \
  --datasets hamburg kdr100 \
  --samplers qmc tpe cmaes
```

**Output:** `outputs/selected_sampler.json` + plots + CSV reports

**Estimated duration:** 8–12 hours (1000 trials)

#### Step 2: XXL Pipeline with auto parameters (recommended)

```bash
python -m dataselector xxl
```

**Phases (0–5) summary:**

- Phase 0: Pre-flight convergence validation → computes `n_trials` from cached convergence baseline
- Phase 1: Hamburg run (e.g., 440 trials) to find best hyperparameters
- Phase 2: Reproducibility (2 seeds) validation
- Phase 3: Statistics aggregation
- Phase 4: Thesis-ready summary/report generation
- Phase 5: Bootstrap UQ (e.g., 500 resamples)

**Automatic parameter computation:** Phase 0 determines `n_trials = 5 × convergence_baseline` and auto-detects the best sampler via `outputs/selected_sampler.json` unless overridden by `--best-sampler`.

---

### B. XXL Run with Monitor (recommended)

```bash
# Run full XXL orchestration
python -m dataselector xxl

# Smoke mode (reduced settings):
python -m dataselector xxl --smoke

# Generate monitor report from latest run artifacts
python -m dataselector generate-monitor
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
python -m dataselector adaptive-pipeline \
  --n-lhs 5 \
  --n-trials 10 \
  --n-boot 5 \
  --skip-optuna
```

**Use cases:** smoke tests, development, parameter tuning

---

### D. Useful CLI workflows

- `python -m dataselector compare-samplers` — fast multi-seed sampler comparison
- `python -m dataselector thesis-pipeline` — complete 4-phase thesis optimization
- `python -m dataselector autoscale` — staged Optuna autoscaling
- `python -m dataselector optuna-optimize` — low-level Optuna experiment runner
- `python -m dataselector bootstrap-pareto --pareto-csv <path>` — bootstrap UQ on Pareto selections
- `python -m dataselector generate-experiment --run-dir <path>` — per-run report generation
- `python -m dataselector check-env` — environment and legacy-reference validation

---

## 📂 Project layout

```
Dataselector/
├── dataselector/                         # Canonical Python package
│   ├── cli.py                           # Unified CLI entry point
│   ├── data/                            # data loading / metadata build
│   ├── features/                        # feature extraction
│   ├── selection/                       # clustering + selection logic
│   ├── pipeline/                        # run/version management helpers
│   └── workflows/                       # canonical workflows
├── dataselector/workflows/              # Canonical workflow implementations
│   ├── thesis_sampler_suite.py         # Sampler suite orchestrator (thesis-grade)
│   ├── xxl.py                          # XXL orchestrator (Phase 0-5)
│   ├── adaptive_pipeline.py            # Adaptive pipeline (LHS-based)
│   ├── compare_samplers.py             # Multi-seed sampler comparer
│   ├── optuna_optimize.py              # Low-level Optuna experiment runner
│   └── [other workflow modules...]
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
  n_samples: 24                # current thesis policy baseline
  min_distance_km: 28.5        # operational policy (geometric reference documented separately)

optimization:
  # Policy default for thesis/adaptive workflows: n_trials = 370 (see docs/PARAMETER_POLICY_LEDGER.md)
  n_trials: 370
  n_candidates: 676            # all KDR100 tiles (100%)
```

---

If you'd like, I can also update the repository README to replace the German file or add a note linking to this English version.

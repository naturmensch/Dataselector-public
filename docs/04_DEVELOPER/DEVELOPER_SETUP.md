# 👨‍💻 DEVELOPER SETUP & ARCHITECTURE

**Dokument:** `docs/04_DEVELOPER/DEVELOPER_SETUP.md`  
**Zielgruppe:** Contributor, Entwickler  
**Status:** Production Ready

---

## ⚡ Schneller Developer Setup (10 Minuten)

### Prerequisites
- Python 3.9+ (empfohlen: 3.11)
- Git
- micromamba (für Lock-File Reproducibility)

### Installation (Option 1: Lock-File - Empfohlen)

```bash
# Clone & enter repo
git clone <repository-url> Dataselector
cd Dataselector

# Install exact versions (reproduzierbar)
micromamba create -f conda-lock.yml -n dataselector

# Pip extras (Testing, Dev Tools)
./scripts/exec_in_env.sh --env dataselector -- pip install -r requirements.txt
./scripts/exec_in_env.sh --env dataselector -- pip install -e .  # Editable install für lokale Entwicklung
```

### Installation (Option 2: From Scratch)

```bash
# Create fresh env
micromamba create -n dataselector python=3.11

# Install all dependencies
./scripts/exec_in_env.sh --env dataselector -- pip install -r requirements.txt
./scripts/exec_in_env.sh --env dataselector -- pip install -r requirements-geo.txt  # Geo tools (optional)
./scripts/exec_in_env.sh --env dataselector -- pip install -e .
```

### Verify Installation

```bash
# Test imports
./scripts/exec_in_env.sh --env dataselector -- python -c "import dataselector; print(dataselector.__version__)"

# Run quick test
./scripts/exec_in_env.sh --env dataselector -- pytest tests/test_smoke.py -v
```

---

## 📦 Code Architecture

### Package Structure

```
dataselector/
├── __init__.py
├── feature_extraction/
│   ├── __init__.py
│   ├── extractors.py          # DINOv2, ResNet50
│   ├── cache.py               # Feature cache management
│   └── handlers.py            # Image I/O, Preprocessing
├── clustering/
│   ├── __init__.py
│   ├── kmeans.py              # K-Means wrapper
│   └── umap_embedding.py      # UMAP 2D embedding
├── selection/
│   ├── __init__.py
│   ├── facility_location.py   # Submodular greedy algorithm
│   ├── multi_criteria.py      # α,β,γ weighting
│   └── constraints.py         # Spatial hard-constraints
├── optimization/
│   ├── __init__.py
│   ├── optuna_handlers.py     # Optuna integration
│   ├── samplers.py            # QMC, TPE, CMA-ES
│   └── bootstrap.py           # Bootstrap UQ
├── experiments/
│   ├── __init__.py
│   ├── manager.py             # ExperimentManager (provenance)
│   └── manifests.py           # JSON manifest handling
├── workflows/
│   ├── __init__.py
│   ├── thesis_pipeline.py     # canonical thesis workflow
│   └── thesis_orchestrate.py  # orchestration + snapshots
├── utils/
│   ├── __init__.py
│   ├── config.py              # YAML loading
│   ├── logging.py             # Logging setup
│   └── metrics.py             # Evaluation metrics
└── cli.py                      # Command-line interface
```

---

## 🔑 Core Module APIs

### 1. Feature Extraction (`dataselector.feature_extraction`)

```python
from dataselector.feature_extraction import DINOv2Extractor

# Initialize
extractor = DINOv2Extractor(model_name="dinov2_vits14")

# Extract features (cached automatically)
features = extractor.extract_batch(
    image_paths=["img1.jpg", "img2.jpg"],
    cache_dir="outputs/features/",
    batch_size=32
)
# Output: (N, 384) or (N, 768) tensor
```

### 2. Clustering (`dataselector.clustering`)

```python
from dataselector.clustering import KMeansEmbedder

# Cluster in 2D space
embedder = KMeansEmbedder(n_clusters=8, metric="cosine")
clusters, centers = embedder.fit_predict(features)
# Output: clusters (N,), centers (8, 384)
```

### 3. Multi-Criteria Selection (`dataselector.selection`)

```python
from dataselector.selection import MultiCriteriaSelection

selector = MultiCriteriaSelection(
    alpha=0.4,      # Visual diversity weight
    beta=0.3,       # Spatial diversity weight
    gamma=0.3,      # Temporal diversity weight
    min_distance_km=40  # Hard constraint
)

selected_indices = selector.select(
    features=features,
    coordinates=tile_coords,  # (N, 2) lat/lon
    dates=tile_dates,         # (N,) timestamps
    n_select=100
)
# Output: (100,) array of selected indices
```

### 4. Experiment Tracking (`dataselector.experiments`)

```python
from dataselector.experiments import ExperimentManager

manager = ExperimentManager(
    base_dir="outputs/",
    experiment_name="thesis_phase1"
)

# Log parameters
manager.log_config({
    "alpha": 0.4,
    "beta": 0.3,
    "feature_extractor": "dinov2"
})

# Save results with provenance
manager.save_results(
    data=selected_indices,
    filename="selected_tiles.csv",
    tags=["phase1", "lhs"]
)

# Retrieve manifest
manifest = manager.load_manifest()
print(manifest["creation_time"])  # Auto-timestamped
```

---

## 🧪 Testing & CI/CD

### Running Tests

```bash
# All tests (89 suite)
pytest

# Fast feedback loop (only recent failures)
pytest --lf

# With coverage
pytest --cov=dataselector

# Specific test file
pytest tests/test_facility_location.py -v
```

### GitHub Actions CI

**Trigger:** On every push

```
1. Lint (pylint, mypy)
2. Unit tests (pytest on 3 Python versions)
3. Integration tests
4. Generate JUnit XML for artifact upload
```

**Artifacts:** `junit-results-<python>-<optuna>.xml`

### Pre-Commit Checklist

```bash
# Before committing
pytest --lf       # Fast
pytest            # Full suite
mypy dataselector/ --ignore-missing-imports
```

---

## 📝 Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/new-sampler

# 2. Implement with type hints
# Example: new sampler

# 3. Add tests
# tests/test_new_sampler.py

# 4. Run tests frequently
pytest --lf

# 5. Before PR: full test suite
pytest

# 6. Push & create PR
git push origin feature/new-sampler
```

### Code Style

| Tool | Config | Purpose |
|------|--------|---------|
| **Black** | `pyproject.toml` | Code formatting (88 chars) |
| **Pylint** | `.pylintrc` | Linting |
| **MyPy** | `mypy.ini` | Type checking |
| **Pytest** | `pytest.ini` | Testing |

**Run all checks:**
```bash
black dataselector/
pylint dataselector/
mypy dataselector/ --ignore-missing-imports
pytest
```

---

## 🔧 Key Development Tasks

### Adding a New Sampler

**Files to modify:**
1. `dataselector/optimization/samplers.py` - Implement sampler class
2. `tests/test_samplers.py` - Add tests
3. `dataselector/cli.py` - Add CLI flag

**Example:**

```python
# In dataselector/optimization/samplers.py
class GridSampler(BaseSampler):
    """Grid-based sampler for parameter exploration."""
    
    def __init__(self, grid_points: List[float]):
        self.grid = grid_points
    
    def suggest(self, trial: optuna.Trial) -> float:
        # Implement grid-based suggestion
        pass
```

### Modifying Pipeline Configuration

**Edit:** `config/pipeline_config.yaml`

```yaml
feature_extractor: dinov2          # or resnet50
n_clusters: 8
weights:
  alpha: 0.4
  beta: 0.3
  gamma: 0.3
min_distance_km: 40
n_lhs_samples: 50
```

**Load in code:**
```python
from dataselector.utils import load_config
cfg = load_config("config/pipeline_config.yaml")
```

---

## 🚨 Common Issues & Debugging

### "ImportError: No module named 'dataselector'"

```bash
# Solution: Install in editable mode
pip install -e .
```

### "CUDA Out of Memory"

```bash
# Reduce batch size in config
feature_extractor:
  batch_size: 16  # from 32

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

### "MyPy: Cannot find type stub"

```bash
# Add to mypy.ini
[mypy]
ignore_missing_imports = True
```

### Test Suite Hangs

```bash
# Check for infinite loops in conftest fixtures
pytest -v --tb=short --timeout=30
```

---

## 📚 Weiterführende Ressourcen

- **Testing Guide:** [../TEST_SUITE_CURATION.md](../08_GOVERNANCE/TEST_SUITE_CURATION.md)
- **CI/CD Setup:** [../.github/workflows/]
- **Architecture Deep Dive:** [../02_THEORY/architecture.md](../02_THEORY/architecture.md)
- **Module Reference:** [../06_REFERENCE/api_reference.md](../06_REFERENCE/api_reference.md)

---

## 🔗 Wichtige Commands (Schnellreferenz)

```bash
# Development
pip install -e .                  # Editable install
pytest --lf                       # Fast loop
pytest                            # Full suite

# Code Quality
black dataselector/               # Format
mypy dataselector/                # Type check
pylint dataselector/              # Lint

# Configuration
python -c "from dataselector.utils import load_config; print(load_config())"

# Debugging
python -c "import dataselector; print(dataselector.__file__)"
```

---

**Last Updated:** 2. Februar 2026  
**Status:** Production Ready

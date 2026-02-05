# E2E Test Strategy & Structure

**Status:** Post-migration test infrastructure planning (2026-02-02)

## Test Categorization

### 1. Smoke Tests (Fast Validation, 1-5 min each)

Quick sanity checks - command runs without errors, basic outputs exist.

| Test | Command | Purpose | Duration | Parallelizable |
|---|---|---|---|---|
| `test_thesis_pipeline_smoke` | `dataselector thesis-pipeline --dry-run --n-lhs 5` | CLI startup test | 2 min | ✅ Yes |
| `test_autoscale_smoke` | `dataselector autoscale --stages 5` | Autoscale basic run | 3 min | ✅ Yes |
| `test_sampler_suite_smoke` | `dataselector sampler-suite --n-trials 5 --n-seeds 2` | Sampler comparison | 2 min | ✅ Yes |
| `test_bootstrap_smoke` | `dataselector bootstrap final --n-boot 10` | Bootstrap basic | 1 min | ✅ Yes |
| `test_build_tiles_smoke` | `dataselector build-tiles --image-dir samples/` | Tile building | 1 min | ✅ Yes |
| `test_tools_check_geo_smoke` | `dataselector tools check-geo` | Geo dependencies | <1 min | ✅ Yes |

**Total Duration:** ~10 min (can run in parallel: ~3 min)  
**Failure Rate:** Expected 0-1% (usually environment issues)

### 2. Integration Tests (Feature Validation, 5-15 min each)

Tests combining multiple components or validating specific features.

| Test | Command | Purpose | Duration | Dependencies |
|---|---|---|---|---|
| `test_optuna_persistence` | 2 seed runs with same study DB | Database persistence | 10 min | - |
| `test_uq_reproducibility` | Same seed bootstrap 2x | Deterministic UQ | 5 min | - |
| `test_final_selection_variants` | 3 ranking methods | Ranking logic | 8 min | - |
| `test_adaptive_auto_samples` | Adaptive n_samples | Auto-sampling logic | 7 min | - |
| `test_bootstrap_multi_seed` | 3 seeds × N bootstrap | Multi-seed stability | 12 min | - |
| `test_build_tiles_real_images` | Real image directory | Full tile building | 6 min | Images needed |
| `test_sampler_comparison_seeds` | 5 seeds, 3 samplers | Sampler evaluation | 15 min | Multi-seed benchmark |

**Total Duration:** ~65 min (parallel: ~15 min with batching)  
**Failure Rate:** Expected 0-5% (usually data-related)

### 3. End-to-End Workflows (Full Pipelines, 10-30+ min each)

Complete workflow testing - multiple phases or data pipelines.

| Test | Workflow | Purpose | Duration | Resources |
|---|---|---|---|---|
| `test_autoscale_multi_stage` | 3 autoscale stages | Multi-stage optimization | 15 min | Medium |
| `test_xxl_complete_5_phases` | Abbreviated XXL (Hamburg subset) | 5-phase pipeline | 25 min | High |
| `test_resume_recovery` | XXL interrupt + resume | Checkpoint/resume logic | 20 min | High |
| `test_geo_workflow_full` | build-tiles → align-audit | Complete geo pipeline | 10 min | Images + GIS |

**Total Duration:** ~70 min (recommended: run sequentially or dedicated CI stage)  
**Failure Rate:** Expected 1-10% (integration complexity)

### 4. Error & Fallback Tests (Edge Cases, 2-5 min each)

Tests for error handling and graceful degradation.

| Test | Scenario | Purpose | Expected Result |
|---|---|---|---|
| `test_error_missing_gis_deps` | Simulate missing geopandas | Graceful error handling | Clear error message, non-zero exit |
| `test_error_missing_images` | Missing image directory | Handle missing inputs | Validation error, suggest fix |
| `test_tools_protect_paths` | Staged protected files | Protected path enforcement | Detection + warning |
| `test_env_compatibility_fallback` | Missing optional deps | Fallback behavior | Works with warnings |
| `test_graceful_interrupt` | Kill process mid-run | State preservation | Clean shutdown, resumable |

**Total Duration:** ~20 min  
**Failure Rate:** Expected 0% (should be reliable)

## Test Parallelization Strategy

### Stage 1: Parallel Smoke Tests (3 min)
All 6 smoke tests can run simultaneously:
```bash
pytest tests/e2e/ -k "smoke" -n 6  # with pytest-xdist
```

**Typical CI time:** ~5 min (startup + shared overhead)

### Stage 2: Parallel Integration Tests (15 min)
7 integration tests, batched into 3 groups (resource limits):
```bash
Group A (Low resource): optuna_persistence, uq_reproducibility, final_selection_variants
Group B (Medium resource): adaptive_auto_samples, bootstrap_multi_seed
Group C (Image-dependent): build_tiles_real_images, sampler_comparison_seeds
```

**Typical CI time:** ~18 min (some sequential due to resource constraints)

### Stage 3: Sequential E2E Workflows (90 min)
Run complete pipelines sequentially to avoid resource conflicts:
```bash
pytest tests/e2e/test_autoscale_workflow.py
pytest tests/e2e/test_xxl_pipeline.py
pytest tests/e2e/test_resume_recovery.py
pytest tests/e2e/test_geo_workflow.py
```

**Typical CI time:** ~95 min (includes setup/teardown)

### Stage 4: Error Tests (20 min)
Run in parallel (isolated, no resource contention):
```bash
pytest tests/e2e/ -k "error" -n 5
```

**Typical CI time:** ~25 min

## Total CI Runtime Estimate

| Stage | Tests | Parallel | Duration |
|---|---|---|---|
| Smoke | 6 | ✅ Yes | 5 min |
| Integration | 7 | 🟡 Partial | 18 min |
| E2E Workflows | 4 | ❌ No | 95 min |
| Error | 5 | ✅ Yes | 25 min |
| **Total** | 22 | - | **143 min** (~2.4 hours) |

**Optimized (parallel where safe):** ~120 min (~2 hours)

## Fixture Strategy

### Shared Fixtures (conftest.py)

```python
@pytest.fixture
def tmp_workspace(tmp_path):
    """Temporary workspace with standard structure."""
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    yield tmp_path
    # Auto-cleanup on teardown

@pytest.fixture
def sample_csv(tmp_path):
    """Small sample CSV for quick tests."""
    csv_file = tmp_path / "sample.csv"
    # Write minimal CSV with 50-100 rows
    yield csv_file

@pytest.fixture(scope="session")
def skip_if_no_gis():
    """Skip test if GIS dependencies missing."""
    pytest.importorskip("geopandas")

@pytest.fixture
def cleanup_on_teardown(tmp_workspace):
    """Auto-cleanup after test."""
    yield tmp_workspace
    # Force cleanup
```

### Fixture Dependencies

```
tmp_workspace (base)
├── sample_csv (derived from tmp_workspace)
├── skip_if_no_gis (independent, session-scoped)
└── cleanup_on_teardown (wraps tmp_workspace)
```

## Running Tests Locally

### Quick sanity check (smoke tests only)
```bash
pytest tests/e2e/ -k "smoke" -v  # ~5 min
```

### Full E2E suite
```bash
pytest tests/e2e/ -v  # ~2-3 hours
```

### Single test for debugging
```bash
pytest tests/e2e/test_autoscale_workflow.py::test_autoscale_multi_stage -vv -s
```

### With coverage report
```bash
pytest tests/e2e/ --cov=dataselector --cov-report=html
```

## CI/CD Integration

### GitHub Actions Workflow

**Stage 1: Fast Tests (~5 min)**
```yaml
- name: "Smoke Tests"
  run: pytest tests/e2e/ -k "smoke" --tb=short
```

**Stage 2: Integration Tests (~20 min)**
```yaml
- name: "Integration Tests"
  run: pytest tests/e2e/ -k "integration" --tb=short
```

**Stage 3: E2E Workflows (~100 min) [Optional, nightly]**
```yaml
- name: "E2E Workflows"
  if: ${{ github.event_name == 'schedule' }}  # Nightly only
  run: pytest tests/e2e/ -k "workflow" --tb=short
```

**Stage 4: Error Tests (~20 min)**
```yaml
- name: "Error Handling Tests"
  run: pytest tests/e2e/ -k "error" --tb=short
```

### Recommended CI Timing

| Pipeline | Trigger | Duration | Resources |
|---|---|---|---|
| **PR checks** | Push to PR | ~10 min | 1 runner |
| **Daily** | Daily 3am UTC | ~2.5 hours | 1 runner |
| **Release** | Tag creation | ~2.5 hours | 2 runners (parallel) |

## Expected Test Outcomes

### Smoke Tests
- ✅ Should pass 100% of the time (basic functionality)
- 🔍 Catch: Broken CLI, import errors, missing dependencies
- ⏱️ Ideal for: Every PR, every commit

### Integration Tests
- ✅ Should pass 95%+ (occasional data-related failures)
- 🔍 Catch: Feature regressions, algorithm changes, I/O issues
- ⏱️ Ideal for: Daily runs, before release

### E2E Workflows
- ✅ Should pass 90%+ (complex, more failure modes)
- 🔍 Catch: Full pipeline integration issues, optimization problems
- ⏱️ Ideal for: Nightly runs, before release

### Error Tests
- ✅ Should pass 100% (deterministic error cases)
- 🔍 Catch: Missing error handling, bad error messages
- ⏱️ Ideal for: Every PR, critical for UX

## Maintenance & Monitoring

### Test Flakiness Detection
Track tests that fail inconsistently:
- Re-run flaky tests 3x
- Log failure rates in CI reports
- Investigate environmental dependencies

### Coverage Tracking
Target: 70%+ of `dataselector.workflows` and `dataselector.tools` modules

```bash
pytest tests/e2e/ --cov=dataselector --cov-report=term --cov-fail-under=70
```

### Performance Monitoring
Track test duration trends (identify performance regressions):
```bash
# Generate timing report
pytest tests/e2e/ -v --durations=10
```

---

**Last Updated:** 2026-02-02  
**Owner:** QA/Testing Team  
**Next Review:** After first 10 test implementations

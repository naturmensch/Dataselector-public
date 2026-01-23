"""
Tests for XXL Thesis Complete Pipeline (Corrected Parameters)
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Load the thesis run script directly (avoid sys.path modifications)
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "xxl_KDR146_run_thesis_complete",
    ROOT / "scripts" / "xxl_KDR146_run_thesis_complete.py",
)
xxl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xxl_mod)

_extract_xxl_final_statistics = xxl_mod._extract_xxl_final_statistics
_find_xxl_runs = xxl_mod._find_xxl_runs
_validate_convergence_from_validation_data = (
    xxl_mod._validate_convergence_from_validation_data
)
phase_1_xxl_hamburg = xxl_mod.phase_1_xxl_hamburg
phase_2_reproducibility = xxl_mod.phase_2_reproducibility
run_cmd_with_retry = xxl_mod.run_cmd_with_retry


def test_extract_statistics_with_valid_data(tmp_path):
    """Test statistics extraction with valid mock data."""
    # Create mock directory structure (matching pattern: *hamburg_xxl_final*)
    run_dir = tmp_path / "outputs" / "runs" / "20260118_T120000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    # Create mock trials.csv
    mock_data = {
        "trial_number": [0, 1, 2, 3, 4],
        "state": ["TrialState.COMPLETE"] * 5,
        "value": [75.5, 76.2, 77.8, 76.0, 75.9],
        "a": [0.55, 0.60, 0.58, 0.52, 0.57],
        "b": [0.10, 0.12, 0.09, 0.11, 0.10],
        "c": [0.35, 0.28, 0.33, 0.37, 0.33],
        "min_distance_km": [40.0, 42.0, 38.0, 41.0, 40.5],
        "n_samples": [34, 35, 33, 34, 34],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)

    # Run extraction
    result = _extract_xxl_final_statistics(tmp_path)

    # Assertions
    assert result is not None, "Extraction should succeed"
    assert result["best_value"] == 77.8, "Should find best value"
    assert result["best_trial"] == 2, "Should find best trial"
    assert result["n_trials"] == 5, "Should count all trials"
    assert "best_params" in result, "Should extract params"
    assert result["best_params"]["a"] == 0.58, "Should extract correct alpha"
    assert "timestamp" in result, "Should include timestamp"

    # Check JSON was written
    json_file = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    assert json_file.exists(), "Should write JSON file"

    with open(json_file) as f:
        saved = json.load(f)
    assert saved["best_value"] == 77.8, "Saved JSON should match"


def test_extract_statistics_includes_config_if_present(tmp_path):
    """Test that extraction reads config_optuna.yaml and includes configured params."""
    run_dir = tmp_path / "outputs" / "runs" / "20260118_T120000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    # create trials.csv
    mock_data = {
        "trial_number": [0, 1],
        "state": ["TrialState.COMPLETE", "TrialState.COMPLETE"],
        "value": [1.0, 2.0],
        "a": [0.1, 0.2],
        "b": [0.1, 0.2],
        "c": [0.1, 0.2],
        "min_distance_km": [10, 10],
        "n_samples": [30, 30],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)

    # Create config file
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir()
    import yaml

    cfg = {"sampler": "cmaes", "n_trials": 500, "n_candidates": 673}
    (cfg_dir / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    result = _extract_xxl_final_statistics(tmp_path)
    assert result is not None
    assert result.get("configured_sampler") == "cmaes"
    assert result.get("configured_n_trials") == 500
    assert result.get("configured_n_candidates") == 673


def test_extract_statistics_missing_run(tmp_path):
    """Test extraction fails gracefully when run directory missing."""
    # No run directory created
    result = _extract_xxl_final_statistics(tmp_path)
    assert result is None, "Should return None when run not found"


def test_extract_statistics_missing_trials_csv(tmp_path):
    """Test extraction fails when trials.csv missing."""
    # Create run dir but no trials.csv
    run_dir = tmp_path / "outputs" / "runs" / "20260118_hamburg_thesis_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    result = _extract_xxl_final_statistics(tmp_path)
    assert result is None, "Should return None when trials.csv not found"


def test_extract_statistics_empty_trials(tmp_path):
    """Test extraction fails when no completed trials."""
    run_dir = tmp_path / "outputs" / "runs" / "20260118_hamburg_thesis_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    # Create empty trials.csv
    mock_data = {
        "trial_number": [0, 1],
        "state": ["TrialState.FAIL", "TrialState.PRUNED"],
        "value": [None, None],
        "a": [None, None],
        "b": [None, None],
        "c": [None, None],
        "min_distance_km": [None, None],
        "n_samples": [None, None],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)

    result = _extract_xxl_final_statistics(tmp_path)
    assert result is None, "Should return None when no completed trials"


def test_validate_convergence_with_mock_validation_data(tmp_path):
    """Test convergence validation with mock 10-seed Hamburg data."""
    # Create mock Hamburg validation runs (seeds 42-51, need at least 5 for validation)
    for seed in range(42, 48):  # Use 6 seeds for test (meets 5 seed minimum)
        run_dir = (
            tmp_path / "outputs" / "runs" / f"20260117_hamburg_cmaes_500trials_s{seed}"
        )
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)

        # Create mock trials with convergence at trial ~80
        trials = []
        for i in range(500):
            # Converge around trial 80
            value = 77.0 - 0.5 * (1 + np.exp(-(i - 80) / 20))
            trials.append(
                {
                    "trial_number": i,
                    "state": "TrialState.COMPLETE",
                    "value": value,
                    "a": 0.55,
                    "b": 0.10,
                    "c": 0.35,
                    "min_distance_km": 40.0,
                    "n_samples": 34,
                }
            )

        df = pd.DataFrame(trials)
        trials_csv = results_dir / "trials.csv"
        df.to_csv(trials_csv, index=False)

    # Run convergence analysis
    result = _validate_convergence_from_validation_data(tmp_path)

    # Should work with at least 5 seeds
    assert result is not None, "Should find convergence data"
    assert result["n_seeds_analyzed"] >= 5, "Should analyze at least 5 seeds"
    assert (
        70 <= result["convergence_99_trials_median"] <= 120
    ), "99% should converge around trial 80-100"


def test_run_cmd_with_retry_success():
    # 'true' exists on POSIX and returns exit code 0
    assert run_cmd_with_retry("true", retries=2, delay=0) == 0


def test_run_cmd_with_retry_failure():
    # 'false' exists on POSIX and returns exit code 1
    assert run_cmd_with_retry("false", retries=1, delay=0) != 0


@pytest.fixture
def mock_run_cmd(monkeypatch):
    """Mock run_cmd_with_retry to avoid starting real subprocesses."""
    captured = {}

    def fake_run(cmd, retries=2, delay=5, cwd=None, fail_ok=False):
        captured["cmd"] = cmd
        return 0

    # Patch the run_cmd_with_retry on the loaded module object to avoid invoking real subprocesses
    monkeypatch.setattr(xxl_mod, "run_cmd_with_retry", fake_run)
    return captured


def test_phase1_pass_params_false(mock_run_cmd):
    # Use the fixture for cleaner mocking
    phase_1_xxl_hamburg(n_trials=123, n_candidates=456, pass_params=False)
    assert "--n-trials" not in mock_run_cmd["cmd"]
    assert "--n-candidates" not in mock_run_cmd["cmd"]


def test_phase1_includes_dry_run_flag_when_requested(mock_run_cmd):
    phase_1_xxl_hamburg(n_trials=50, n_candidates=20, pass_params=True, dry_run=True)
    assert "--dry-run" in mock_run_cmd["cmd"]


def test_phase2_includes_dry_run_flag_when_requested(mock_run_cmd):
    phase_2_reproducibility(
        seeds=[99], n_trials=10, n_candidates=20, pass_params=True, dry_run=True
    )
    assert "--dry-run" in mock_run_cmd["cmd"]


def test_phase2_pass_params_false(mock_run_cmd):
    # Use the fixture for cleaner mocking
    phase_2_reproducibility(
        seeds=[99], n_trials=123, n_candidates=456, pass_params=False
    )
    assert "--n-trials" not in mock_run_cmd["cmd"]
    assert "--n-candidates" not in mock_run_cmd["cmd"]


# New tests for caching, archive fallback, and NaN handling


def test_validate_convergence_uses_cache(tmp_path, monkeypatch):
    # Prepare a cached baseline file
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    baseline = {
        "n_seeds_analyzed": 3,
        "convergence_99_trials_median": 100,
        "convergence_99_trials_min": 90,
        "convergence_99_trials_max": 110,
        "convergence_99_trials_all": [95, 100, 105],
        "generated_at": "2026-01-01T00:00:00Z",
    }
    (outputs / "convergence_baseline.json").write_text(json.dumps(baseline))

    # Call function - should return cached data without error
    result = _validate_convergence_from_validation_data(tmp_path)
    assert result is not None
    assert result["n_seeds_analyzed"] == 3
    assert result["convergence_99_trials_median"] == 100


def test_validate_convergence_falls_back_to_archive_runs(tmp_path):
    # Create mock runs inside archive_local/old_runs
    for seed in range(42, 45):
        run_dir = (
            tmp_path
            / "archive_local"
            / "old_runs"
            / f"20260117_hamburg_cmaes_500trials_s{seed}"
        )
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)
        # simple trials: converge at trial 50
        mock_data = {
            "trial_number": list(range(200)),
            "state": ["TrialState.COMPLETE"] * 200,
            "value": [float(i <= 50 and i or 50.0) for i in range(200)],
        }
        df = pd.DataFrame(mock_data)
        (results_dir / "trials.csv").write_text(df.to_csv(index=False))

    result = _validate_convergence_from_validation_data(tmp_path)
    assert result is not None
    assert result["n_seeds_analyzed"] >= 3
    assert result["convergence_99_trials_median"] <= 60


def test_extract_statistics_with_thesis_named_run(tmp_path):
    """Ensure extraction also works with run directories like 'thesis_xxl_hamburg_final'."""
    run_dir = (
        tmp_path / "outputs" / "runs" / "20260119_T124819_thesis_xxl_hamburg_final"
    )
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)

    # Create mock trials.csv
    mock_data = {
        "trial_number": [0, 1, 2],
        "state": ["TrialState.COMPLETE"] * 3,
        "value": [10.0, 12.5, 11.0],
        "a": [0.55, 0.60, 0.58],
        "b": [0.10, 0.12, 0.09],
        "c": [0.35, 0.28, 0.33],
        "min_distance_km": [40.0, 42.0, 38.0],
        "n_samples": [34, 35, 33],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)

    result = _extract_xxl_final_statistics(tmp_path)
    assert result is not None
    assert result["best_value"] == 12.5
    assert result["best_trial"] == 1


def test_find_xxl_runs_detects_various_names(tmp_path):
    """Helper should find runs with different naming conventions."""
    p1 = tmp_path / "outputs" / "runs" / "20260118_T120000_hamburg_xxl_final"
    p2 = tmp_path / "outputs" / "runs" / "20260119_T124819_thesis_xxl_hamburg_final"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)

    found = _find_xxl_runs(tmp_path)
    assert p1 in found and p2 in found

    # Create mock runs inside archive_local/old_runs
    for seed in range(42, 45):
        run_dir = (
            tmp_path
            / "archive_local"
            / "old_runs"
            / f"20260117_hamburg_cmaes_500trials_s{seed}"
        )
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)
        # simple trials: converge at trial 50
        mock_data = {
            "trial_number": list(range(200)),
            "state": ["TrialState.COMPLETE"] * 200,
            "value": [float(i <= 50 and i or 50.0) for i in range(200)],
        }
        df = pd.DataFrame(mock_data)
        (results_dir / "trials.csv").write_text(df.to_csv(index=False))

    result = _validate_convergence_from_validation_data(tmp_path)
    assert result is not None
    assert result["n_seeds_analyzed"] >= 3
    assert result["convergence_99_trials_median"] <= 60


def test_validate_convergence_ignores_nan_values(tmp_path):
    # Create runs where some values are NaN but others valid
    for seed in range(42, 45):
        run_dir = (
            tmp_path / "outputs" / "runs" / f"20260117_hamburg_cmaes_500trials_s{seed}"
        )
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)
        values = [np.nan] * 10 + [1.0] * 490
        mock_data = {
            "trial_number": list(range(500)),
            "state": ["TrialState.COMPLETE"] * 500,
            "value": values,
        }
        df = pd.DataFrame(mock_data)
        (results_dir / "trials.csv").write_text(df.to_csv(index=False))

    result = _validate_convergence_from_validation_data(tmp_path)
    assert result is not None
    assert result["n_seeds_analyzed"] >= 3
    # median should be small since valid values mostly at end
    assert result["convergence_99_trials_median"] < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

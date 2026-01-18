"""
Tests for XXL Thesis Complete Pipeline (Corrected Parameters)
"""
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.xxl_KDR146_run_thesis_complete import _extract_xxl_final_statistics, _validate_convergence_from_validation_data, run_cmd_with_retry, phase_1_xxl_hamburg, phase_2_reproducibility


def test_extract_statistics_with_valid_data(tmp_path):
    """Test statistics extraction with valid mock data."""
    # Create mock directory structure (matching pattern: *hamburg_xxl_final*)
    run_dir = tmp_path / "outputs" / "runs" / "20260118_T120000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    
    # Create mock trials.csv
    mock_data = {
        'trial_number': [0, 1, 2, 3, 4],
        'state': ['TrialState.COMPLETE'] * 5,
        'value': [75.5, 76.2, 77.8, 76.0, 75.9],
        'a': [0.55, 0.60, 0.58, 0.52, 0.57],
        'b': [0.10, 0.12, 0.09, 0.11, 0.10],
        'c': [0.35, 0.28, 0.33, 0.37, 0.33],
        'min_distance_km': [40.0, 42.0, 38.0, 41.0, 40.5],
        'n_samples': [34, 35, 33, 34, 34],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)
    
    # Run extraction
    result = _extract_xxl_final_statistics(tmp_path)
    
    # Assertions
    assert result is not None, "Extraction should succeed"
    assert result['best_value'] == 77.8, "Should find best value"
    assert result['best_trial'] == 2, "Should find best trial"
    assert result['n_trials'] == 5, "Should count all trials"
    assert 'best_params' in result, "Should extract params"
    assert result['best_params']['a'] == 0.58, "Should extract correct alpha"
    assert 'timestamp' in result, "Should include timestamp"
    
    # Check JSON was written
    json_file = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    assert json_file.exists(), "Should write JSON file"
    
    with open(json_file) as f:
        saved = json.load(f)
    assert saved['best_value'] == 77.8, "Saved JSON should match"


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
        'trial_number': [0, 1],
        'state': ['TrialState.FAIL', 'TrialState.PRUNED'],
        'value': [None, None],
        'a': [None, None],
        'b': [None, None],
        'c': [None, None],
        'min_distance_km': [None, None],
        'n_samples': [None, None],
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
        run_dir = tmp_path / "outputs" / "runs" / f"20260117_hamburg_cmaes_500trials_s{seed}"
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True)
        
        # Create mock trials with convergence at trial ~80
        trials = []
        for i in range(500):
            # Converge around trial 80
            value = 77.0 - 0.5 * (1 + np.exp(-(i - 80) / 20))
            trials.append({
                'trial_number': i,
                'state': 'TrialState.COMPLETE',
                'value': value,
                'a': 0.55, 'b': 0.10, 'c': 0.35,
                'min_distance_km': 40.0,
                'n_samples': 34,
            })
        
        df = pd.DataFrame(trials)
        trials_csv = results_dir / "trials.csv"
        df.to_csv(trials_csv, index=False)
    
    # Run convergence analysis
    result = _validate_convergence_from_validation_data(tmp_path)
    
    # Should work with at least 5 seeds
    assert result is not None, "Should find convergence data"
    assert result['n_seeds_analyzed'] >= 5, "Should analyze at least 5 seeds"
    assert 70 <= result['convergence_99_trials_median'] <= 120, "99% should converge around trial 80-100"


def test_run_cmd_with_retry_success():
    # 'true' exists on POSIX and returns exit code 0
    assert run_cmd_with_retry('true', retries=2, delay=0) == 0


def test_run_cmd_with_retry_failure():
    # 'false' exists on POSIX and returns exit code 1
    assert run_cmd_with_retry('false', retries=1, delay=0) != 0


def test_phase1_pass_params_false(monkeypatch):
    captured = {}
    def fake_run(cmd, retries=2, delay=5, cwd=None, fail_ok=False):
        captured['cmd'] = cmd
        return 0
    monkeypatch.setattr('scripts.xxl_KDR146_run_thesis_complete.run_cmd_with_retry', fake_run)

    phase_1_xxl_hamburg(n_trials=123, n_candidates=456, pass_params=False)
    assert '--n-trials' not in captured['cmd']
    assert '--n-candidates' not in captured['cmd']


def test_phase2_pass_params_false(monkeypatch):
    captured = {}
    def fake_run(cmd, retries=2, delay=5, cwd=None, fail_ok=False):
        captured['cmd'] = cmd
        return 0
    monkeypatch.setattr('scripts.xxl_KDR146_run_thesis_complete.run_cmd_with_retry', fake_run)

    phase_2_reproducibility(seeds=[99], n_trials=123, n_candidates=456, pass_params=False)
    assert '--n-trials' not in captured['cmd']
    assert '--n-candidates' not in captured['cmd']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


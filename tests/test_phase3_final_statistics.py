import json
from pathlib import Path

import pandas as pd

from scripts.xxl_KDR146_run_thesis_complete import _extract_xxl_final_statistics


def test_extract_statistics_handles_missing_n_samples(tmp_path):
    """Phase 3 should succeed even if n_samples is missing (NaN)."""
    # Create a fake run dir matching the naming convention the script expects
    run_dir = tmp_path / "outputs" / "runs" / "20260118_T120000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create trials.csv with NaN n_samples and a distinct best value at trial 2
    mock_data = {
        'trial_number': [0, 1, 2, 3, 4],
        'state': ['TrialState.COMPLETE'] * 5,
        'value': [70.0, 71.0, 80.0, 69.0, 68.0],  # best at index 2
        'a': [0.5, 0.6, 0.58, 0.52, 0.57],
        'b': [0.1, 0.12, 0.09, 0.11, 0.10],
        'c': [0.35, 0.28, 0.33, 0.37, 0.33],
        'min_distance_km': [40.0, 42.0, 38.0, 41.0, 40.5],
        'n_samples': [None, None, None, None, None],
    }
    df = pd.DataFrame(mock_data)
    trials_csv = results_dir / "trials.csv"
    df.to_csv(trials_csv, index=False)

    result = _extract_xxl_final_statistics(tmp_path)
    assert result is not None, "Extraction should succeed even with missing n_samples"
    assert result['best_trial'] == 2
    assert result['best_params']['n_samples'] is None

    # Check saved JSON
    json_file = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    assert json_file.exists()
    with json_file.open() as f:
        saved = json.load(f)
    assert saved['best_params']['n_samples'] is None

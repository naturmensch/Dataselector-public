import json
from pathlib import Path
import pandas as pd
import types

import pytest

import scripts.xxl_KDR146_run_thesis_complete as xxl


def _make_fake_xxl_run(root: Path, name: str = "20260119_T000000_hamburg_xxl_final") -> Path:
    run_dir = root / 'outputs' / 'runs' / name
    results_dir = run_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create trials.csv with a clear best at trial 2
    mock_data = {
        'trial_number': [0, 1, 2, 3],
        'state': ['TrialState.COMPLETE'] * 4,
        'value': [70.0, 71.0, 85.0, 69.0],  # best at index 2
        'a': [0.5, 0.6, 0.58, 0.52],
        'b': [0.1, 0.12, 0.09, 0.11],
        'c': [0.35, 0.28, 0.33, 0.37],
        'min_distance_km': [40.0, 42.0, 38.0, 41.0],
        'n_samples': [30, 30, 25, 30],
    }
    df = pd.DataFrame(mock_data)
    (results_dir / 'trials.csv').write_text(df.to_csv(index=False))

    # Also write a run config to simulate values Phase 2 might have preserved
    cfg_dir = run_dir / 'config'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        'sampler': 'sobol',
        'n_trials': 440,
        'n_candidates': 673,
    }
    import yaml
    (cfg_dir / 'config_optuna.yaml').write_text(yaml.safe_dump(cfg))
    return run_dir


def test_phase2_to_phase3_transition(tmp_path, monkeypatch):
    """Simulate Phase 2 completing (via mocked runner) and verify Phase 3 runs
    and consumes the artifacts produced (e.g., trials.csv --> THESIS_FINAL_SELECTION_XXL.json).
    """
    # Point module ROOT to our temp directory
    monkeypatch.setattr(xxl, 'ROOT', tmp_path)

    # Prepare a fake XXL run directory that Phase 3 should pick up
    run_dir = _make_fake_xxl_run(tmp_path)

    # Monkeypatch run_cmd_with_retry used by phase_2 to a stub that records calls
    calls = []

    def fake_run_cmd_with_retry(cmd, retries=0, delay=0):
        calls.append(cmd)
        # Simulate that reproducibility run also writes a small artifact (marker file)
        # This mimics the child process side-effect. Create one marker per seed invocation.
        if '--exp-name' in cmd:
            # find exp-name value
            m = [p for p in cmd.split() if p.startswith('--exp-name')]
            if m:
                name = m[0].split('=')[-1] if '=' in m[0] else cmd.split()[-1]
            else:
                name = 'unknown'
            marker_dir = tmp_path / 'outputs' / 'repro_runs' / name
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / 'completed.marker').write_text('ok')
        return 0

    monkeypatch.setattr(xxl, 'run_cmd_with_retry', fake_run_cmd_with_retry)

    # Run phase 2: should return True and call our fake runner
    ok2 = xxl.phase_2_reproducibility(seeds=[43, 44], n_trials=10, n_candidates=20, pass_params=True, dry_run=False)
    assert ok2 is True
    assert len(calls) == 2, "Expected two reproducibility runs (one per seed)"

    # Now run Phase 3 which should find the fake XXL run we created and produce final selection
    ok3 = xxl.phase_3_final_statistics()
    assert ok3 is True

    # Ensure THESIS_FINAL_SELECTION_XXL.json was written and contains expected best_trial
    json_file = tmp_path / 'outputs' / 'THESIS_FINAL_SELECTION_XXL.json'
    assert json_file.exists(), "Phase 3 should write final selection JSON"
    with json_file.open() as f:
        sel = json.load(f)

    assert sel['best_trial'] == 2
    assert sel['best_params']['n_samples'] == 25
    # Also ensure Phase 2 left its marker artifacts
    repro_marker = tmp_path / 'outputs' / 'repro_runs' / 'thesis_hamburg_reproducibility_s43' / 'completed.marker'
    assert repro_marker.exists()

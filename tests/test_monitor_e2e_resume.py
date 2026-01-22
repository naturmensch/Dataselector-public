import json

import pandas as pd

import scripts.xxl_full_run_monitor as monitor


def test_monitor_auto_resume_finalizes_from_trials_csv(monkeypatch, tmp_path):
    """End-to-end: Monitor resume should detect trials.csv, plan finalize, and produce final selection."""
    # Prepare run dir with trials.csv and config
    run_dir = tmp_path / "outputs" / "runs" / "20260120_T123000_hamburg_xxl_final"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Create trials.csv with a clear best at trial 2
    mock_data = {
        "trial_number": [0, 1, 2, 3],
        "state": ["TrialState.COMPLETE"] * 4,
        "value": [70.0, 71.0, 85.0, 69.0],  # best at index 2
        "a": [0.5, 0.6, 0.58, 0.52],
        "b": [0.1, 0.12, 0.09, 0.11],
        "c": [0.35, 0.28, 0.33, 0.37],
        "min_distance_km": [40.0, 42.0, 38.0, 41.0],
        "n_samples": [30, 30, 25, 30],
    }
    df = pd.DataFrame(mock_data)
    (results_dir / "trials.csv").write_text(df.to_csv(index=False))

    # Write a config with n_trials equal to completed (so remaining == 0)
    import yaml

    (cfg_dir / "config_optuna.yaml").write_text(yaml.safe_dump({"n_trials": 4}))

    # Monkeypatch ROOT to our tmp path
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Monkeypatch run_hook: for finalize, call the extractor to create final selection
    def fake_run_hook(
        name,
        cmd_str,
        base_log_dir,
        active_log,
        timeout,
        retries,
        env,
        start_new_session,
        pass_dry_run,
    ):
        if name == "resume_phase_finalize":
            from scripts.xxl_KDR146_run_thesis_complete import (
                _extract_xxl_final_statistics,
            )

            rc = _extract_xxl_final_statistics(tmp_path)
            return {"success": bool(rc)}
        # For any other phase, just pretend success
        return {"success": True}

    monkeypatch.setattr(monitor, "run_hook", fake_run_hook)

    # Run the resume flow (force)
    res = monitor._resume_run(
        "last", tmp_path / "monitor.log", force=True, dry_run=False
    )

    assert res.get("ok") is True
    # Finalize should have been in phases
    assert any(p["name"] == "finalize" for p in res.get("phases", []))

    # Final selection JSON should exist in ROOT/outputs
    json_file = tmp_path / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    assert json_file.exists()
    with json_file.open() as f:
        j = json.load(f)
    assert j["best_trial"] == 2
    assert j["best_params"]["n_samples"] == 25
    # Resume meta should indicate we used trials_csv as source
    meta_file = run_dir / "results" / "resume_meta.json"
    assert meta_file.exists()
    with meta_file.open() as f:
        meta = json.load(f)
    assert meta["resume_source"] == "trials_csv"

import json

import dataselector.workflows.xxl as mod


def test_finalize_without_run_dir_succeeds(tmp_path):
    rc = mod.main(phase="finalize", output_dir=str(tmp_path / "outputs"), smoke=True)
    assert rc == 0
    assert (tmp_path / "outputs" / "thesis_finalization_summary.json").exists()


def test_finalize_with_run_dir_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "run_workflow", lambda *args, **kwargs: 0)

    # Create a fake run dir with a best_trial.json so bootstrap/finalize can proceed
    run = tmp_path / "outputs" / "runs" / "rr1"
    (run / "results").mkdir(parents=True, exist_ok=True)
    best = {"a": 0.2, "b": 0.3, "c": 0.5, "min_distance_km": 50, "n_samples": 40}
    (run / "results" / "best_trial.json").write_text(json.dumps(best))

    rc = mod.main(
        phase="finalize",
        run_dir=str(run),
        output_dir=str(tmp_path / "outputs"),
        smoke=True,
    )
    assert rc == 0
    assert (tmp_path / "outputs" / "thesis_finalization_summary.json").exists()

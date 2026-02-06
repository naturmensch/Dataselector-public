from __future__ import annotations

from dataselector.workflows.monitor_state import ExperimentStateAnalyzer


def test_experiment_state_analyzer_basic(tmp_path):
    """ExperimentStateAnalyzer should detect CSV counts and best value."""
    run_dir = tmp_path / "run_test"
    (run_dir / "results").mkdir(parents=True)

    csv = run_dir / "results" / "trials.csv"
    csv.write_text(
        "trial_number,state,value,datetime_complete\n"
        "0,COMPLETE,1.0,2026-01-01T00:00:00Z\n"
        "1,FAIL,0.5,\n"
        "2,COMPLETE,0.9,2026-01-01T01:00:00Z\n"
    )

    analyzer = ExperimentStateAnalyzer(run_dir)
    state = analyzer.inspect()

    assert state["csv_exists"] is True
    assert state["csv_completed"] == 2
    assert state["csv_best"] == 1.0
    assert state["db_exists"] is False

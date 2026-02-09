from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataselector.workflows.generate_reports import _generate_single_run_thesis_report


def _write_minimal_artifacts(run_dir: Path, *, n_selected_values: list[int]) -> None:
    optuna_dir = run_dir / "optuna" / "results"
    pareto_dir = run_dir / "tuning_weights" / "pareto"
    validation_dir = run_dir / "validation"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"trial_number": 0, "state": "TrialState.COMPLETE", "value": 1.23},
        ]
    ).to_csv(optuna_dir / "trials.csv", index=False)

    (optuna_dir / "best_trial.json").write_text(
        json.dumps({"params": {"a": 0.2, "b": 0.3, "c": 0.5}}),
        encoding="utf-8",
    )

    pd.DataFrame([{"alpha": 0.2, "beta": 0.3, "gamma": 0.5}]).to_csv(
        pareto_dir / "pareto_solutions.csv",
        index=False,
    )

    pd.DataFrame({"n_selected": n_selected_values}).to_csv(
        validation_dir / "validation_results.csv",
        index=False,
    )


def test_report_adds_diagnostic_hint_for_zero_non_empty(tmp_path: Path):
    run_dir = tmp_path / "run_zero_non_empty"
    _write_minimal_artifacts(run_dir, n_selected_values=[0, 0, 0])

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T0")
    report = report_file.read_text(encoding="utf-8")

    assert "- Configurations with non-empty selection: **0**" in report
    assert "Diagnostic hint" in report
    assert "does not automatically mean exploration/optuna failed globally" in report


def test_report_omits_diagnostic_hint_when_non_empty_exists(tmp_path: Path):
    run_dir = tmp_path / "run_has_non_empty"
    _write_minimal_artifacts(run_dir, n_selected_values=[0, 2, 0])

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T1")
    report = report_file.read_text(encoding="utf-8")

    assert "- Configurations with non-empty selection: **1**" in report
    assert "Diagnostic hint" not in report

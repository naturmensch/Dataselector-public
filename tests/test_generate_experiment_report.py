import os
from pathlib import Path

import pytest
import yaml

from dataselector.workflows.generate_reports import generate_experiment_report


def make_fake_run_dir(tmpdir: Path):
    d = tmpdir / "run_fake"
    d.mkdir()
    # create a fake log
    (d / "coarse_sweep.log").write_text("coarse step log\nline1\n")
    (d / "fine_sweep.log").write_text("fine step log\nline1\n")
    # create a fake optuna config
    opt_cfg = {
        "selection": {"alpha_visual": 0.5, "beta_spatial": 0.3, "gamma_temporal": 0.2}
    }
    (d / "pipeline_config.optuna.yaml").write_text(yaml.safe_dump(opt_cfg))
    # create a csv
    (d / "coarse_sweep_results.csv").write_text("n_selected,temporal_std\n34,5.0\n")
    return d


def test_report_generation(tmp_path):
    run_dir = make_fake_run_dir(tmp_path)
    generate_experiment_report(run_dir)
    report = run_dir / "experiment_report.md"
    assert report.exists()
    text = report.read_text()
    assert "Experiment Report" in text
    # includes file list and YAML snippet
    assert "pipeline_config.optuna.yaml" in text


@pytest.mark.smoke
def test_report_finds_local_pareto(tmp_path):
    # Use env var to point report generator to a temp outputs root (no moving/copying repo outputs)
    outputs_root = tmp_path / "outputs"
    prev = os.environ.get("DATASELECTOR_OUTPUTS_ROOT")
    os.environ["DATASELECTOR_OUTPUTS_ROOT"] = str(outputs_root)

    # Create LHS pareto under tuning_weights and an experiments run dir
    pareto_dir = outputs_root / "tuning_weights" / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    pareto_csv = pareto_dir / "pareto_solutions.csv"
    pareto_csv.write_text("id,min_distance_km\na,35\n")

    run_dir = outputs_root / "experiments" / "run_local_pareto"
    run_dir.mkdir(parents=True, exist_ok=True)

    generate_experiment_report(run_dir)
    report = run_dir / "experiment_report.md"
    assert report.exists()
    text = report.read_text()
    assert "pareto_solutions.csv" in text
    # ensure it lists a file under tuning_weights
    assert "tuning_weights" in text

    # restore env
    if prev is None:
        os.environ.pop("DATASELECTOR_OUTPUTS_ROOT", None)
    else:
        os.environ["DATASELECTOR_OUTPUTS_ROOT"] = prev

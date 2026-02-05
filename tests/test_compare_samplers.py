"""Tests for dataselector.workflows.compare_samplers module."""

import pandas as pd
import pytest


def test_run_single_optuna_missing_run_dir(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    # Patch repo root to tmp_path
    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)

    monkeypatch.setattr(compare_samplers, "_get_run_optuna", lambda: (lambda **kwargs: None))

    with pytest.raises(FileNotFoundError, match="No run dir found"):
        compare_samplers.run_single_optuna(
            sampler="cmaes",
            seed=42,
            n_trials=10,
            n_candidates=100,
            preselection_flag=None,
            exp_desc="desc",
            dataset="hamburg",
        )


def test_run_single_optuna_success(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)

    monkeypatch.setattr(compare_samplers, "_get_run_optuna", lambda: (lambda **kwargs: None))

    exp_name = "hamburg_cmaes_10trials_s42"
    run_dir = tmp_path / "outputs" / "runs" / exp_name
    (run_dir / "results").mkdir(parents=True)

    df = pd.DataFrame({"trial_number": range(10), "value": range(10)})
    df.to_csv(run_dir / "results" / "trials.csv", index=False)

    res = compare_samplers.run_single_optuna(
        sampler="cmaes",
        seed=42,
        n_trials=10,
        n_candidates=100,
        preselection_flag=None,
        exp_desc="desc",
        dataset="hamburg",
    )

    assert res["n_trials"] == 10
    assert res["best_value"] == 9.0
    assert res["run_dir"] == str(run_dir)


def test_compare_multi_seed_creates_summary(tmp_path, monkeypatch):
    from dataselector.workflows import compare_samplers

    monkeypatch.setattr(compare_samplers, "_get_repo_root", lambda: tmp_path)

    # Prepare a fake run_dir with trials.csv
    run_dir = tmp_path / "outputs" / "runs" / "full_qmc_1trials_s1"
    (run_dir / "results").mkdir(parents=True)
    df = pd.DataFrame({"trial_number": [0, 1], "value": [0.1, 0.2]})
    df.to_csv(run_dir / "results" / "trials.csv", index=False)

    def fake_run_single_optuna(*args, **kwargs):
        return {
            "sampler": "qmc",
            "seed": 1,
            "n_trials": 2,
            "best_value": 0.2,
            "best_trial": 1,
            "mean_value": 0.15,
            "std_value": 0.05,
            "convergence_trial": 1,
            "convergence_ratio": 0.5,
            "run_dir": str(run_dir),
            "exp_desc": "desc",
            "preselection_flag": None,
        }

    monkeypatch.setattr(compare_samplers, "run_single_optuna", fake_run_single_optuna)

    out_dir = tmp_path / "outputs" / "runs" / "sampler_test"
    result = compare_samplers.compare_multi_seed(
        samplers=["qmc"],
        seeds=[1],
        n_trials=1,
        n_candidates=10,
        datasets=["full"],
        output=str(out_dir),
    )

    summary = out_dir / "summary.csv"
    assert summary.exists()
    assert isinstance(result, dict)

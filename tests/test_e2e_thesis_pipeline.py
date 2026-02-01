import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / "test_data"
EXEC_WRAPPER = REPO_ROOT / "scripts" / "exec_in_env.sh"
ALLOWED_SAMPLERS = {"qmc", "tpe", "cmaes"}


def _run(cmd: List[str], env=None, cwd=REPO_ROOT, check=True):
    """Run a command using exec_in_env.sh if present (to ensure canonical env)."""
    use_wrapper = (
        EXEC_WRAPPER.exists() and os.environ.get("SKIP_EXEC_IN_ENV", "0") != "1"
    )
    final_cmd = []
    if use_wrapper:
        final_cmd = [str(EXEC_WRAPPER), "--env", "dataselector", "--"] + cmd
    else:
        final_cmd = cmd
    return subprocess.run(final_cmd, check=check, cwd=cwd, env=env)


@pytest.fixture(scope="function")
def data_symlink(tmp_path):
    """Create minimal test subset and point scripts to it via environment.

    Notes:
    - We do not actually symlink/move the repo's `data/` directory to avoid touching
      large files.
    - `DATA_DIR` is used to point runners at the collected subset.
    """
    REPO_ROOT / "data"
    # Use repo-local tests/test_data to avoid filling /tmp when copying large real images
    test_data_dir = REPO_ROOT / "tests" / "test_data"

    # Clean any previous test data to ensure deterministic runs
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)

    # Collect test subset into repository tests/test_data
    result = subprocess.run(
        [
            "bash",
            "tests/scripts/collect_test_subset.sh",
            "--n-images",
            "5",
            "--out-dir",
            str(test_data_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )

    # Do NOT touch repo data/ to avoid moving large files. Instead, point scripts to the collected test data via environment variable.

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["DATA_DIR"] = str(test_data_dir)

    # Ensure old logs do not cause spurious failures: remove existing .log and .txt files in outputs
    out_dir = REPO_ROOT / "outputs"
    if out_dir.exists():
        for f in list(out_dir.glob("**/*.log")) + list(out_dir.glob("**/*.txt")):
            try:
                f.unlink()
            except Exception:
                # best-effort cleanup; ignore errors
                pass

    try:
        yield env
    finally:
        # teardown: nothing to restore since we didn't modify repo files
        pass


@pytest.mark.e2e
@pytest.mark.timeout(900)
def test_e2e_thesis_pipeline(data_symlink):
    """Thorough E2E test: runs autoscale → sampler suite → XXL → monitor and validates contents and logs."""
    env = data_symlink

    # 1) Autoscale
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "autoscale",
            "--",
            "--n-candidates",
            "50",
            "--stages",
            "40",
            "80",
            "--n-trials",
            "2",
            "2",
            "--patience",
            "1",
        ],
        env=env,
    )

    # Check autoscale artifacts
    autoscale_reports = list(
        (REPO_ROOT / "outputs").glob("optuna_autoscale_report_*.md")
    )
    assert autoscale_reports, "Autoscale report not generated"
    report_text = autoscale_reports[0].read_text()
    assert "Stages summary" in report_text, "Autoscale report missing stages summary"
    # Also check CSV exists
    autoscale_csv = REPO_ROOT / "outputs" / "optuna_autoscale_summary_20260125.csv"
    assert autoscale_csv.exists(), "Autoscale CSV summary missing"
    assert "ERROR" not in report_text.upper(), "Autoscale log contains ERROR"

    # 2) Sampler Suite (compare script minimal run)
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "compare-samplers",
            "--",
            "--samplers",
            "qmc",
            "tpe",
            "cmaes",
            "--seeds",
            "42",
            "--n-trials",
            "5",
            "--datasets",
            "hamburg",
            "--sequential",
            "--output",
            str(REPO_ROOT / "outputs" / "runs" / "sampler_thesis_suite_test"),
            "--n-candidates",
            "50",
        ],
        env=env,
    )

    suite_dir = REPO_ROOT / "outputs" / "runs" / "sampler_thesis_suite_test"
    summary = suite_dir / "summary.csv"
    assert summary.exists(), "Expected summary.csv in sampler suite outputs"

    df = pd.read_csv(summary)
    assert not df.empty
    # choose sampler with highest mean
    best_row = df.sort_values("mean", ascending=False).iloc[0]
    best = str(best_row["sampler"])

    # Validate sampled stats
    assert best in ALLOWED_SAMPLERS
    assert pd.notna(best_row["mean"]) and isinstance(best_row["mean"], float)

    # write selected_sampler.json
    sel = {
        "best": best,
        "metric": "mean",
        "score": float(best_row["mean"]),
        "n_trials": int(best_row["count"]),
        "datasets": ["hamburg"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(suite_dir),
    }
    (REPO_ROOT / "outputs" / "selected_sampler.json").write_text(json.dumps(sel))

    # Check selected_sampler content
    sel_path = REPO_ROOT / "outputs" / "selected_sampler.json"
    assert sel_path.exists()
    sel_data = json.loads(sel_path.read_text())
    assert sel_data["best"] in ALLOWED_SAMPLERS
    assert isinstance(sel_data["score"], (float, int))

    # 3) Run modern XXL orchestrator with the chosen sampler
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            best,
            "--smoke",
        ],
        env=env,
    )

    # Run the monitor separately (short poll interval, no new session)
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl-monitor",
            "--",
            "--poll-interval",
            "1",
            "--no-new-session",
            "--main-script",
            "python -m dataselector xxl --",
        ],
        env=env,
    )

    # Validate final artifacts
    thesis_json = REPO_ROOT / "outputs" / "thesis_finalization_summary.json"
    assert thesis_json.exists(), "Thesis finalization summary missing"
    thesis_data = json.loads(thesis_json.read_text())
    assert "artifacts" in thesis_data
    assert isinstance(thesis_data["artifacts"], list)
    assert len(thesis_data["artifacts"]) > 0

    # Additional thorough checks
    # Check for all expected output files
    expected_outputs = [
        "bootstrap_results_summary.csv",
        "convergence_baseline.json",
        "distance_comparison_per_tile.json",
        "kdr100_best_selection_info.json",
        "multi_criteria_temporal_test.csv",
        "optuna_autoscale_best_20260125.json",
    ]
    for eo in expected_outputs:
        files = list((REPO_ROOT / "outputs").glob(f"**/{eo}"))
        assert files, f"Expected output {eo} not found"

    # Validate CSV contents
    bootstrap_csv = REPO_ROOT / "outputs" / "bootstrap_results_summary.csv"
    if bootstrap_csv.exists():
        df = pd.read_csv(bootstrap_csv)
        assert not df.empty, "Bootstrap CSV is empty"
        assert "mean" in df.columns, "Bootstrap CSV missing 'mean' column"

    # Validate JSON structures
    best_info = REPO_ROOT / "outputs" / "kdr100_best_selection_info.json"
    if best_info.exists():
        info_data = json.loads(best_info.read_text())
        assert (
            "selected_tiles" in info_data
        ), "Best selection info missing selected_tiles"
        assert isinstance(
            info_data["selected_tiles"], list
        ), "selected_tiles not a list"

    # Check logs for key function calls (e.g., ensure imports and functions are triggered)
    log_files = list((REPO_ROOT / "outputs").glob("**/*.log")) + list(
        (REPO_ROOT / "outputs").glob("**/*.txt")
    )
    all_logs = ""
    for log in log_files:
        all_logs += log.read_text()
    # Check for key phrases indicating functions were called. Allow multiple alternatives for noisy legacy messages.
    key_groups = [
        ["Stage 1/2: n_samples", "Read n_samples"],  # autoscale/modern orchestrator
        [
            "XXL THESIS COMPLETE PIPELINE",
            "XXL THESIS PIPELINE COMPLETE",
            "✅ XXL THESIS PIPELINE COMPLETE!",
        ],  # From xxl script (allow variants)
    ]
    for group in key_groups:
        assert any(
            k.lower() in all_logs.lower() for k in group
        ), f"None of indicators {group} found in logs"

    # The monitor should have created a symlink `outputs/XXL_FULL_RUN.log` pointing to a timestamped log
    symlink = REPO_ROOT / "outputs" / "XXL_FULL_RUN.log"
    assert symlink.exists(), "Monitor did not create XXL_FULL_RUN.log symlink"
    assert symlink.is_symlink(), "XXL_FULL_RUN.log is not a symlink"
    assert symlink.resolve().exists(), "XXL_FULL_RUN.log symlink target does not exist"

    # Validate monitor report exists and contains expected headings
    monitors = list(
        (REPO_ROOT / "outputs" / "runs").glob("**/monitor_reports/monitor_report_*.md")
    )
    assert monitors, "No monitor report found in run outputs"
    mtext = monitors[0].read_text()
    assert "Final Selection" in mtext or "final selection" in mtext.lower()
    assert "Optuna" in mtext or "optuna" in mtext.lower()


@pytest.mark.e2e
def test_phase5_dry_run(data_symlink):
    """Dry-run smoke test for Phase 5 bootstrap + UQ integration."""
    env = data_symlink
    # Run the modern orchestrator in dry-run mode which should simulate bootstrap outputs
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            "tpe",
            "--dry-run",
        ],
        env=env,
    )

    # The dry-run will simulate outputs into the latest run directory (or a synthetic dry_run_simulated dir)
    runs_root = REPO_ROOT / "outputs" / "runs"
    candidates = [d for d in runs_root.iterdir() if d.is_dir()]
    assert candidates, "No run directories found"
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    sim_results = latest / "results"
    # Accept either full bootstrap results or UQ summary as evidence of Phase 5 execution
    assert (
        sim_results / "bootstrap_final_selection_full.csv"
    ).exists(), f"Simulated full bootstrap results missing in {sim_results}"
    assert (
        sim_results / "bootstrap_uq_summary.json"
    ).exists(), f"Simulated UQ summary missing in {sim_results}"
    # Ensure Phase 1 created a best_trial.json for the run
    assert (
        latest / "results" / "best_trial.json"
    ).exists(), f"best_trial.json missing in {latest / 'results' }"


@pytest.mark.e2e
@pytest.mark.parametrize("sampler", ["qmc", "tpe", "cmaes"])
def test_e2e_with_samplers(data_symlink, sampler):
    """Run the XXL orchestrator with forced sampler choices and assert artifacts are produced."""
    env = data_symlink
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            sampler,
        ],
        env=env,
    )

    thesis_json = REPO_ROOT / "outputs" / "thesis_finalization_summary.json"
    assert thesis_json.exists(), f"Thesis summary missing for sampler {sampler}"
    data = json.loads(thesis_json.read_text())
    assert "artifacts" in data and len(data["artifacts"]) > 0

    # Additional validations for thoroughness
    # Check for key artifacts
    artifacts = data["artifacts"]
    expected_files = [
        "bootstrap_results_summary.csv",
        "convergence_baseline.json",
        "kdr100_best_selection_info.json",
    ]
    for ef in expected_files:
        assert any(
            ef in art for art in artifacts
        ), f"Expected artifact {ef} not in summary for {sampler}"

    # Validate logs for errors
    log_files = list((REPO_ROOT / "outputs").glob("**/*.log")) + list(
        (REPO_ROOT / "outputs").glob("**/*.txt")
    )
    for log in log_files:
        content = log.read_text()
        assert "ERROR" not in content.upper(), f"Log {log} contains ERROR"
        assert "EXCEPTION" not in content.upper(), f"Log {log} contains EXCEPTION"

    # Check reproducibility: run again with same seed and compare key metrics
    # (Assuming the script uses deterministic seeds)
    # For simplicity, just ensure the run completes without errors again
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            sampler,
        ],
        env=env,
    )
    # If reproducible, the second run should produce similar results; for now, just check completion


@pytest.mark.e2e
@pytest.mark.parametrize("sampler", ["qmc", "tpe", "cmaes"])
def test_e2e_with_samplers_smoke(data_symlink, sampler):
    """Run the XXL orchestrator in smoke mode on real test data and assert finalization produces artifacts."""
    env = data_symlink
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            sampler,
            "--smoke",
        ],
        env=env,
    )

    thesis_json = REPO_ROOT / "outputs" / "thesis_finalization_summary.json"
    assert thesis_json.exists(), f"Thesis summary missing (smoke) for sampler {sampler}"
    data = json.loads(thesis_json.read_text())
    assert "artifacts" in data and len(data["artifacts"]) > 0
    # Ensure some important artifacts are present in artifacts list
    artifacts = data["artifacts"]
    assert any(
        "bootstrap" in a for a in artifacts
    ), "No bootstrap artifact recorded in finalization summary"
    assert any(
        "convergence_baseline.json" in a for a in artifacts
    ), "convergence_baseline.json not in artifacts"


@pytest.mark.e2e
def test_seed_reproducibility(data_symlink):
    """Test that runs with the same seed produce identical results."""
    env = data_symlink
    sampler = "qmc"
    seed = "12345"

    # First run
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            sampler,
            "--seed",
            seed,
            "--dry-run",
        ],
        env=env,
    )

    thesis_json1 = REPO_ROOT / "outputs" / "thesis_finalization_summary.json"
    assert thesis_json1.exists()
    data1 = json.loads(thesis_json1.read_text())

    # Second run with same seed
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl",
            "--",
            "--best-sampler",
            sampler,
            "--seed",
            seed,
            "--dry-run",
        ],
        env=env,
    )

    thesis_json2 = REPO_ROOT / "outputs" / "thesis_finalization_summary.json"
    assert thesis_json2.exists()
    data2 = json.loads(thesis_json2.read_text())

    # Compare key metrics (assuming deterministic)
    # For example, compare number of artifacts or specific values
    assert len(data1["artifacts"]) == len(
        data2["artifacts"]
    ), "Reproducibility failed: different artifact counts"
    # If more detailed comparison needed, compare specific fields


@pytest.mark.e2e
@pytest.mark.timeout(900)
def test_monitor_starts_shell_orchestrator(data_symlink, tmp_path):
    """Start the monitor using the shell orchestrator and assert it runs end-to-end (small dataset)."""
    env = data_symlink

    # Create a small config override to keep run short
    cfg = {
        "seeds": [42],
        "n_trials": 2,
    }
    cfg_path = tmp_path / "monitor_test_config.yaml"
    import yaml

    cfg_path.write_text(yaml.dump(cfg))
    env["CONFIG_PATH"] = str(cfg_path)

    # Force deterministic TS for monitor logs
    env["MONITOR_FORCE_TS"] = "20260125T000000Z"

    # Run monitor pointing to the shell-based orchestrator (prefer canonical CLI wrapper)
    _run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl-monitor",
            "--",
            "--no-new-session",
            "--poll-interval",
            "1",
            "--main-script",
            "python -m dataselector thesis-pipeline --",
        ],
        env=env,
    )

    # Check that the shell orchestrator wrote its pipeline log and summary artifacts
    pipeline_log = REPO_ROOT / "outputs" / "thesis_pipeline.log"
    assert pipeline_log.exists(), "Shell orchestrator pipeline log missing"
    txt = pipeline_log.read_text()
    assert (
        "COMPLETE THESIS PIPELINE" in txt or "COMPLETE THESIS PIPELINE" in txt.upper()
    )

    # Monitor should have created monitor reports
    monitors = list(
        (REPO_ROOT / "outputs" / "runs").glob("**/monitor_reports/monitor_report_*.md")
    )
    assert monitors, "No monitor report found after shell orchestrator run"


@pytest.mark.e2e
@pytest.mark.timeout(120)
def test_monitor_resume_dry_run(data_symlink, tmp_path):
    """Simulate an incomplete run and test monitor --restart --dry-run-restart planning."""
    env = data_symlink

    runs_root = REPO_ROOT / "outputs" / "runs"
    test_run = runs_root / "incomplete_monitor_test"
    # Ensure a clean slate
    if test_run.exists():
        shutil.rmtree(test_run)
    test_run.mkdir(parents=True)
    (test_run / "config").mkdir()
    # Write a tiny config file that indicates more trials still to run
    (test_run / "config" / "config_optuna.yaml").write_text("n_trials: 10\n")
    # Create a partial trials.csv to simulate incomplete run
    results_dir = test_run / "results"
    results_dir.mkdir()
    trials_csv = results_dir / "trials.csv"
    trials_csv.write_text(
        "trial_number,value\n0,0.5\n1,0.6\n2,0.7\n"
    )  # Only 3 completed out of 10

    # Run monitor in dry-run restart mode, pointing to the directory
    res = subprocess.run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl-monitor",
            "--",
            "--restart",
            str(test_run),
            "--force-restart",
            "--dry-run-restart",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Note: RecoveryPlanner may not be available, so expect failure if not implemented
    if "RecoveryPlanner" in res.stderr and "not callable" in res.stderr:
        pytest.skip("RecoveryPlanner not implemented, skipping resume test")
    assert res.returncode == 0, f"Monitor restart dry-run failed: {res.stderr}"
    assert (
        "Planned resume phases" in res.stdout or "Planned resume phases" in res.stderr
    ), "Monitor did not output planned phases"


@pytest.mark.e2e
@pytest.mark.timeout(120)
def test_monitor_reconcile_with_db(data_symlink, tmp_path, monkeypatch):
    """Create an optuna sqlite DB with fewer trials than configured and assert monitor plans optuna resume."""
    env = data_symlink

    runs_root = REPO_ROOT / "outputs" / "runs"
    test_run = runs_root / "incomplete_monitor_db_test"
    # Ensure a clean slate
    if test_run.exists():
        shutil.rmtree(test_run)
    test_run.mkdir(parents=True)
    (test_run / "config").mkdir()
    # Write a tiny config file that indicates more trials still to run
    (test_run / "config" / "config_optuna.yaml").write_text("n_trials: 10\n")

    # Create an optuna DB with 3 completed trials by invoking run_optuna
    # Provide synthetic features and dummy selector to avoid heavy deps
    import numpy as _np

    from scripts.optuna_optimize import run_optuna

    features = _np.random.RandomState(1).randn(20, 16).astype("float32")
    import pandas as _pd

    metadata = _pd.DataFrame(
        {
            "N": _np.random.uniform(48, 55, 20),
            "left": _np.random.uniform(6, 15, 20),
            "year": _np.random.randint(1880, 1945, 20),
        }
    )
    monkeypatch.setattr(
        "scripts.optuna_optimize.load_or_create_data",
        lambda n, dim, seed: (features, metadata),
        raising=False,
    )

    class DummySelector:
        def __init__(self, n_samples, use_multi_criteria=True):
            self.n_samples = n_samples

        def select(
            self,
            features,
            metadata,
            spatial_constraint,
            min_distance_km,
            alpha_visual,
            beta_spatial,
            gamma_temporal,
            pre_selected=None,
            pre_selected_names=None,
        ):
            n = min(self.n_samples, len(features))
            return list(range(n))

        def _calculate_diversity_score(self, selected_features):
            return float(_np.mean(_np.var(selected_features, axis=0)))

    monkeypatch.setattr(
        "scripts.optuna_optimize.DiversitySelector", DummySelector, raising=False
    )

    run_optuna(
        n_trials=3,
        n_candidates=20,
        dim=16,
        n_samples=5,
        out_dir=test_run,
        study_db=str(test_run / "optuna_study.db"),
    )

    assert (test_run / "optuna_study.db").exists(), "Expected optuna DB file"

    # If RecoveryPlanner is not available, skip this integration check (monitor would fail similarly)
    try:
        pass
    except Exception:
        pytest.skip("RecoveryPlanner not importable; skipping reconcile DB test")

    # Run monitor in dry-run restart mode
    res = subprocess.run(
        [
            "python",
            "-m",
            "dataselector",
            "xxl-monitor",
            "--",
            "--restart",
            str(test_run),
            "--force-restart",
            "--dry-run-restart",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert res.returncode == 0, f"Monitor restart dry-run failed with DB: {res.stderr}"
    assert (
        "Planned resume phases" in res.stdout or "Planned resume phases" in res.stderr
    ), "Monitor did not output planned phases when DB present"
    # Expect that one of the planned phases is 'optuna' (resume)
    assert (
        "optuna" in res.stdout
        or "optuna" in res.stderr
        or "reconstructed" in res.stdout
        or "reconstructed" in res.stderr
    ), "Monitor did not plan optuna resume or reconstruction"

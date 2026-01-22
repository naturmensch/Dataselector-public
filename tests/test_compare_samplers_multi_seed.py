from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._helpers.load_script import load_script


@pytest.fixture(scope="module")
def compare_mod():
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(ROOT / "scripts" / "compare_samplers_multi_seed.py", module_name="scripts.compare_samplers_multi_seed")


@pytest.fixture()
def run_single_optuna(compare_mod):
    return compare_mod.run_single_optuna


def test_run_single_optuna_no_run_dir_includes_subprocess_output(tmp_path, monkeypatch, run_single_optuna):
    """If the subprocess completes but no run dir is created, the FileNotFoundError should include stdout/stderr."""
    # Ensure outputs/runs does not contain the expected run
    monkeypatch.setattr("scripts.compare_samplers_multi_seed.ROOT", tmp_path)

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "Some output\nModuleNotFoundError: No module named 'optuna'\n"
    fake_proc.stderr = "Error details"

    with patch("subprocess.run", return_value=fake_proc):
        with pytest.raises(FileNotFoundError) as excinfo:
            run_single_optuna("cmaes", 42, 1000, 673, None, "exp", dataset="hamburg")

        msg = str(excinfo.value)
        assert "No run dir found" in msg
        assert "ModuleNotFoundError" in msg or "No module named" in msg
        assert "Subprocess stdout" in msg and "Subprocess stderr" in msg


def test_run_single_optuna_success(tmp_path, monkeypatch, run_single_optuna):
    """Test successful run with mocked subprocess and file system."""
    # Mock ROOT to point to tmp_path (patch both the loaded module and any existing package-loaded module)
    monkeypatch.setattr(compare_mod, "ROOT", tmp_path)
    monkeypatch.setattr("scripts.compare_samplers_multi_seed.ROOT", tmp_path, raising=False)
    monkeypatch.setattr("scripts.compare_samplers_multi_seed.ROOT", tmp_path, raising=False)

    # Setup dummy run dir structure
    exp_name = "hamburg_cmaes_10trials_s42"
    run_dir = tmp_path / "outputs" / "runs" / exp_name
    (run_dir / "results").mkdir(parents=True)

    # Create dummy trials.csv
    df = pd.DataFrame({"trial_number": range(10), "value": range(10)})
    df.to_csv(run_dir / "results" / "trials.csv", index=False)

    # Mock subprocess to return success
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        res = run_single_optuna("cmaes", 42, 10, 100, None, "desc", dataset="hamburg")

        assert res["n_trials"] == 10
        assert res["best_value"] == 9.0
        assert res["run_dir"] == str(run_dir)


def test_run_single_optuna_missing_trials_csv(tmp_path, monkeypatch, run_single_optuna):
    """Test that missing trials.csv raises FileNotFoundError after retries."""
    monkeypatch.setattr("scripts.compare_samplers_multi_seed.ROOT", tmp_path)

    exp_name = "hamburg_cmaes_10trials_s42"
    run_dir = tmp_path / "outputs" / "runs" / exp_name
    run_dir.mkdir(parents=True)  # Run dir exists, but results/trials.csv does not

    with patch("subprocess.run") as mock_run, patch("time.sleep"):  # skip sleep delay
        mock_run.return_value.returncode = 0

        with pytest.raises(FileNotFoundError, match="trials.csv missing"):
            run_single_optuna("cmaes", 42, 10, 100, None, "desc", dataset="hamburg")

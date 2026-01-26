import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.e2e
def test_run_pipeline_smoke(tmp_path):
    """Run `run_pipeline.py` in smoke mode and ensure it completes and writes outputs."""
    # Create a minimal workspace layout that uses real test data
    ws = tmp_path / "workspace"
    data = ws / "data"
    out = ws / "outputs"
    data.mkdir(parents=True)
    out.mkdir(parents=True)

    # Use the repo test data (small real subset) instead of synthetic CSV
    src_csv = REPO_ROOT / "tests" / "test_data" / "new_all_tiles.csv"
    dst_csv = data / "new_all_tiles.csv"
    dst_csv.write_text(src_csv.read_text())

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"), "--smoke", "--workspace", str(ws), "--tune"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"run_pipeline smoke failed: {res.stdout}\n{res.stderr}"

    # Check that tuning outputs exist inside workspace
    tune_dir = ws / "outputs" / "tuning_weights"
    assert tune_dir.exists()
    assert (tune_dir / "tuning_results.csv").exists()

@pytest.mark.e2e
def test_optuna_optimize_smoke(tmp_path):
    ws = tmp_path / "workspace2"
    data = ws / "data"
    out = ws / "outputs"
    data.mkdir(parents=True)
    out.mkdir(parents=True)

    # Use real test metadata: copy tests/test_data/new_all_tiles.csv into the workspace outputs as metadata.csv expected by optuna_optimize
    src_meta = REPO_ROOT / "tests" / "test_data" / "new_all_tiles.csv"
    out_meta = out / "metadata.csv"
    out_meta.write_text(src_meta.read_text())

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "optuna_optimize.py"), "--smoke", "--workspace", str(ws), "--n-trials", "2"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"optuna_optimize smoke failed: {res.stdout}\n{res.stderr}"

    # Check for checkpoint file or results presence
    chk = list((ws / "outputs").glob("optuna_results_checkpoint_*.csv"))
    # Either checkpoint created or results.csv exists
    assert chk or (ws / "outputs" / "optuna_results.csv").exists()

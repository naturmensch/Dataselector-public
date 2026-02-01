import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_run_adaptive_pipeline_seed_propagation(tmp_path):
    # Run a minimal dry-run adaptive pipeline to create experiment metadata
    cmd = [
        "PYTHONPATH=.",
        "python",
        "scripts/run_adaptive_pipeline.py",
        "--dry-run",
        "--yes",
        "--n-lhs",
        "2",
        "--n-trials",
        "2",
        "--n-boot",
        "1",
        "--seed",
        "12345",
    ]
    # Execute command
    # Ensure we run against a clean outputs/runs to avoid picking up earlier runs
    import shutil

    out_runs = Path("outputs/runs")
    if out_runs.exists():
        shutil.rmtree(out_runs)
    subprocess.check_call(" ".join(cmd), shell=True)

    # Find latest run dir
    run_dirs = sorted(Path("outputs/runs").glob("*_adaptive_full"))
    assert run_dirs, "No adaptive run dirs found"
    run_dir = run_dirs[-1]

    # Check that run config contains the seed
    cfg = run_dir / "config" / "config_run.yaml"
    assert cfg.exists(), f"Config missing: {cfg}"
    text = cfg.read_text()
    assert "seed: 12345" in text, f"Seed not propagated in config: {text}"

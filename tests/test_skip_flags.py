import subprocess

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_skip_optuna_and_skip_flags_dryrun(tmp_path):
    # Ensure skip-optuna is honored by run_adaptive_pipeline
    cmd = "PYTHONPATH=. python -m scripts.run_adaptive_pipeline --dry-run --yes --skip-optuna --n-lhs 2 --n-trials 2 --n-boot 1"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    assert (
        "Skipping Optuna stage" in out or "Skipping Optuna stage" in out
    ), "skip-optuna not honored"

    # Ensure skip-exploration prevents exploration cmd invocation
    cmd2 = "PYTHONPATH=. python -m scripts.run_adaptive_pipeline --dry-run --yes --skip-exploration --n-trials 2 --n-boot 1"
    proc2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    out2 = proc2.stdout + proc2.stderr
    assert "Exploration SKIPPED" in out2, "--skip-exploration not honored"

    # Ensure skip-fine prevents fine sweep execution
    cmd3 = "PYTHONPATH=. python -m scripts.run_adaptive_pipeline --dry-run --yes --skip-fine --n-lhs 2 --n-trials 2"
    proc3 = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
    out3 = proc3.stdout + proc3.stderr
    assert (
        "Fine Sweep SKIPPED" in out3 or "Skipping fine sweep execution" in out3
    ), "--skip-fine not honored"

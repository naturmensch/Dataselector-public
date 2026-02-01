import sys

import pytest

pytest.mark.integration


@pytest.fixture(scope="module")
def run_pipe():
    pytest.importorskip("numba", exc_type=ImportError)
    import importlib

    return importlib.import_module("scripts.run_adaptive_pipeline")


@pytest.fixture(autouse=True)
def mock_run_cmd(monkeypatch, run_pipe):
    """Prevent actual subprocess execution in all tests in this module."""
    monkeypatch.setattr(
        run_pipe, "run_cmd", lambda cmd: print(f"MOCKED run_cmd: {cmd}")
    )


def test_optuna_failure_aborts(monkeypatch, run_pipe):
    # Patch run_cmd to raise an exception simulating subprocess failure
    def raise_error(cmd):
        raise Exception("subprocess fail")

    monkeypatch.setattr(run_pipe, "run_cmd", raise_error)
    monkeypatch.setattr(sys, "argv", ["run_adaptive_pipeline.py", "--yes"])
    # Ensure we do not continue on failure
    with pytest.raises(SystemExit):
        run_pipe.main()


def test_optuna_continue_on_failure(monkeypatch, run_pipe):
    # Patch run_cmd to raise, but set continue flag so pipeline does not abort
    def raise_error(cmd):
        raise Exception("subprocess fail")

    monkeypatch.setattr(run_pipe, "run_cmd", raise_error)
    # Simulate args by patching sys.argv
    monkeypatch.setattr(
        "sys.argv",
        ["run_adaptive_pipeline.py", "--yes", "--continue-on-analysis-failure"],
    )
    # Should complete without raising SystemExit
    run_pipe.main()

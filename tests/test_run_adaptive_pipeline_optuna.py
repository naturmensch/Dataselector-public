import pytest
from unittest.mock import patch, MagicMock
import sys
import scripts.run_adaptive_pipeline as run_pipe

@pytest.fixture(autouse=True)
def mock_run_cmd(monkeypatch):
    """Prevent actual subprocess execution in all tests in this module."""
    monkeypatch.setattr(run_pipe, 'run_cmd', lambda cmd: print(f"MOCKED run_cmd: {cmd}"))

def test_optuna_failure_aborts(monkeypatch):
    # Patch run_cmd to raise an exception simulating subprocess failure
    def raise_error(cmd):
        raise Exception('subprocess fail')
    monkeypatch.setattr(run_pipe, 'run_cmd', raise_error)
    monkeypatch.setattr(sys, 'argv', ['run_adaptive_pipeline.py', '--yes'])
    # Ensure we do not continue on failure
    with pytest.raises(SystemExit):
        run_pipe.main()


def test_optuna_continue_on_failure(monkeypatch):
    # Patch run_cmd to raise, but set continue flag so pipeline does not abort
    def raise_error(cmd):
        raise Exception('subprocess fail')
    monkeypatch.setattr(run_pipe, 'run_cmd', raise_error)
    # Simulate args by patching sys.argv
    monkeypatch.setattr('sys.argv', ['run_adaptive_pipeline.py', '--yes', '--continue-on-analysis-failure'])
    # Should complete without raising SystemExit
    run_pipe.main()

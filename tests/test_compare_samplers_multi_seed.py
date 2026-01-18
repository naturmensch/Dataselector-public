import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_samplers_multi_seed import run_single_optuna


def test_run_single_optuna_no_run_dir_includes_subprocess_output(tmp_path, monkeypatch):
    """If the subprocess completes but no run dir is created, the FileNotFoundError should include stdout/stderr."""
    # Ensure outputs/runs does not contain the expected run
    monkeypatch.chdir(tmp_path)

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "Some output\nModuleNotFoundError: No module named 'optuna'\n"
    fake_proc.stderr = "Error details"

    with patch('subprocess.run', return_value=fake_proc):
        with pytest.raises(FileNotFoundError) as excinfo:
            run_single_optuna('cmaes', 42, 1000, 673, None, 'exp', dataset='hamburg')

        msg = str(excinfo.value)
        assert 'No run dir found' in msg
        assert 'ModuleNotFoundError' in msg or 'No module named' in msg
        assert 'Subprocess stdout' in msg and 'Subprocess stderr' in msg

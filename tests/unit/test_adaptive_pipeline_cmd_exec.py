from __future__ import annotations

import subprocess

import pytest

from dataselector.workflows import adaptive_pipeline


def test_normalize_cmd_argv_from_string() -> None:
    argv = adaptive_pipeline._normalize_cmd_argv("python -m dataselector --help")
    assert argv == ["python", "-m", "dataselector", "--help"]


def test_normalize_cmd_argv_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        adaptive_pipeline._normalize_cmd_argv("")


def test_run_cmd_safe_raises_system_exit_on_failure(monkeypatch) -> None:
    def _fake_run(argv, check=False):
        raise subprocess.CalledProcessError(returncode=7, cmd=argv)

    monkeypatch.setattr(adaptive_pipeline.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit, match="exit code 7"):
        adaptive_pipeline.run_cmd_safe(["false"])

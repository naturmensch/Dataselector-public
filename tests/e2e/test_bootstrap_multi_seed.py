"""Integration test for bootstrap CLI contracts after hard-cut migration."""

import pytest


@pytest.mark.integration
@pytest.mark.bootstrap
def test_bootstrap_contract_migrated(run_dataselector_cli):
    """Legacy bootstrap-uq must be rejected; new bootstrap commands must exist."""
    legacy = run_dataselector_cli(["bootstrap-uq", "--help"], capture_output=True)
    assert legacy.returncode != 0
    legacy_err = (legacy.stderr or b"").decode().lower()
    assert "invalid choice" in legacy_err or "unrecognized" in legacy_err

    result_final = run_dataselector_cli(["bootstrap-final", "--help"], capture_output=True)
    assert result_final.returncode == 0

    result_pareto = run_dataselector_cli(["bootstrap-pareto", "--help"], capture_output=True)
    assert result_pareto.returncode == 0

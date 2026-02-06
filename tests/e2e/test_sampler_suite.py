"""E2E tests for sampler-suite CLI contract after migration."""

import pytest


@pytest.mark.integration
@pytest.mark.sampler
def test_sampler_suite_alias_help(run_dataselector_cli):
    """Alias command remains available and exposes help."""
    result = run_dataselector_cli(["sampler-suite", "--help"], capture_output=True)
    assert result.returncode == 0
    help_text = result.stdout.decode().lower()
    assert "sampler-suite" in help_text


@pytest.mark.integration
@pytest.mark.sampler
def test_thesis_sampler_suite_help_and_legacy_flag_rejection(run_dataselector_cli):
    """Thesis sampler suite uses new flags; legacy --csv flag must be rejected."""
    help_result = run_dataselector_cli(["thesis-sampler-suite", "--help"], capture_output=True)
    assert help_result.returncode == 0
    help_text = help_result.stdout.decode().lower()
    assert "--seeds" in help_text
    assert "--n-trials" in help_text

    legacy = run_dataselector_cli(["sampler-suite", "--csv", "data/new_all_tiles.csv"], capture_output=True)
    assert legacy.returncode != 0
    assert "unrecognized arguments" in legacy.stderr.decode().lower()

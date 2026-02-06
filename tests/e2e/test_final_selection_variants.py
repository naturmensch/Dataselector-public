"""Integration tests for final-selection CLI contract after migration."""

import pytest


@pytest.mark.integration
@pytest.mark.selection
def test_final_selection_help_shows_current_flags(run_dataselector_cli):
    """Current command advertises metadata-path based contract."""
    result = run_dataselector_cli(["final-selection", "--help"], capture_output=True)
    assert result.returncode == 0
    help_text = result.stdout.decode().lower()
    assert "--metadata-path" in help_text
    assert "--output-dir" in help_text
    assert "--n-samples" in help_text


@pytest.mark.integration
@pytest.mark.selection
def test_final_selection_rejects_legacy_csv_flag(run_dataselector_cli):
    """Legacy --csv flag is removed in CLI-only architecture."""
    result = run_dataselector_cli(
        ["final-selection", "--csv", "data/new_all_tiles.csv"],
        capture_output=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr.decode().lower()

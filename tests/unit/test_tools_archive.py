"""Unit tests for archive tool CLI decorators."""

from __future__ import annotations

from dataselector.cli_decorators import _CLI_COMMANDS
from dataselector.tools import archive


def test_verify_archive_registered():
    """Test that verify-archive command is registered."""
    assert "verify-archive" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["verify-archive"]
    assert cmd_def.func == archive.verify_archive
    assert "fail_on_reference" in cmd_def.args


def test_archive_outputs_registered():
    """Test that archive-outputs command is registered."""
    assert "archive-outputs" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["archive-outputs"]
    assert cmd_def.func == archive.archive_outputs
    assert "outputs" in cmd_def.args
    assert "dest" in cmd_def.args
    assert "exclude" in cmd_def.args
    assert cmd_def.args["outputs"].required is True


def test_list_archives_registered():
    """Test that list-archives command is registered."""
    assert "list-archives" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["list-archives"]
    assert cmd_def.func == archive.list_archives
    assert "dir" in cmd_def.args


def test_list_archives_nonexistent_dir(tmp_path):
    """Test list_archives with non-existent directory."""
    result = archive.list_archives(dir=str(tmp_path / "nonexistent"))
    assert result == 0  # Should not crash


def test_list_archives_empty_dir(tmp_path):
    """Test list_archives with empty directory."""
    result = archive.list_archives(dir=str(tmp_path))
    assert result == 0

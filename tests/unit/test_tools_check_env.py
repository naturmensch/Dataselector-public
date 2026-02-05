"""Unit tests for check-env tool CLI decorator."""

from __future__ import annotations

import pytest
from dataselector.cli_decorators import _CLI_COMMANDS
from dataselector.tools import check


def test_check_env_registered():
    """Test that check-env command is registered via @cli_command."""
    # Import triggers decorator execution
    import dataselector.tools.check as check_module
    
    assert "check-env" in _CLI_COMMANDS, "check-env should be registered"
    cmd_def = _CLI_COMMANDS["check-env"]
    assert cmd_def.func == check_module.check_env_usage
    assert cmd_def.help == "Check environment usage in scripts/CI"
    assert "paths" in cmd_def.args


def test_check_env_args():
    """Test that check-env has correct argument definition."""
    cmd_def = _CLI_COMMANDS["check-env"]
    
    paths_arg = cmd_def.args["paths"]
    assert paths_arg.type == str
    assert paths_arg.nargs == "*"
    assert paths_arg.help is not None


def test_check_env_callable():
    """Test that check_env_usage is directly callable."""
    # Should be callable without args (uses defaults)
    result = check.check_env_usage()
    assert isinstance(result, int)
    assert result in (0, 2)  # 0=ok, 2=suspicious findings


def test_check_env_with_paths():
    """Test check_env_usage with explicit paths."""
    # Non-existent paths should not crash
    result = check.check_env_usage(paths=["nonexistent/path"])
    assert isinstance(result, int)

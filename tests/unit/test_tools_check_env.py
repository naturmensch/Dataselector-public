"""Unit tests for check-env and check-protected tool CLI decorators."""

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


# ============================================================================
# check-protected tests
# ============================================================================

def test_check_protected_registered():
    """Test that check-protected command is registered via @cli_command."""
    assert "check-protected" in _CLI_COMMANDS, "check-protected should be registered"
    cmd_def = _CLI_COMMANDS["check-protected"]
    assert cmd_def.func == check.check_protected
    assert cmd_def.help == "Check for modifications inside protected paths"
    assert "list" in cmd_def.args
    assert "all" in cmd_def.args
    assert "protect" in cmd_def.args


def test_check_protected_args():
    """Test that check-protected has correct argument definitions."""
    cmd_def = _CLI_COMMANDS["check-protected"]
    
    list_arg = cmd_def.args["list"]
    assert list_arg.type == bool
    assert list_arg.action == "store_true"
    
    all_arg = cmd_def.args["all"]
    assert all_arg.type == bool
    assert all_arg.action == "store_true"
    
    protect_arg = cmd_def.args["protect"]
    assert protect_arg.type == str
    assert protect_arg.nargs == "*"


def test_check_protected_list_only():
    """Test check_protected with list=True."""
    result = check.check_protected(list=True)
    assert result == 0  # Just lists, always returns 0


def test_check_protected_empty_staged():
    """Test check_protected with empty staged files."""
    result = check.check_protected(staged_override=[])
    assert result == 0  # No files = OK


# ============================================================================
# check-geo tests
# ============================================================================

def test_check_geo_registered():
    """Test that check-geo command is registered via @cli_command."""
    assert "check-geo" in _CLI_COMMANDS, "check-geo should be registered"
    cmd_def = _CLI_COMMANDS["check-geo"]
    assert cmd_def.func == check.check_geo
    assert cmd_def.help == "Check geo dependencies (geopandas, pyproj, shapely, fiona, rtree)"
    assert len(cmd_def.args) == 0  # No arguments


def test_check_geo_callable():
    """Test that check_geo is directly callable."""
    result = check.check_geo()
    assert isinstance(result, int)
    assert result in (0, 2)  # 0=ok, 2=missing deps

"""Unit tests for remaining tool CLI decorators (audit, clean, docs_link)."""

from __future__ import annotations

from dataselector.cli_decorators import _CLI_COMMANDS
from dataselector.tools import audit, clean, docs_link

# ============================================================================
# align-audit tests
# ============================================================================


def test_align_audit_registered():
    """Test that align-audit command is registered."""
    assert "align-audit" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["align-audit"]
    assert cmd_def.func == audit.align_audit
    assert "csv" in cmd_def.args
    assert "base_dir" in cmd_def.args


# ============================================================================
# clean-workspace tests
# ============================================================================


def test_clean_workspace_registered():
    """Test that clean-workspace command is registered."""
    assert "clean-workspace" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["clean-workspace"]
    assert cmd_def.func == clean.clean_workspace
    assert "delete_outputs" in cmd_def.args
    assert "delete_cache" in cmd_def.args
    assert "yes" in cmd_def.args


def test_clean_workspace_dry_run():
    """Test clean_workspace in dry-run mode (default)."""
    result = clean.clean_workspace(yes=False)
    assert result == 0  # Should not crash in dry-run


# ============================================================================
# docs-link-check tests
# ============================================================================


def test_docs_link_check_registered():
    """Test that docs-link-check command is registered."""
    assert "docs-link-check" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["docs-link-check"]
    assert cmd_def.func == docs_link.check_links
    assert len(cmd_def.args) == 0


def test_check_links_callable():
    """Test check_links is directly callable."""
    result = docs_link.check_links()
    assert isinstance(result, int)


# ============================================================================
# docs-link-autofix tests
# ============================================================================


def test_docs_link_autofix_registered():
    """Test that docs-link-autofix command is registered."""
    assert "docs-link-autofix" in _CLI_COMMANDS
    cmd_def = _CLI_COMMANDS["docs-link-autofix"]
    assert cmd_def.func == docs_link.autofix_links
    assert "yes" in cmd_def.args
    assert "no_backup" in cmd_def.args


def test_autofix_links_dry_run():
    """Test autofix_links in dry-run mode (default)."""
    result = docs_link.autofix_links(yes=False)
    assert result == 0  # Should not crash in dry-run

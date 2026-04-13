"""Unit tests for remaining tool CLI decorators (audit, clean, docs_link)."""

from __future__ import annotations

from pathlib import Path

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
    assert "include_historical" in cmd_def.args


def test_check_links_callable():
    """Test check_links is directly callable."""
    result = docs_link.check_links()
    assert isinstance(result, int)


def test_docs_link_default_scan_skips_historical_reports(monkeypatch, tmp_path: Path):
    docs_root = tmp_path / "docs"
    reports_dir = docs_root / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "bad.md").write_text("[broken](missing.md)\n", encoding="utf-8")
    (docs_root / "README.md").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(docs_link, "ROOT", tmp_path)

    broken = docs_link.find_broken_links()
    assert broken == []


def test_docs_link_include_historical_scans_reports(monkeypatch, tmp_path: Path):
    docs_root = tmp_path / "docs"
    reports_dir = docs_root / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "bad.md").write_text("[broken](missing.md)\n", encoding="utf-8")
    (docs_root / "README.md").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(docs_link, "ROOT", tmp_path)

    broken = docs_link.find_broken_links(include_historical=True)
    assert len(broken) == 1
    assert broken[0][0] == reports_dir / "bad.md"


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
    assert "include_historical" in cmd_def.args


def test_autofix_links_dry_run():
    """Test autofix_links in dry-run mode (default)."""
    result = docs_link.autofix_links(yes=False)
    assert result == 0  # Should not crash in dry-run


def test_tools_reference_documents_current_docs_link_commands():
    """Tool reference should describe the current docs-link commands and policy."""
    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "docs" / "06_REFERENCE" / "TOOLS_REFERENCE.md").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "python -m dataselector docs-link-check" in text
    assert "python -m dataselector docs-link-autofix" in text
    assert "--include-historical" in text
    assert "dataselector tools docs-check" not in text
    assert "dataselector tools docs-fix" not in text

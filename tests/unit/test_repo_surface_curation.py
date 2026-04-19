from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SURFACE_DOC = ROOT / "docs" / "08_GOVERNANCE" / "REPO_SURFACE_CURATION.md"
INDEX_DOC = ROOT / "docs" / "00_OVERVIEW" / "OVERVIEW.md"
SCRIPTS_REF_DOC = ROOT / "docs" / "06_REFERENCE" / "scripts_reference.md"
ARCHIVE_DOC = ROOT / "docs" / "07_ARCHIVE" / "README.md"
TEST_ARCHIVE_DOC = ROOT / "tests" / "archive" / "README.md"
PHASE4H_README = ROOT / "scripts" / "phase4h" / "README.md"


def test_repo_surface_doc_maps_active_secondary_and_historical_zones():
    text = SURFACE_DOC.read_text(encoding="utf-8", errors="ignore")
    assert "Canonical active surface" in text
    assert "Secondary active surface" in text
    assert "Historical / archived surface" in text
    assert "docs/07_ARCHIVE/" in text
    assert "tests/archive/" in text
    assert "archive_local/" in text
    assert "scripts/phase4h/" in text
    assert "scripts/handoff_check.sh" in text


def test_scripts_reference_is_cli_first_and_explicitly_non_canonical():
    text = SCRIPTS_REF_DOC.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    assert "secondary / historical" in lowered
    assert "thesis-orchestrate" in text
    assert "thesis-pipeline" in text
    assert "scripts/handoff_check.sh" in text
    assert "scripts/copy_selection_tiles.py" in text
    assert "scripts/phase4h/" in text
    assert "84→9 scripts" not in text


def test_docs_index_demotes_secondary_and_historical_material():
    text = INDEX_DOC.read_text(encoding="utf-8", errors="ignore")
    assert "Secondary active reference" in text
    assert "advanced / legacy" in text
    assert "07_ARCHIVE" in text
    assert "archive_local/" in text
    assert "REPO_SURFACE_CURATION.md" in text
    assert "TEST_SUITE_CURATION.md" in text


def test_archive_readmes_explain_non_authoritative_role():
    archive_text = ARCHIVE_DOC.read_text(encoding="utf-8", errors="ignore")
    tests_archive_text = TEST_ARCHIVE_DOC.read_text(encoding="utf-8", errors="ignore")
    phase4h_text = PHASE4H_README.read_text(encoding="utf-8", errors="ignore")

    assert "historical documentation" in archive_text
    assert "not authoritative" in archive_text
    assert "README.md" in archive_text
    assert "docs/00_OVERVIEW/OVERVIEW.md" in archive_text

    assert "historical or archived tests" in tests_archive_text
    assert "not" in tests_archive_text.lower()
    assert "docs/08_GOVERNANCE/TEST_SUITE_CURATION.md" in tests_archive_text

    assert "Historical closeout automation" in phase4h_text
    assert "not" in phase4h_text.lower()
    assert "thesis-orchestrate" in phase4h_text

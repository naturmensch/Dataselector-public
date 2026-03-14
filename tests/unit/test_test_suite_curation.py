from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
HISTORICAL_INVENTORY = ROOT / "docs" / "reorganize_tests_plans" / "test_inventory.md"
CURATION_DOC = ROOT / "docs" / "TEST_SUITE_CURATION.md"
PHASE2_SKIP_REASON = "Phase 2 merge artifact"
UNCONDITIONAL_SKIP_PATTERNS = [
    re.compile(r"skipif\(\s*True\b"),
    re.compile(r"pytestmark\s*=\s*pytest\.mark\.skip\b"),
    re.compile(r"@pytest\.mark\.skip\("),
]


def test_no_phase2_merge_artifact_skip_files_remain():
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name == "test_test_suite_curation.py":
            continue
        if PHASE2_SKIP_REASON in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "Skip-only Phase 2 merge artifact files should not remain in the active suite:\n"
        + "\n".join(offenders)
    )


def test_historical_test_inventory_is_marked_non_authoritative():
    text = HISTORICAL_INVENTORY.read_text(encoding="utf-8", errors="ignore")
    assert (
        "Historical planning note" in text
    ), "Historical test inventory must be marked as non-authoritative"
    assert (
        "docs/TEST_SUITE_CURATION.md" in text
    ), "Historical inventory must point readers to the active curation doc"


def test_curation_doc_distinguishes_active_and_historical_tests():
    text = CURATION_DOC.read_text(encoding="utf-8", errors="ignore")
    assert (
        "Historical / compatibility" in text
    ), "Curation doc must classify historical/compatibility tests explicitly"
    assert (
        "Pure `Phase 2 merge artifact` skip-only files were removed" in text
    ), "Curation doc must document the current cleanup decision"
    assert (
        "Expected release skips" in text
    ), "Curation doc must explain the remaining acceptable release skip classes"


def test_active_suite_has_no_unconditional_skip_stubs():
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        if "tests/archive/" in str(path.as_posix()):
            continue
        if path.name == "test_test_suite_curation.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in UNCONDITIONAL_SKIP_PATTERNS:
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
                break
    assert (
        not offenders
    ), "Active suite must not contain unconditional skip stubs:\n" + "\n".join(
        offenders
    )

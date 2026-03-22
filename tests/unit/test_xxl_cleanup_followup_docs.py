from __future__ import annotations

from pathlib import Path

from dataselector.tools import docs_link

ROOT = Path(__file__).resolve().parents[2]

FOLLOWUP_DOCS = [
    ROOT / "docs" / "DEVELOPER.md",
    ROOT / "docs" / "02_THEORY" / "architecture.md",
    ROOT / "docs" / "adr" / "ADR-002-workflow-decision-matrix.md",
    ROOT / "docs" / "03_USER_GUIDES" / "UQ_VALIDATION.md",
    ROOT / "tests" / "smoke" / "README.md",
    ROOT / "tests" / "e2e" / "README_TEST_STRATEGY.md",
]


def test_followup_docs_do_not_promote_removed_xxl_surface():
    offenders: list[str] = []
    banned_patterns = [
        "dataselector xxl",
        "dataselector xxl-monitor",
        "scripts/run_xxl_pipeline.py",
        "scripts/xxl_full_run_monitor.py",
        "outputs/xxl/",
        "resume_meta.json",
        "test_xxl_pipeline.py",
        "test_resume_recovery.py",
    ]

    for path in FOLLOWUP_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned_patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {pattern}")

    assert not offenders, (
        "Follow-up docs still promote removed XXL surface:\n" + "\n".join(offenders)
    )


def test_adr_002_is_marked_superseded():
    text = (
        ROOT / "docs" / "adr" / "ADR-002-workflow-decision-matrix.md"
    ).read_text(encoding="utf-8", errors="ignore")
    assert "**Status**: Superseded" in text


def test_legacy_xxl_ops_archive_has_no_broken_relative_links():
    archive_dir = ROOT / "docs" / "07_ARCHIVE" / "legacy_xxl_ops"
    broken = docs_link.find_broken_links(archive_dir)
    assert not broken, (
        "legacy_xxl_ops archive still contains broken relative links:\n"
        + "\n".join(
            f"{src.relative_to(ROOT)}: [{text}]({target})"
            for src, target, text in broken
        )
    )


def test_historical_reports_and_cleanup_docs_have_no_broken_relative_links():
    checked_dirs = [
        ROOT / "docs" / "reports",
        ROOT / "docs" / "cleanup_scripts",
        ROOT / "docs" / "07_ARCHIVE" / "phase4h_closeout_2026-02-10" / "legacy_docs_root",
    ]
    broken = []
    for docs_dir in checked_dirs:
        broken.extend(docs_link.find_broken_links(docs_dir))

    assert not broken, (
        "Historical doc bundles still contain broken relative links:\n"
        + "\n".join(
            f"{src.relative_to(ROOT)}: [{text}]({target})"
            for src, target, text in broken
        )
    )

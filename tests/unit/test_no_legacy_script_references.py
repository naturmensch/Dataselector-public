"""Guards against reintroducing legacy script-era entrypoints."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CODE_PATTERNS = [
    re.compile(r"^\s*from\s+scripts(?:\.|\s)", re.MULTILINE),
    re.compile(r"^\s*import\s+scripts(?:\.|\s|$)", re.MULTILINE),
    re.compile(r"\bpython(?:3)?\s+scripts/[^\s]+\.py\b"),
    re.compile(r"\bpython(?:3)?\s+-m\s+scripts(?:\.|\b)"),
]

CI_PATTERNS = [
    re.compile(r"\bpython(?:3)?\s+scripts/[^\s]+\.py\b"),
    re.compile(r"\bpython(?:3)?\s+-m\s+scripts(?:\.|\b)"),
]

REPO_SCRIPT_PATH_PATTERNS = [
    re.compile(
        r"Path\(__file__\)\.resolve\(\)\.parents\[\d+\]\s*/\s*['\"]scripts['\"]\s*/\s*['\"][^'\"]+\.py['\"]"
    ),
    re.compile(
        r"\b(?:REPO_ROOT|ROOT|repo_root)\s*/\s*['\"]scripts['\"]\s*/\s*['\"][^'\"]+\.py['\"]"
    ),
]

LEGACY_SPATIAL_SCHEMA_PATTERNS = [
    re.compile(
        r"required_columns\s*=\s*\[[^\]]*['\"]N['\"][^\]]*['\"]left['\"][^\]]*\]"
    ),
    re.compile(r"\[\s*['\"]N['\"]\s*,\s*['\"]left['\"]\s*\]"),
    re.compile(r"['\"]N['\"]\s*\)\s*and\s*\(\s*['\"]left['\"]"),
]

# Transitional wrappers may still import internal modules while migration
# to CLI-first subcommands is being completed.
TRANSITIONAL_WRAPPER_ALLOWLIST_VERSION = "2026-02-10"
TRANSITIONAL_WRAPPER_ALLOWLIST: set[str] = set()
INTERNAL_DOMAIN_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+dataselector\.(pipeline|selection|features|workflows)\b"),
    re.compile(
        r"^\s*import\s+dataselector\.(pipeline|selection|features|workflows)\b"
    ),
]


def _scan_file(path: Path, patterns: list[re.Pattern[str]]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in patterns:
            if pat.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
                break
    return hits


def _scan_dynamic_repo_script_paths(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits = []
    has_loader = "spec_from_file_location" in text

    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in REPO_SCRIPT_PATH_PATTERNS:
            if pat.search(line):
                prefix = (
                    "dynamic loader + repo scripts path"
                    if has_loader
                    else "repo scripts path"
                )
                hits.append(
                    f"{path.relative_to(ROOT)}:{lineno}: {prefix}: {line.strip()}"
                )
                break
    return hits


def test_no_legacy_script_references_in_code_or_tests():
    offenders: list[str] = []
    this_file = Path(__file__).resolve()

    for base in (ROOT / "dataselector", ROOT / "tests"):
        for py_file in sorted(base.rglob("*.py")):
            if py_file.resolve() == this_file:
                continue
            offenders.extend(_scan_file(py_file, CODE_PATTERNS))
            offenders.extend(_scan_dynamic_repo_script_paths(py_file))

    assert not offenders, "Found legacy script references:\n" + "\n".join(offenders)


def test_no_legacy_script_references_in_ci_or_makefile():
    offenders: list[str] = []

    makefile = ROOT / "Makefile"
    if makefile.exists():
        offenders.extend(_scan_file(makefile, CI_PATTERNS))

    workflows_dir = ROOT / ".github" / "workflows"
    if workflows_dir.exists():
        for wf in sorted(workflows_dir.glob("*.y*ml")):
            offenders.extend(_scan_file(wf, CI_PATTERNS))

    assert not offenders, "Found legacy CI/Makefile references:\n" + "\n".join(
        offenders
    )


def test_no_hardcoded_legacy_spatial_schema_in_production_code():
    offenders: list[str] = []

    for py_file in sorted((ROOT / "dataselector").rglob("*.py")):
        offenders.extend(_scan_file(py_file, LEGACY_SPATIAL_SCHEMA_PATTERNS))

    assert not offenders, (
        "Found hardcoded legacy spatial schema expectations (N/left) "
        "in production code:\n" + "\n".join(offenders)
    )


def test_wrapper_scripts_do_not_duplicate_domain_logic():
    """Keep scripts thin; only explicit transitional wrappers may import domain internals."""
    offenders: list[str] = []
    scripts_dir = ROOT / "scripts"
    for script in sorted(scripts_dir.glob("*.py")):
        rel = str(script.relative_to(ROOT))
        if rel in TRANSITIONAL_WRAPPER_ALLOWLIST:
            continue
        offenders.extend(_scan_file(script, INTERNAL_DOMAIN_IMPORT_PATTERNS))

    assert not offenders, (
        "Found non-allowlisted scripts importing domain internals "
        "(wrapper contract violation):\n" + "\n".join(offenders)
    )

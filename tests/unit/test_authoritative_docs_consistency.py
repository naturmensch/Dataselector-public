"""Consistency guards for authoritative documentation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

AUTHORITATIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "EXPERIMENT_MANAGER_GUIDE.md",
    ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md",
    ROOT / "docs" / "ENV_SETUP.md",
    ROOT / "docs" / "DEVELOPER.md",
]

ACTIVE_COMMAND_STYLE_DOCS = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "docs" / "DEVELOPER.md",
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "TEST_SUITE_CURATION.md",
]


def test_authoritative_docs_no_hardcoded_miniconda_paths():
    offenders: list[str] = []
    for doc in AUTHORITATIVE_DOCS:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "/opt/miniconda3" in line:
                offenders.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Authoritative docs still contain hardcoded /opt/miniconda3 paths:\n"
        + "\n".join(offenders)
    )


def test_authoritative_docs_use_outputs_runs_not_outputs_experiments():
    offenders: list[str] = []
    for doc in AUTHORITATIVE_DOCS:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "outputs/experiments" in line:
                offenders.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Authoritative docs still contain legacy outputs/experiments references:\n"
        + "\n".join(offenders)
    )


def test_authoritative_docs_reference_src_only_as_legacy():
    offenders: list[str] = []
    for doc in AUTHORITATIVE_DOCS:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "src/" not in line:
                continue
            lower = line.lower()
            if "legacy" not in lower and "compat" not in lower:
                offenders.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Authoritative docs reference src/ without explicit legacy context:\n"
        + "\n".join(offenders)
    )


def test_active_docs_use_python_module_invocation_for_micromamba_commands():
    offenders: list[str] = []
    legacy_patterns = (
        "micromamba run -n dataselector pytest",
        "micromamba run -n dataselector pip ",
        "micromamba run -n dataselector -- ",
    )

    for doc in ACTIVE_COMMAND_STYLE_DOCS:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in legacy_patterns):
                offenders.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Active docs still contain legacy micromamba command styles:\n"
        + "\n".join(offenders)
    )

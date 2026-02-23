"""Documentation policy checks for active vs. historical configs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Authoritative docs must not accidentally promote historical configs as defaults.
AUTHORITATIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "EXPERIMENT_MANAGER_GUIDE.md",
    ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md",
    ROOT / "docs" / "ENV_SETUP.md",
    ROOT / "docs" / "DEVELOPER.md",
    ROOT / "docs" / "OPERATIONS.md",
]

HISTORICAL_CONFIG = "config/pipeline_config.best_trial_70.yaml"
ACTIVE_CONFIG = "config/pipeline_config.yaml"


def _line_is_historical(line: str) -> bool:
    l = line.lower()
    return "historical" in l or "reference" in l or "non-default" in l


def test_authoritative_docs_use_active_config_as_default():
    offenders: list[str] = []

    for path in AUTHORITATIVE_DOCS:
        if not path.exists():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if HISTORICAL_CONFIG in line and not _line_is_historical(line):
                offenders.append(
                    f"{path.relative_to(ROOT)}:{lineno}: historical config mentioned "
                    f"without explicit marker -> {line.strip()}"
                )

    assert not offenders, (
        "Historical config must not be presented as active default in "
        "authoritative docs:\n" + "\n".join(offenders)
    )


def test_config_policy_doc_declares_active_default():
    policy_path = ROOT / "docs" / "CONFIG_POLICY.md"
    assert policy_path.exists(), "Missing docs/CONFIG_POLICY.md"
    text = policy_path.read_text(encoding="utf-8", errors="ignore")
    assert ACTIVE_CONFIG in text, "Config policy must declare active default config"


def test_config_policy_doc_declares_micromamba_canonical_runtime():
    policy_path = ROOT / "docs" / "CONFIG_POLICY.md"
    assert policy_path.exists(), "Missing docs/CONFIG_POLICY.md"
    text = policy_path.read_text(encoding="utf-8", errors="ignore")
    assert "micromamba run -n dataselector <command>" in text, (
        "Config policy must declare micromamba runtime as canonical"
    )
    assert "compatibility wrapper" in text.lower(), (
        "Config policy must classify exec_in_env as compatibility layer"
    )


def test_parameter_policy_ledger_exists_and_declares_active_config():
    ledger_path = ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md"
    assert ledger_path.exists(), "Missing docs/PARAMETER_POLICY_LEDGER.md"
    text = ledger_path.read_text(encoding="utf-8", errors="ignore")
    assert ACTIVE_CONFIG in text, (
        "Parameter policy ledger must reference active config policy"
    )


def test_config_policy_doc_declares_warning_policy_contract():
    policy_path = ROOT / "docs" / "CONFIG_POLICY.md"
    assert policy_path.exists(), "Missing docs/CONFIG_POLICY.md"
    text = policy_path.read_text(encoding="utf-8", errors="ignore")
    assert "Warning Policy (Thesis Gates)" in text, (
        "Config policy must declare thesis warning policy contract"
    )
    assert "Broad warning suppression" in text, (
        "Config policy must forbid broad warning suppression"
    )

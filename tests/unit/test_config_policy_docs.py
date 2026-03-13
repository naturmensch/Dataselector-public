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
THESIS_FREEZE_DOCS = [
    ROOT / "docs" / "METHODOLOGY.md",
    ROOT / "docs" / "THESIS_METHOD_CONTRACT.md",
    ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md",
    ROOT / "docs" / "CONFIG_POLICY.md",
    ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md",
    ROOT / "docs" / "thesis_chapter_training_data_selection.tex",
]


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


def test_thesis_freeze_docs_declare_dataset_vs_parameter_authority():
    for path in THESIS_FREEZE_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "Dataset authority:" in text, (
            f"{path.relative_to(ROOT)} must declare dataset authority"
        )
        assert "Parameter authority:" in text, (
            f"{path.relative_to(ROOT)} must declare parameter authority"
        )
        assert "selection_core.csv" in text, (
            f"{path.relative_to(ROOT)} must reference selection_core.csv"
        )
        assert "selection_final_with_cases.csv" in text, (
            f"{path.relative_to(ROOT)} must reference selection_final_with_cases.csv"
        )
        assert "selection_contract.json" in text, (
            f"{path.relative_to(ROOT)} must reference selection_contract.json"
        )


def test_index_prioritizes_canonical_thesis_path():
    index_path = ROOT / "docs" / "INDEX.md"
    text = index_path.read_text(encoding="utf-8", errors="ignore")
    assert "thesis-orchestrate" in text, (
        "Documentation hub must reference thesis-orchestrate as canonical thesis path"
    )
    assert "advanced / legacy" in text, (
        "Documentation hub must label XXL guidance as advanced / legacy"
    )
    assert "dataselector xxl --best-sampler cmaes" not in text, (
        "Documentation hub must not present dataselector xxl as the default thesis workflow"
    )


def test_active_thesis_docs_do_not_repeat_obsolete_selection_story():
    offenders: list[str] = []
    banned_patterns = [
        "34 Kacheln",
        "34 selektierten",
        "alpha=0.40, beta=0.30, gamma=0.30",
        "scripts/run_full_experiment.sh",
    ]
    for path in THESIS_FREEZE_DOCS + [ROOT / "docs" / "INDEX.md"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned_patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {pattern}")
    assert not offenders, (
        "Active thesis docs still contain obsolete selection narrative:\n"
        + "\n".join(offenders)
    )

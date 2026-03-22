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
THESIS_POLICY_DOCS = [
    ROOT / "docs" / "METHODOLOGY.md",
    ROOT / "docs" / "THESIS_METHOD_CONTRACT.md",
    ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md",
    ROOT / "docs" / "CONFIG_POLICY.md",
    ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md",
]
MIN_DISTANCE_EVIDENCE_DOC = ROOT / "docs" / "MIN_DISTANCE_EVIDENCE_ADDENDUM.md"
N_SAMPLES_EVIDENCE_DOC = ROOT / "docs" / "N_SAMPLES_EVIDENCE_ADDENDUM.md"
TEST_SUITE_CURATION_DOC = ROOT / "docs" / "TEST_SUITE_CURATION.md"
THESIS_MODEL_BOUNDARY_DOCS = [
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "docs" / "METHODOLOGY.md",
    ROOT / "docs" / "THESIS_METHOD_CONTRACT.md",
    ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md",
]


def _line_is_historical(line: str) -> bool:
    lowered = line.lower()
    return "historical" in lowered or "reference" in lowered or "non-default" in lowered


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
    assert (
        "micromamba run -n dataselector <command>" in text
    ), "Config policy must declare micromamba runtime as canonical"
    assert (
        "compatibility wrapper" in text.lower()
    ), "Config policy must classify exec_in_env as compatibility layer"


def test_parameter_policy_ledger_exists_and_declares_active_config():
    ledger_path = ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md"
    assert ledger_path.exists(), "Missing docs/PARAMETER_POLICY_LEDGER.md"
    text = ledger_path.read_text(encoding="utf-8", errors="ignore")
    assert (
        ACTIVE_CONFIG in text
    ), "Parameter policy ledger must reference active config policy"


def test_config_policy_doc_declares_warning_policy_contract():
    policy_path = ROOT / "docs" / "CONFIG_POLICY.md"
    assert policy_path.exists(), "Missing docs/CONFIG_POLICY.md"
    text = policy_path.read_text(encoding="utf-8", errors="ignore")
    assert (
        "Warning Policy (Thesis Gates)" in text
    ), "Config policy must declare thesis warning policy contract"
    assert (
        "Broad warning suppression" in text
    ), "Config policy must forbid broad warning suppression"


def test_thesis_freeze_docs_declare_dataset_vs_parameter_authority():
    for path in THESIS_FREEZE_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert (
            "Dataset authority:" in text
        ), f"{path.relative_to(ROOT)} must declare dataset authority"
        assert (
            "Parameter authority:" in text
        ), f"{path.relative_to(ROOT)} must declare parameter authority"
        assert (
            "selection_core.csv" in text
        ), f"{path.relative_to(ROOT)} must reference selection_core.csv"
        assert (
            "selection_final_with_cases.csv" in text
        ), f"{path.relative_to(ROOT)} must reference selection_final_with_cases.csv"
        assert (
            "selection_contract.json" in text
        ), f"{path.relative_to(ROOT)} must reference selection_contract.json"


def test_index_prioritizes_canonical_thesis_path():
    index_path = ROOT / "docs" / "INDEX.md"
    text = index_path.read_text(encoding="utf-8", errors="ignore")
    assert (
        "thesis-orchestrate" in text
    ), "Documentation hub must reference thesis-orchestrate as canonical thesis path"
    assert (
        "advanced / legacy" in text
    ), "Documentation hub must label XXL guidance as advanced / legacy"
    assert (
        "dataselector xxl --best-sampler cmaes" not in text
    ), "Documentation hub must not present dataselector xxl as the default thesis workflow"


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
    assert (
        not offenders
    ), "Active thesis docs still contain obsolete selection narrative:\n" + "\n".join(
        offenders
    )


def test_active_config_declares_v2_selection_policy_defaults():
    import yaml

    cfg_path = ROOT / "config" / "pipeline_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    selection = cfg.get("selection", {})
    assert (
        selection.get("selection_authority") == "snapshot_primary"
    ), "Active config must default to selection.selection_authority=snapshot_primary"
    assert (
        selection.get("objective_authority") == "unified_normalized"
    ), "Active config must default to selection.objective_authority=unified_normalized"


def test_active_policy_docs_declare_v2_selection_authority_keys():
    for path in THESIS_POLICY_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert (
            "selection.selection_authority" in text
        ), f"{path.relative_to(ROOT)} must declare selection.selection_authority"
        assert (
            "selection.objective_authority" in text
        ), f"{path.relative_to(ROOT)} must declare selection.objective_authority"
        assert (
            "snapshot_primary" in text
        ), f"{path.relative_to(ROOT)} must mention snapshot_primary as thesis default"
        assert (
            "unified_normalized" in text
        ), f"{path.relative_to(ROOT)} must mention unified_normalized objective authority"


def test_active_entry_docs_declare_model_agnostic_selection_boundary():
    for path in THESIS_MODEL_BOUNDARY_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        assert (
            "model-agnostic" in lower or "architektur-neutral" in lower
        ), f"{path.relative_to(ROOT)} must declare model-agnostic / architektur-neutral selection"
        assert (
            "frozen dataset" in lower
        ), f"{path.relative_to(ROOT)} must use frozen dataset terminology"
        assert (
            "No direct model-metric optimization (SegFormer/MapSAM/UNet++)." in text
        ), f"{path.relative_to(ROOT)} must explicitly state the no-direct-model-metric boundary"


def test_active_entry_docs_do_not_claim_direct_model_metric_optimization():
    offenders: list[str] = []
    banned_patterns = [
        "direkt auf SegFormer/MapSAM/UNet++-Leistungsmetriken optimiert",
        "directly optimized for SegFormer/MapSAM/UNet++",
    ]
    for path in THESIS_MODEL_BOUNDARY_DOCS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in banned_patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {pattern}")
    assert not offenders, (
        "Active entry docs must not claim direct model-metric optimization:\n"
        + "\n".join(offenders)
    )


def test_readmes_prioritize_canonical_thesis_release_path():
    for path in [ROOT / "README.md", ROOT / "README_EN.md"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert (
            "thesis-orchestrate" in text
        ), f"{path.relative_to(ROOT)} must present thesis-orchestrate"
        assert (
            "thesis-pipeline" in text
        ), f"{path.relative_to(ROOT)} must mention thesis-pipeline"
        assert (
            ACTIVE_CONFIG in text
        ), f"{path.relative_to(ROOT)} must mention the active config"
        assert (
            "outputs/runs/" in text
        ), f"{path.relative_to(ROOT)} must mention the canonical run root"
        assert (
            "Advanced / historical workflows" in text or "Advanced / historical" in text
        ), f"{path.relative_to(ROOT)} must demote historical workflows explicitly"


def test_min_distance_docs_capture_primary_and_supplementary_evidence():
    addendum_text = MIN_DISTANCE_EVIDENCE_DOC.read_text(
        encoding="utf-8", errors="ignore"
    )
    assert "28.5 km" in addendum_text, "Addendum must document active policy value"
    assert "40.0 km" in addendum_text, "Addendum must document comparison candidate"
    assert "45.0 km" in addendum_text, "Addendum must document geometric reference"
    assert "5 km" in addendum_text, "Addendum must document historical low-distance run"
    assert "8 km" in addendum_text, "Addendum must document historical low-distance run"
    assert (
        "MIN_DISTANCE_DECISION_2026-02-09.md" in addendum_text
    ), "Addendum must cite the primary pre-registered decision evidence"
    assert (
        "thesis_pipeline_double_run_analysis_2026-02-11.md" in addendum_text
    ), "Addendum must cite the supplementary historical analysis"

    calc_text = (ROOT / "docs" / "MIN_DISTANCE_CALCULATION.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert (
        "docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md" in calc_text
    ), "Min-distance calculation doc must link the active evidence addendum"
    assert (
        "40.0 km" in calc_text
    ), "Min-distance calculation doc must explicitly classify 40.0 km as comparison candidate"
    assert (
        "5 km" in calc_text and "8 km" in calc_text
    ), "Min-distance calculation doc must mention supplementary low-distance evidence"

    howto_text = (
        ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md"
    ).read_text(encoding="utf-8", errors="ignore")
    assert (
        "docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md" in howto_text
    ), "Thesis HOWTO must link the min-distance evidence addendum"
    assert (
        "5 km" in howto_text and "8 km" in howto_text
    ), "Thesis HOWTO must mention the historical low-distance drift evidence"

    ledger_text = (ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert (
        "5-8 km" in ledger_text
    ), "Parameter policy ledger must classify the historical low-distance evidence"
    assert (
        "docs/MIN_DISTANCE_EVIDENCE_ADDENDUM.md" in ledger_text
    ), "Parameter policy ledger must cite the evidence addendum"


def test_n_samples_docs_capture_policy_and_supplementary_architecture_evidence():
    addendum_text = N_SAMPLES_EVIDENCE_DOC.read_text(encoding="utf-8", errors="ignore")
    assert "5%" in addendum_text, "Addendum must document the corridor center"
    assert "4-8%" in addendum_text, "Addendum must document the bounded corridor"
    assert (
        "minimal-feasible plateau" in addendum_text
    ), "Addendum must document the selection rule"
    assert (
        "Few-Shot Segmentation of Historical Maps" in addendum_text
    ), "Addendum must cite the strongest direct historical-map evidence"
    assert (
        "MapSAM" in addendum_text
    ), "Addendum must cite the supplementary MapSAM-family evidence"
    assert (
        "SegFormer" in addendum_text
    ), "Addendum must classify SegFormer as indirect support"
    assert (
        "UNet++" in addendum_text
    ), "Addendum must classify UNet++ as conservative cautionary support"
    assert (
        "external papers prove `4-8%` for KDR100" in addendum_text
    ), "Addendum must explicitly limit claim strength"

    config_text = (ROOT / "docs" / "CONFIG_POLICY.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert (
        "docs/N_SAMPLES_EVIDENCE_ADDENDUM.md" in config_text
    ), "Config policy doc must link the n_samples evidence addendum"
    assert (
        "supplementary only" in config_text
    ), "Config policy doc must classify architecture-specific evidence as supplementary"

    howto_text = (
        ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md"
    ).read_text(encoding="utf-8", errors="ignore")
    assert (
        "docs/N_SAMPLES_EVIDENCE_ADDENDUM.md" in howto_text
    ), "Thesis HOWTO must link the n_samples evidence addendum"
    assert (
        "MapSAM" in howto_text and "UNet++" in howto_text
    ), "Thesis HOWTO must explain the supplementary architecture-specific interpretation"
    assert (
        "not directly derived from" in howto_text
    ), "Thesis HOWTO must preserve the model-agnostic selection boundary"

    ledger_text = (ROOT / "docs" / "PARAMETER_POLICY_LEDGER.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert (
        "docs/N_SAMPLES_EVIDENCE_ADDENDUM.md" in ledger_text
    ), "Parameter policy ledger must cite the n_samples evidence addendum"
    assert (
        "foundation-model-based downstream training" in ledger_text
    ), "Parameter policy ledger must capture the supplementary architecture-facing rationale"


def test_phase5_docs_capture_optional_post_freeze_boundary():
    config_text = (ROOT / "docs" / "CONFIG_POLICY.md").read_text(
        encoding="utf-8", errors="ignore"
    )
    assert (
        "Integrated Phase 5 Policy" in config_text
    ), "Config policy must document the integrated optional Phase 5 policy"
    assert (
        "--build-handoffs" in config_text
    ), "Config policy must document the Phase 5 gate flag"
    assert (
        "core-only" in config_text
    ), "Config policy must document the default integrated patch scope"

    howto_text = (
        ROOT / "docs" / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md"
    ).read_text(encoding="utf-8", errors="ignore")
    assert (
        "Optional Phase 5: Annotation Plan + Handoff Bundle" in howto_text
    ), "Thesis HOWTO must document the integrated Phase 5 flow"
    assert (
        "--build-handoffs" in howto_text
    ), "Thesis HOWTO must document the integrated handoff flag"
    assert (
        "--patch-include-case false" in howto_text
    ), "Thesis HOWTO must document the default core-only integrated patch scope"


def test_active_test_suite_curation_doc_exists_and_maps_release_tiers():
    text = TEST_SUITE_CURATION_DOC.read_text(encoding="utf-8", errors="ignore")
    assert (
        "tests/test_thesis_pipeline.py" in text
    ), "Test suite curation doc must name the authoritative thesis pipeline test"
    assert (
        "tests/unit/test_thesis_orchestrate.py" in text
    ), "Test suite curation doc must name the authoritative orchestrator test"
    assert (
        "tests/unit/test_crs_strict_thesis_mode.py" in text
    ), "Test suite curation doc must cover CRS strict mode"
    assert (
        "tests/unit/test_handoff_bundle.py" in text
    ), "Test suite curation doc must cover the handoff bundle"
    assert (
        "Long manual release checks" in text
    ), "Test suite curation doc must distinguish long manual checks"


def test_makefile_uses_canonical_micromamba_runtime_for_quality_targets():
    makefile_text = (ROOT / "Makefile").read_text(encoding="utf-8", errors="ignore")
    assert (
        "EXEC_ENV ?= env XDG_CACHE_HOME=/tmp/mamba-cache" in makefile_text
    ), "Makefile must pin micromamba cache to a temporary XDG cache root for reliable local runs"
    assert (
        "EXEC_PYTHON ?= micromamba run -n $(ENV_NAME) python" in makefile_text
    ), "Makefile must default to the canonical micromamba Python runtime"
    assert (
        "EXEC_TOOL ?= micromamba run -n $(ENV_NAME)" in makefile_text
    ), "Makefile must default to the canonical micromamba tool runtime"
    assert (
        "FORMAT_PATHS ?= dataselector tests scripts" in makefile_text
    ), "Makefile quality targets must scope formatting to project-owned source paths"
    assert (
        "$(EXEC_ENV) $(EXEC_PYTHON) -m isort --check-only $(FORMAT_PATHS)"
        in makefile_text
    ), "format-check must run isort inside the dataselector environment"
    assert (
        "$(EXEC_ENV) $(EXEC_PYTHON) -m black --check $(FORMAT_PATHS)" in makefile_text
    ), "format-check must run black inside the dataselector environment"
    assert (
        "$(EXEC_ENV) $(EXEC_TOOL) ruff check $(FORMAT_PATHS)" in makefile_text
    ), "format-check must run ruff inside the dataselector environment"
    assert (
        "-m pip install isort black ruff" not in makefile_text
    ), "Makefile quality targets must not install formatting tools at runtime"

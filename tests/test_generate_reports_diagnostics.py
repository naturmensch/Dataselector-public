from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataselector.workflows.generate_reports import _generate_single_run_thesis_report


def _write_minimal_artifacts(run_dir: Path, *, n_selected_values: list[int]) -> None:
    optuna_dir = run_dir / "optuna"
    resolution_dir = run_dir / "parameter_resolution"
    pareto_dir = run_dir / "tuning_weights" / "pareto"
    validation_dir = run_dir / "validation"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    resolution_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"number": 0, "state": "COMPLETE", "value": 1.23},
            {"number": 1, "state": "COMPLETE", "value": 1.11},
        ]
    ).to_csv(optuna_dir / "optuna_results.csv", index=False)

    (resolution_dir / "optuna_autoscale_best_latest.json").write_text(
        json.dumps(
            {
                "value": 1.23,
                "params": {"a": 0.2, "b": 0.3, "c": 0.5, "min_distance_km": 29},
                "best_selection_rule": "minimal_feasible_plateau",
                "study_sampler": "TPESampler",
                "study_seed": 42,
                "user_attrs": {"n_samples": 40},
            }
        ),
        encoding="utf-8",
    )
    (resolution_dir / "optuna_autoscale_stage_policy.json").write_text(
        json.dumps(
            {
                "mode": "corridor",
                "effective_candidates": 675,
                "stages_resolved": [27, 34, 40, 54],
                "trials_per_stage": [30, 40, 60, 80],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"stage": 0, "n_samples": 27, "stage_feasible": True},
            {"stage": 1, "n_samples": 34, "stage_feasible": True},
        ]
    ).to_csv(resolution_dir / "optuna_autoscale_summary_20260212.csv", index=False)

    pd.DataFrame([{"alpha": 0.2, "beta": 0.3, "gamma": 0.5}]).to_csv(
        pareto_dir / "pareto_solutions.csv",
        index=False,
    )

    (run_dir / "tuning_weights" / "meta.json").write_text(
        json.dumps(
            {
                "best_metrics": {
                    "alpha": 0.2,
                    "beta": 0.3,
                    "gamma": 0.5,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "selection_rank": 0,
                "shortName": "KDR_001",
                "city": "CityA",
                "year": 1900,
            },
            {
                "selection_rank": 1,
                "shortName": "KDR_002",
                "city": "CityB",
                "year": 1910,
            },
        ]
    ).to_csv(
        run_dir / "tuning_weights" / "selection_a0.2_b0.3_g0.5.csv",
        index=False,
    )

    pd.DataFrame({"n_selected": n_selected_values}).to_csv(
        validation_dir / "validation_results.csv",
        index=False,
    )


def test_report_adds_diagnostic_hint_for_zero_non_empty(tmp_path: Path):
    run_dir = tmp_path / "run_zero_non_empty"
    _write_minimal_artifacts(run_dir, n_selected_values=[0, 0, 0])

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T0")
    report = report_file.read_text(encoding="utf-8")

    assert "- Configurations with non-empty selection: **0**" in report
    assert "Diagnostic hint" in report
    assert "does not automatically mean exploration/optuna failed globally" in report
    assert "optuna/results/trials.csv" not in report
    assert "- ⚠️ Missing:" not in report
    assert "## Tile Selection" in report
    assert "KDR_001" in report
    assert "KDR_002" in report
    assert "| Rank | Tile | City | Year |" in report
    assert "| 0 | `KDR_001` | CityA | 1900 |" in report


def test_report_omits_diagnostic_hint_when_non_empty_exists(tmp_path: Path):
    run_dir = tmp_path / "run_has_non_empty"
    _write_minimal_artifacts(run_dir, n_selected_values=[0, 2, 0])

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T1")
    report = report_file.read_text(encoding="utf-8")

    assert "- Configurations with non-empty selection: **1**" in report
    assert "Diagnostic hint" not in report
    assert "- Best selection rule: `minimal_feasible_plateau`" in report
    assert "- Selected tiles: **2**" in report
    assert "- Unique cities: **2**" in report


def test_report_flags_missing_canonical_artifact(tmp_path: Path):
    run_dir = tmp_path / "run_missing_best"
    _write_minimal_artifacts(run_dir, n_selected_values=[1, 1])
    (run_dir / "parameter_resolution" / "optuna_autoscale_best_latest.json").unlink()

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T2")
    report = report_file.read_text(encoding="utf-8")

    assert "Missing: `parameter_resolution/optuna_autoscale_best_latest.json`" in report
    assert "Report is partial because required thesis artifacts are missing." in report


def test_report_documents_sampler_resolution_contract(tmp_path: Path):
    run_dir = tmp_path / "run_sampler_contract"
    _write_minimal_artifacts(run_dir, n_selected_values=[1, 2, 3])

    sampler_resolution_dir = run_dir / "parameter_resolution" / "sampler_resolution"
    sampler_resolution_dir.mkdir(parents=True, exist_ok=True)

    (sampler_resolution_dir / "selected_sampler.json").write_text(
        json.dumps({"best": "qmc", "sampler": "qmc", "source": "config_policy"}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"sampler": "qmc", "mean": 1.115869},
            {"sampler": "tpe", "mean": 1.098005},
            {"sampler": "cmaes", "mean": 1.094759},
        ]
    ).to_csv(sampler_resolution_dir / "summary.csv", index=False)

    snapshot_path = run_dir / "final_config_resolution.yaml"
    snapshot_path.write_text(
        "\n".join(
            [
                "parameters:",
                "  selection:",
                "    parameter_provenance:",
                "      optuna_sampler:",
                "        method: auto_compare",
                "        source_file: parameter_resolution/sampler_resolution/selected_sampler.json",
            ]
        ),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "resolved_sampler": "qmc",
                    "resolved_sampler_source": "config_policy",
                    "resolved_exploration_sampler": "sobol",
                    "resolved_exploration_sampler_source": "config_policy",
                    "snapshot_path": "final_config_resolution.yaml",
                }
            }
        ),
        encoding="utf-8",
    )

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T3")
    report = report_file.read_text(encoding="utf-8")

    assert "## Sampler Resolution & Scientific Contract" in report
    assert "- Production optuna sampler: `qmc` (source: `config_policy`)" in report
    assert "- Optuna sampler decision provenance: `auto_compare`" in report
    assert "no re-selection during production" in report


def test_report_writes_method_audit_and_key_claims(tmp_path: Path):
    run_dir = tmp_path / "run_method_artifacts"
    _write_minimal_artifacts(run_dir, n_selected_values=[1, 2, 3])

    pd.DataFrame(
        [
            {
                "selection_rank": 0,
                "shortName": "KDR_001",
                "city": "CityA",
                "year": 1900,
            },
            {
                "selection_rank": 1,
                "shortName": "KDR_002",
                "city": "CityB",
                "year": 1910,
            },
        ]
    ).to_csv(run_dir / "selection_core.csv", index=False)
    pd.DataFrame(
        [{"selection_rank": 0, "shortName": "KDR_146", "city": "Hamburg", "year": 1918}]
    ).to_csv(run_dir / "selection_case.csv", index=False)
    pd.DataFrame(
        [
            {
                "selection_rank": 0,
                "shortName": "KDR_001",
                "city": "CityA",
                "year": 1900,
            },
            {
                "selection_rank": 1,
                "shortName": "KDR_002",
                "city": "CityB",
                "year": 1910,
            },
            {
                "selection_rank": 2,
                "shortName": "KDR_146",
                "city": "Hamburg",
                "year": 1918,
            },
        ]
    ).to_csv(run_dir / "selection_final_with_cases.csv", index=False)
    (run_dir / "selection_contract.json").write_text(
        json.dumps(
            {
                "selection_source": "tuning_weights_best_metrics",
                "selection_source_file": "tuning_weights/selection_a0.2_b0.3_g0.5.csv",
                "case_exclude_from_core": True,
                "case_attach_mode": "append_unique",
                "case_tile_names": ["Hamburg"],
                "core_count": 2,
                "case_count_resolved": 1,
                "case_count_attached": 1,
                "case_count": 1,
                "final_count": 3,
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = run_dir / "final_config_resolution.yaml"
    snapshot_path.write_text(
        "\n".join(
            [
                "parameters:",
                "  selection:",
                "    alpha_visual: 0.6",
                "    beta_spatial: 0.2",
                "    gamma_temporal: 0.2",
                "    case_tile_names: []",
                "    _provenance:",
                "      optuna_sampler:",
                "        method: auto_compare",
                "        source_file: parameter_resolution/sampler_resolution/selected_sampler.json",
            ]
        ),
        encoding="utf-8",
    )

    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "validation_method_contract.md").write_text(
        "\n".join(
            [
                "# Validation Method Contract",
                "- Requested replicate mode: `bootstrap_candidates`",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"metric": "n_selected", "mean": 30.0}]).to_csv(
        validation_dir / "validation_summary_stats.csv", index=False
    )
    pd.DataFrame([{"replicate_mode": "bootstrap_candidates"}]).to_csv(
        validation_dir / "validation_results_bootstrap.csv", index=False
    )
    (run_dir / "data_quality").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"phase": "after_exclusion", "year_max": 1921}]).to_csv(
        run_dir / "data_quality" / "year_scope_audit.csv", index=False
    )
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics" / "exceptions.log").write_text(
        "phase: example\ntraceback: example\n",
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "validation_replicate_mode": "bootstrap_candidates",
                    "tile_exclusions_applied": True,
                    "tile_exclusions_count": 1,
                    "tile_excluded_shortnames": ["KDR_155b"],
                    "tile_flagged_count": 2,
                    "tile_flagged_shortnames": ["KDR_039", "KDR_521"],
                    "tile_flagged_classes": ["temporal_scope_outlier"],
                    "tile_flagged_caveats": [
                        {
                            "shortName": "KDR_039",
                            "year": 1980,
                            "class": "temporal_scope_outlier",
                        },
                        {
                            "shortName": "KDR_521",
                            "year": 1985,
                            "class": "temporal_scope_outlier",
                        },
                    ],
                    "tile_exclusion_policy_sha256": "abc123",
                    "exceptions_log_path": "diagnostics/exceptions.log",
                    "case_tile_names": ["Hamburg"],
                    "snapshot_path": "final_config_resolution.yaml",
                    "pipeline_metadata_snapshot": {"extra": {"case_tile_names": []}},
                }
            }
        ),
        encoding="utf-8",
    )

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T4")
    report = report_file.read_text(encoding="utf-8")

    method_audit = run_dir / "THESIS_METHOD_AUDIT.md"
    key_claims = run_dir / "THESIS_KEY_CLAIMS.csv"
    assert method_audit.exists()
    assert key_claims.exists()
    assert "## Method Audit" in report
    assert "THESIS_METHOD_AUDIT.md" in report
    assert "THESIS_KEY_CLAIMS.csv" in report
    assert "## Selection Provenance" in report
    assert "- Parameter snapshot: `final_config_resolution.yaml`" in report
    assert "- Materialized selection source: `tuning_weights_best_metrics`" in report
    assert (
        "- Materialized selection source file: `tuning_weights/selection_a0.2_b0.3_g0.5.csv`"
        in report
    )
    assert (
        "- Snapshot selection weights: `alpha=0.600000, beta=0.200000, gamma=0.200000`"
        in report
    )
    assert (
        "- Materialized selection weights: `alpha=0.200000, beta=0.300000, gamma=0.500000`"
        in report
    )
    assert "- Selection reconciliation status: `documented_difference`" in report

    claims_df = pd.read_csv(key_claims)
    assert {"claim", "evidence_file", "status"}.issubset(claims_df.columns)
    assert (claims_df["status"] == "supported").any()

    method_audit_text = method_audit.read_text(encoding="utf-8")
    assert "- `case_count_resolved`: `1`" in method_audit_text
    assert "- `case_count_attached`: `1`" in method_audit_text

    tile_claim = claims_df.loc[
        claims_df["claim"]
        == "Tile exclusion policy is applied and provenance-tracked in run metadata"
    ]
    assert len(tile_claim) == 1
    assert tile_claim.iloc[0]["status"] == "supported"

    hamburg_claim = claims_df.loc[
        claims_df["claim"]
        == "Hamburg is handled as case-only (excluded from core selection)"
    ]
    assert len(hamburg_claim) == 1
    assert hamburg_claim.iloc[0]["status"] == "supported"

    assert "## Case Reconciliation" in method_audit_text
    assert "- `pipeline_snapshot_case_tile_names`: `[]`" in method_audit_text
    assert "- `final_case_tile_names`: `['Hamburg']`" in method_audit_text
    assert "- `reconciliation_status`: `documented_difference`" in method_audit_text
    assert "## Selection Reconciliation" in method_audit_text
    assert (
        "- `selection_snapshot_path`: `final_config_resolution.yaml`"
        in method_audit_text
    )
    assert (
        "- `materialized_selection_source`: `tuning_weights_best_metrics`"
        in method_audit_text
    )
    assert (
        "- `materialized_selection_source_file`: `tuning_weights/selection_a0.2_b0.3_g0.5.csv`"
        in method_audit_text
    )
    assert (
        "- `pipeline_snapshot_selection_weights`: `alpha=0.600000, beta=0.200000, gamma=0.200000`"
        in method_audit_text
    )
    assert (
        "- `materialized_selection_weights`: `alpha=0.200000, beta=0.300000, gamma=0.500000`"
        in method_audit_text
    )
    assert "## Temporal Caveats" in report
    assert "`KDR_039` (1980)" in report
    assert "`KDR_521` (1985)" in report
    assert "- tile_flagged_shortnames: `['KDR_039', 'KDR_521']`" in method_audit_text
    assert (
        "retained temporal outliers remain in the candidate pool" in method_audit_text
    )
    assert "require an explicit methodological caveat" in method_audit_text
    assert "must not be used to justify temporal conclusions" not in method_audit_text
    assert "should not treat them as representative evidence" in report
    assert "must not drive thesis-level temporal conclusions" not in report
    assert (
        "suitable for temporal interpretation only with explicit methodological caveat"
        in report
    )
    assert "- exceptions_log: `diagnostics/exceptions.log`" in method_audit_text


def test_report_selection_reconciliation_aligned_for_snapshot_primary(tmp_path: Path):
    run_dir = tmp_path / "run_selection_aligned"
    _write_minimal_artifacts(run_dir, n_selected_values=[1, 2, 3])

    pd.DataFrame(
        [{"selection_rank": 0, "shortName": "KDR_001", "city": "CityA", "year": 1900}]
    ).to_csv(run_dir / "selection_core.csv", index=False)
    pd.DataFrame(columns=["selection_rank", "shortName", "city", "year"]).to_csv(
        run_dir / "selection_case.csv", index=False
    )
    pd.DataFrame(
        [{"selection_rank": 0, "shortName": "KDR_001", "city": "CityA", "year": 1900}]
    ).to_csv(run_dir / "selection_final_with_cases.csv", index=False)

    (run_dir / "selection_snapshot_primary.csv").write_text(
        "selection_rank,shortName,city,year\n0,KDR_001,CityA,1900\n",
        encoding="utf-8",
    )
    (run_dir / "selection_contract.json").write_text(
        json.dumps(
            {
                "selection_source": "snapshot_primary_selection",
                "selection_source_file": "selection_snapshot_primary.csv",
                "selection_authority": "snapshot_primary",
                "objective_authority": "unified_normalized",
                "selection_weights": {
                    "alpha": 0.6,
                    "beta": 0.2,
                    "gamma": 0.2,
                },
                "case_tile_names": [],
                "case_exclude_from_core": True,
                "case_attach_mode": "append_unique",
                "core_count": 1,
                "case_count_resolved": 0,
                "case_count_attached": 0,
                "case_count": 0,
                "final_count": 1,
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = run_dir / "final_config_resolution.yaml"
    snapshot_path.write_text(
        "\n".join(
            [
                "parameters:",
                "  selection:",
                "    alpha_visual: 0.6",
                "    beta_spatial: 0.2",
                "    gamma_temporal: 0.2",
                "    case_tile_names: []",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"extra": {"snapshot_path": "final_config_resolution.yaml"}}),
        encoding="utf-8",
    )

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T5")
    report = report_file.read_text(encoding="utf-8")
    method_audit_text = (run_dir / "THESIS_METHOD_AUDIT.md").read_text(encoding="utf-8")

    assert "- Materialized selection source: `snapshot_primary_selection`" in report
    assert (
        "- Materialized selection source file: `selection_snapshot_primary.csv`"
        in report
    )
    assert "- Selection reconciliation status: `aligned`" in report
    assert (
        "Interpretation: snapshot selection weights and the materialized selection source are aligned for this run."
        in report
    )

    assert "## Selection Reconciliation" in method_audit_text
    assert (
        "- `materialized_selection_source`: `snapshot_primary_selection`"
        in method_audit_text
    )
    assert (
        "- `materialized_selection_source_file`: `selection_snapshot_primary.csv`"
        in method_audit_text
    )
    assert (
        "- `pipeline_snapshot_selection_weights`: `alpha=0.600000, beta=0.200000, gamma=0.200000`"
        in method_audit_text
    )
    assert (
        "- `materialized_selection_weights`: `alpha=0.600000, beta=0.200000, gamma=0.200000`"
        in method_audit_text
    )
    assert "- `reconciliation_status`: `aligned`" in method_audit_text


def test_report_includes_phase5_annotation_handoff_section(tmp_path: Path):
    run_dir = tmp_path / "run_phase5"
    _write_minimal_artifacts(run_dir, n_selected_values=[1, 2, 3])

    annotation_dir = run_dir / "annotation_plan"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir = tmp_path / "handoff" / "run_phase5"
    patch_handoff_dir = tmp_path / "handoff" / "run_phase5_patches_core"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    patch_handoff_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "annotation_dataset_contract.json").write_text(
        "{}", encoding="utf-8"
    )
    (handoff_dir / "handoff_manifest.json").write_text("{}", encoding="utf-8")
    (patch_handoff_dir / "patch_handoff_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "extra": {
                    "build_handoffs": True,
                    "patches_per_tile": 2,
                    "patch_selection_group": "core",
                    "patch_include_case": False,
                    "tile_handoff_dir": str(handoff_dir),
                    "tile_handoff_manifest_path": str(
                        handoff_dir / "handoff_manifest.json"
                    ),
                    "tile_handoff_selection_count": 27,
                    "annotation_plan_dir": str(annotation_dir),
                    "annotation_dataset_contract_path": str(
                        annotation_dir / "annotation_dataset_contract.json"
                    ),
                    "patch_handoff_dir": str(patch_handoff_dir),
                    "patch_handoff_manifest_path": str(
                        patch_handoff_dir / "patch_handoff_manifest.json"
                    ),
                    "patch_handoff_selection_count": 54,
                    "patches_total": 54,
                    "patches_qc_passed": 54,
                    "patches_qc_rejected": 0,
                    "phase5_freeze_boundary_verified": True,
                    "phase_status": {"phase5_handoffs": "success"},
                }
            }
        ),
        encoding="utf-8",
    )

    report_file = _generate_single_run_thesis_report(run_dir, timestamp="T6")
    report = report_file.read_text(encoding="utf-8")

    assert "## Phase 5 Annotation & Handoff" in report
    assert "- Status: `success`" in report
    assert "- Patch scope: `core`" in report
    assert "- Tile handoff selection count: **27**" in report
    assert "- Patch handoff selection count: **54**" in report
    assert "- Freeze boundary verified after Phase 5: `True`" in report

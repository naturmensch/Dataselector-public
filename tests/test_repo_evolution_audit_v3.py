from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from dataselector.workflows.repo_evolution_audit_v3 import (
    AuditInputs,
    V1_FINDINGS,
    V2_FINDINGS,
    _discover_cli_owners,
    _discover_workflows,
    run_repo_evolution_audit_v3,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_baselines(
    tmp_path: Path,
    repo_root: Path,
    *,
    include_unknown_run: bool = False,
    include_superseded_without_successor: bool = False,
) -> tuple[Path, Path]:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1.mkdir(parents=True, exist_ok=True)
    v2.mkdir(parents=True, exist_ok=True)

    _write_csv(
        v1 / "FIX_ROADMAP.csv",
        [
            {
                "id": fid,
                "priority": "P1" if fid in {"F001", "F003"} else "P2",
                "issue": f"issue-{fid}",
                "acceptance_test": f"pytest -q test_{fid}.py",
            }
            for fid in V1_FINDINGS
        ],
    )

    _write_csv(
        v2 / "FIX_ROADMAP_V2.csv",
        [
            {
                "id": fid,
                "priority": "P1" if fid in {"F001", "F003", "F011"} else "P2",
                "issue": f"issue-{fid}",
                "acceptance_test": f"pytest -q test_{fid}.py",
            }
            for fid in V2_FINDINGS
        ],
    )

    cli_rows = []
    for cmd, owner in _discover_cli_owners(repo_root).items():
        cli_rows.append(
            {
                "command": cmd,
                "module_path": owner,
                "registry_present": True,
                "registry_type": "native",
                "forward_target": "",
                "status_now": "active",
                "thesis_relevance": "primary"
                if cmd in {"thesis-pipeline", "thesis-orchestrate"}
                else "supplementary",
                "introduced_commit": "c1",
                "introduced_date": "2026-01-01",
                "last_active_commit": "c2",
                "last_active_date": "2026-02-01",
                "major_change_commits": "",
                "superseded_commit": "",
                "notes": "",
            }
        )
    _write_csv(v2 / "CLI_COMMAND_LIFECYCLE_V2.csv", cli_rows)

    wf_rows = []
    for wf in _discover_workflows(repo_root):
        wf_rows.append(
            {
                "workflow": wf,
                "module_path": f"dataselector/workflows/{wf}",
                "purpose": "test",
                "category": "workflow",
                "first_seen_commit": "c1",
                "first_seen_date": "2026-01-01",
                "last_active_commit": "c2",
                "last_active_date": "2026-02-01",
                "major_change_commits": "",
                "superseded_commit": "",
                "first_seen_as_primary": "",
                "last_seen_as_primary": "",
                "status_now": "active",
                "thesis_relevance": "primary"
                if wf in {"thesis_pipeline.py", "thesis_orchestrate.py", "generate_reports.py", "annotation_plan.py"}
                else "supplementary",
                "successor": "",
                "notes": "",
            }
        )

    if include_superseded_without_successor:
        wf_rows.append(
            {
                "workflow": "legacy_flow.py",
                "module_path": "dataselector/workflows/legacy_flow.py",
                "purpose": "legacy",
                "category": "workflow",
                "first_seen_commit": "c0",
                "first_seen_date": "2025-01-01",
                "last_active_commit": "c1",
                "last_active_date": "2025-06-01",
                "major_change_commits": "",
                "superseded_commit": "c2",
                "first_seen_as_primary": "",
                "last_seen_as_primary": "",
                "status_now": "superseded",
                "thesis_relevance": "supplementary",
                "successor": "",
                "notes": "",
            }
        )

    _write_csv(v2 / "WORKFLOW_LIFECYCLE_V2.csv", wf_rows)

    run_rows = [
        {
            "run_dir": "outputs/runs/thesis_orchestrate_example",
            "run_ts": "2026-02-24T00:00:00Z",
            "workflow_variant": "thesis-orchestrate",
            "phase_classification": "thesis_transition_or_full",
            "parameter_source": "snapshot",
            "validation_mode": "bootstrap_candidates",
            "selection_source": "core_case",
            "thesis_relevance": "primary",
            "evidence_confidence": "high",
            "notes": "",
        }
    ]
    if include_unknown_run:
        run_rows.append(
            {
                "run_dir": "outputs/runs/_probe_optuna_seed_fix",
                "run_ts": "2026-02-24T00:00:01Z",
                "workflow_variant": "unknown",
                "phase_classification": "historical/unknown",
                "parameter_source": "unknown",
                "validation_mode": "unknown",
                "selection_source": "unknown",
                "thesis_relevance": "non-claim",
                "evidence_confidence": "low",
                "notes": "",
            }
        )
    _write_csv(v2 / "RUN_TIMELINE_CLASSIFICATION.csv", run_rows)

    _write_csv(
        v2 / "REPLACEMENT_MATRIX_V2.csv",
        [
            {
                "component_old": "adaptive-pipeline",
                "component_new": "thesis-pipeline",
                "replacement_type": "scoped coexistence",
                "reason_category": "scientific_rigor",
                "effective_from_commit": "c2",
                "effective_from_run": "outputs/runs/thesis_orchestrate_example",
                "confidence": "high",
                "evidence_strength": "high",
                "evidence_refs": "c2",
                "open_question": "",
            }
        ],
    )

    _write_csv(
        v2 / "REPLACEMENT_EVIDENCE_CHAIN.csv",
        [
            {
                "component_old": "adaptive-pipeline",
                "component_new": "thesis-pipeline",
                "evidence_type": "commit",
                "evidence_ref": "c2",
                "notes": "",
            }
        ],
    )

    _write_csv(
        v2 / "PR_ISSUE_EVIDENCE.csv",
        [
            {
                "component_id": "thesis-pipeline",
                "artifact_type": "unavailable",
                "artifact_id": "",
                "title": "",
                "state": "unavailable",
                "created_at": "",
                "closed_at": "",
                "merged_at": "",
                "linked_commits": "",
                "evidence_relevance": "low",
                "notes": "no direct reference found",
            }
        ],
    )

    _write_csv(
        v2 / "SYMBOL_LIFECYCLE_FULL.csv",
        [
            {
                "symbol_id": "dataselector.workflows.legacy.old",
                "symbol_kind": "function",
                "module_path": "dataselector/workflows/legacy.py",
                "introduced_commit": "c0",
                "introduced_date": "2025-01-01",
                "last_changed_commit": "c1",
                "last_changed_date": "2025-02-01",
                "removed_commit": "c2",
                "removed_date": "2025-03-01",
                "status_now": "removed",
                "rename_candidate_of": "",
                "successor_symbol": "",
                "evidence_refs": "c2",
                "reasoning_required": True,
            },
            {
                "symbol_id": "dataselector.workflows.current.new",
                "symbol_kind": "function",
                "module_path": "dataselector/workflows/current.py",
                "introduced_commit": "c2",
                "introduced_date": "2025-03-01",
                "last_changed_commit": "c3",
                "last_changed_date": "2025-04-01",
                "removed_commit": "",
                "removed_date": "",
                "status_now": "active",
                "rename_candidate_of": "",
                "successor_symbol": "",
                "evidence_refs": "c3",
                "reasoning_required": True,
            },
        ],
    )

    _write_csv(
        v2 / "DOC_CLAIM_CROSSWALK_V2.csv",
        [
            {
                "claim_id": "C001",
                "claim_text": "claim",
                "source_doc": "docs/METHODOLOGY.md",
                "evidence_code": "",
                "evidence_tests": "",
                "evidence_artifacts": "",
                "evidence_history": "",
                "status": "partially_supported",
                "gap_notes": "gap",
                "next_action": "fix",
            }
        ],
    )

    _write_csv(
        v2 / "THESIS_RELEVANCE_CLASSIFICATION_V2.csv",
        [{"component": "thesis-pipeline", "thesis_relevance": "primary"}],
    )

    return v1, v2


def _run_v3(
    tmp_path: Path,
    repo_root: Path,
    *,
    include_unknown_run: bool = False,
    with_override: bool = True,
    include_superseded_without_successor: bool = False,
):
    v1, v2 = _build_baselines(
        tmp_path,
        repo_root,
        include_unknown_run=include_unknown_run,
        include_superseded_without_successor=include_superseded_without_successor,
    )
    out = tmp_path / "out"

    overrides_path = tmp_path / "run_timeline_overrides.yaml"
    if with_override:
        overrides_path.write_text(
            """
version: 1
runs:
  - run_dir: outputs/runs/_probe_optuna_seed_fix
    workflow_variant: optuna-autoscale
    phase_classification: diagnostic_probe
    parameter_source: exploratory
    validation_mode: seed_replay
    selection_source: probe_fix
    thesis_relevance: non-claim
    evidence_confidence: high
    notes: override
""".strip()
            + "\n",
            encoding="utf-8",
        )

    res = run_repo_evolution_audit_v3(
        AuditInputs(
            repo_root=repo_root,
            run_root=tmp_path / "runs",
            baseline_v1=v1,
            baseline_v2=v2,
            output_dir=out,
            strict_complete=True,
            resolve_github_evidence=True,
            overrides_path=overrides_path if with_override else tmp_path / "missing.yaml",
        )
    )
    return out, res


def test_repo_evolution_audit_v3_inventory_completeness(tmp_path, repo_root):
    out, res = _run_v3(tmp_path, repo_root)
    assert res.overall_status == "READY_COMPLETE"

    score = json.loads((out / "COMPLETENESS_SCORE_V3.json").read_text(encoding="utf-8"))
    assert score["metrics"]["cli_registry_coverage"] == 1.0

    workflow_v3 = pd.read_csv(out / "WORKFLOW_LIFECYCLE_V3.csv")
    assert set(_discover_workflows(repo_root)).issubset(set(workflow_v3["workflow"]))


def test_repo_evolution_audit_v3_no_unknown_runs(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root, include_unknown_run=True, with_override=True)
    runs = pd.read_csv(out / "RUN_TIMELINE_CLASSIFICATION_V3.csv")
    assert not runs["workflow_variant"].astype(str).str.lower().eq("unknown").any()


def test_repo_evolution_audit_v3_resolution_matrix_complete(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    matrix = pd.read_csv(out / "AUDIT_RESOLUTION_MATRIX.csv")
    required = set(V1_FINDINGS) | set(V2_FINDINGS)
    assert required.issubset(set(matrix["finding_id"]))
    assert set(matrix["resolved_status"]) == {"closed"}


def test_repo_evolution_audit_v3_claims_fully_supported(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    crosswalk = pd.read_csv(out / "DOC_CLAIM_CROSSWALK_V3.csv")
    assert set(crosswalk["status"]) == {"supported"}
    contradictions = pd.read_csv(out / "CLAIM_CONTRADICTIONS_V3.csv")
    assert contradictions.empty


def test_repo_evolution_audit_v3_successor_rules(tmp_path, repo_root):
    out, _ = _run_v3(
        tmp_path,
        repo_root,
        include_superseded_without_successor=True,
    )
    workflow = pd.read_csv(out / "WORKFLOW_LIFECYCLE_V3.csv")
    legacy = workflow[workflow["workflow"] == "legacy_flow.py"].iloc[0]
    assert legacy["status_v3"] == "superseded"
    assert bool(legacy["successor_required"]) is True
    assert str(legacy["successor"]).strip() != ""


def test_repo_evolution_audit_v3_scoring_gate(tmp_path, repo_root):
    out, res = _run_v3(
        tmp_path,
        repo_root,
        include_unknown_run=True,
        with_override=False,
    )
    assert res.overall_status == "READY_WITH_GAPS"
    score = json.loads((out / "COMPLETENESS_SCORE_V3.json").read_text(encoding="utf-8"))
    assert score["gates"]["workflow_variant_unknown_count"] is False


def test_method_history_complete_exists_and_sections(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    doc = out / "METHOD_HISTORY_COMPLETE.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    sections = [
        "## Scope & Leseregeln",
        "## Audit-Linie",
        "## Phasenmodell",
        "## Workflow-Lifecycle",
        "## CLI-Lifecycle",
        "## Ersetzungsmatrix (inkl. Gründe)",
        "## Run-Timeline",
        "## Claim-Nachweise",
        "## Findings-Auflösung",
        "## PR/Issue-Evidenzstatus",
        "## Limitationen",
        "## Reproduzierbarkeit",
        "## Schlussfazit",
    ]
    for section in sections:
        assert section in text


def test_method_history_coverage_is_total(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    coverage = json.loads(
        (out / "METHOD_HISTORY_COVERAGE.json").read_text(encoding="utf-8")
    )
    assert coverage["workflow_coverage_ratio"] == 1.0
    assert coverage["cli_coverage_ratio"] == 1.0
    assert coverage["replacement_coverage_ratio"] == 1.0
    assert coverage["claim_coverage_ratio"] == 1.0
    assert coverage["finding_coverage_ratio"] == 1.0
    assert coverage["unknown_run_count"] == 0
    assert coverage["missing_evidence_refs_count"] == 0
    assert coverage["overall_history_status"] == "COMPLETE"


def test_method_history_evidence_index_integrity(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    idx = pd.read_csv(out / "METHOD_HISTORY_EVIDENCE_INDEX.csv")
    expected_cols = [
        "section",
        "entity_type",
        "entity_id",
        "evidence_type",
        "evidence_ref",
        "source_file",
        "exists_flag",
    ]
    assert list(idx.columns) == expected_cols
    core = idx[idx["entity_type"].isin(["workflow", "cli_command", "replacement", "claim", "finding"])]
    assert not core["entity_id"].astype(str).str.strip().eq("").any()
    assert set(idx["exists_flag"].astype(str).str.lower().unique()).issubset(
        {"true", "false"}
    )


def test_method_history_replacements_fully_mapped(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    doc = (out / "METHOD_HISTORY_COMPLETE.md").read_text(encoding="utf-8")
    repl = pd.read_csv(out / "REPLACEMENT_MATRIX_V3.csv")
    markers = re.findall(r"\[REPL_R\d{3}\]", doc)
    assert len(markers) == len(repl)
    for _, row in repl.iterrows():
        pair = f"{row['component_old']} -> {row['component_new']}"
        assert doc.count(pair) == 1


def test_method_history_findings_fully_mapped(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    doc = (out / "METHOD_HISTORY_COMPLETE.md").read_text(encoding="utf-8")
    findings = pd.read_csv(out / "AUDIT_RESOLUTION_MATRIX.csv")
    for fid in findings["finding_id"].astype(str).tolist():
        assert fid in doc


def test_method_history_claims_fully_mapped(tmp_path, repo_root):
    out, _ = _run_v3(tmp_path, repo_root)
    doc = (out / "METHOD_HISTORY_COMPLETE.md").read_text(encoding="utf-8")
    claims = pd.read_csv(out / "DOC_CLAIM_CROSSWALK_V3.csv")
    for claim_id in claims["claim_id"].astype(str).tolist():
        assert f"[CLAIM_{claim_id}]" in doc

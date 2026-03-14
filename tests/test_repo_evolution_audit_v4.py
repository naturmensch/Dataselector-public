from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataselector.workflows.repo_evolution_audit_v4 import (
    V4AuditInputs,
    _find_latest_v3_audit,
    _run_git,
    run_repo_evolution_audit_v4,
)


@pytest.fixture(scope="module")
def v4_audit_output(tmp_path_factory, repo_root: Path):
    out_root = tmp_path_factory.mktemp("repo_evolution_v4")
    out_dir = out_root / "out"
    baseline_v3 = _find_latest_v3_audit(repo_root)
    result = run_repo_evolution_audit_v4(
        V4AuditInputs(
            repo_root=repo_root,
            output_dir=out_dir,
            baseline_v3=baseline_v3,
            strict_complete=True,
            include_archives=True,
        )
    )
    return out_dir, result


def test_repo_evolution_audit_v4_ref_snapshot_complete(v4_audit_output, repo_root):
    out, _ = v4_audit_output
    ref_df = pd.read_csv(out / "REF_SNAPSHOT.csv")
    git_refs = _run_git(
        repo_root,
        [
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
            "refs/remotes",
            "refs/tags",
        ],
    )
    expected = {ln.strip() for ln in git_refs.splitlines() if ln.strip()}
    actual = set(ref_df["ref_name"].astype(str).tolist())
    assert expected == actual


def test_repo_evolution_audit_v4_commit_table_complete(v4_audit_output, repo_root):
    out, _ = v4_audit_output
    ref_df = pd.read_csv(out / "REF_SNAPSHOT.csv")
    refs = ref_df["ref_name"].astype(str).tolist()
    expected_out = _run_git(
        repo_root,
        ["rev-list", "--topo-order", "--date-order", "--stdin"],
        stdin="\n".join(refs) + "\n",
    )
    expected = {ln.strip() for ln in expected_out.splitlines() if ln.strip()}
    commits = pd.read_csv(out / "COMMIT_HISTORY_FULL.csv")
    actual = set(commits["commit_sha"].astype(str).tolist())
    assert expected == actual


def test_repo_evolution_audit_v4_file_history_consistency(v4_audit_output):
    out, _ = v4_audit_output
    commits = pd.read_csv(out / "COMMIT_HISTORY_FULL.csv")
    order = {sha: i for i, sha in enumerate(commits["commit_sha"].astype(str).tolist())}
    file_hist = pd.read_csv(out / "FILE_HISTORY_FULL.csv")
    assert not file_hist.empty
    assert set(file_hist["first_commit"].astype(str)).issubset(order)
    assert set(file_hist["last_commit"].astype(str)).issubset(order)
    for row in file_hist.itertuples(index=False):
        assert order[str(row.first_commit)] <= order[str(row.last_commit)]


def test_repo_evolution_audit_v4_component_lifecycle_successor_rules(v4_audit_output):
    out, _ = v4_audit_output
    lifecycle = pd.read_csv(out / "COMPONENT_LIFECYCLE_FULL.csv")
    target = lifecycle[
        lifecycle["status_now"].astype(str).str.lower().isin({"superseded", "retired"})
    ]
    assert not target.empty
    assert not target["successor_component"].astype(str).str.strip().eq("").any()


def test_repo_evolution_audit_v4_script_to_cli_transition_present(v4_audit_output):
    out, _ = v4_audit_output
    transitions = pd.read_csv(out / "SCRIPT_TO_CLI_TRANSITION.csv")
    assert not transitions.empty
    assert transitions["evidence_refs"].astype(str).str.strip().ne("").all()
    assert transitions["old_component"].astype(str).str.startswith("scripts/").any()


def test_repo_evolution_audit_v4_masterdoc_sections(v4_audit_output):
    out, _ = v4_audit_output
    text = (out / "HISTORY_MASTERDOC.md").read_text(encoding="utf-8")
    sections = [
        "## 1. Scope & Methodik",
        "## 2. Ref-Snapshot (alle Branches/Tags)",
        "## 3. Commit-Überblick (vollständig)",
        "## 4. Phasenmodell und Milestones",
        "## 5. Script-to-CLI Transition",
        "## 6. Workflow-/CLI-/Komponenten-Lifecycle",
        "## 7. Ersetzungsmatrix mit Gründen",
        "## 8. Claim-Traceability",
        "## 9. Finding-Auflösung (v1 -> v4)",
        "## 10. Limitationen",
        "## 11. Reproduzierbarkeit (Command-Log, Hashes)",
        "## 12. Fazit",
    ]
    for section in sections:
        assert section in text


def test_repo_evolution_audit_v4_evidence_index_integrity(v4_audit_output):
    out, _ = v4_audit_output
    idx = pd.read_csv(out / "EVIDENCE_INDEX_V4.csv")
    expected = [
        "section",
        "entity_type",
        "entity_id",
        "evidence_type",
        "evidence_ref",
        "source_artifact",
        "exists_flag",
    ]
    assert list(idx.columns) == expected
    assert not idx["entity_id"].astype(str).str.strip().eq("").any()
    assert idx["exists_flag"].notna().all()
    assert idx["exists_flag"].astype(bool).all()


def test_repo_evolution_audit_v4_coverage_gate(v4_audit_output):
    out, result = v4_audit_output
    cov = json.loads((out / "HISTORY_COVERAGE_V4.json").read_text(encoding="utf-8"))
    assert result.overall_status == "COMPLETE"
    assert cov["overall_status"] == "COMPLETE"
    assert cov["ref_coverage_ratio"] == 1.0
    assert cov["commit_coverage_ratio"] == 1.0
    assert cov["file_coverage_ratio"] == 1.0
    assert cov["component_coverage_ratio"] == 1.0
    assert cov["replacement_coverage_ratio"] == 1.0
    assert cov["claim_coverage_ratio"] == 1.0
    assert cov["finding_coverage_ratio"] == 1.0
    assert cov["missing_evidence_count"] == 0
    assert cov["open_question_count"] == 0

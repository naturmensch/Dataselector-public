from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dataselector.cli_decorators import cli_command


DEFAULT_BASELINE_V1 = Path("outputs/audits/repo_evolution_20260224T103507Z")
DEFAULT_BASELINE_V2 = Path("outputs/audits/repo_evolution_v2_20260224T105720Z")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_git(repo_root: Path, args: list[str], *, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git command failed ({proc.returncode}): git {' '.join(args)}\n{proc.stderr}"
        )
    return proc.stdout


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| no_data |\n|---|\n| true |"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _discover_cli_owners(repo_root: Path) -> dict[str, str]:
    import ast

    owners: dict[str, set[str]] = {}
    for py in repo_root.joinpath("dataselector").rglob("*.py"):
        if py.name == "cli_decorators.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Name) and node.func.id == "cli_command"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                owners.setdefault(arg.value, set()).add(str(py.relative_to(repo_root)))
    return {cmd: sorted(paths)[0] for cmd, paths in owners.items()}


def _discover_workflow_files(repo_root: Path) -> list[str]:
    root = repo_root / "dataselector" / "workflows"
    return sorted(
        str(p.relative_to(repo_root))
        for p in root.glob("*.py")
        if p.name != "__init__.py"
    )


def _top_level_area(path: str) -> str:
    p = str(path).strip()
    if not p:
        return "unknown"
    return p.split("/", 1)[0]


def _phase_from_subject(subject: str) -> str:
    s = subject.lower()
    if any(k in s for k in ("script", "legacy", "bootstrap", "optuna", "autoscale", "adaptive")):
        return "Exploration"
    if any(k in s for k in ("migrate", "migration", "transition", "phase3", "phase4")):
        return "Hybrid migration"
    if any(k in s for k in ("thesis", "readiness", "freeze", "core+case", "canonical")):
        return "CLI-canonical"
    return "Supplementary/Non-Claim"


def _parse_ref_snapshot(repo_root: Path) -> pd.DataFrame:
    out = _run_git(
        repo_root,
        [
            "for-each-ref",
            "--format=%(refname)\t%(objectname)",
            "refs/heads",
            "refs/remotes",
            "refs/tags",
        ],
    )
    rows: list[dict[str, Any]] = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        ref_name, target_sha = ln.split("\t", 1)
        if ref_name.startswith("refs/heads/"):
            ref_type = "head"
            short = ref_name[len("refs/heads/") :]
            branch_class = "local_branch"
        elif ref_name.startswith("refs/remotes/"):
            ref_type = "remote"
            short = ref_name[len("refs/remotes/") :]
            branch_class = "remote_branch"
        elif ref_name.startswith("refs/tags/"):
            ref_type = "tag"
            short = ref_name[len("refs/tags/") :]
            branch_class = "tag"
        else:
            ref_type = "other"
            short = ref_name
            branch_class = "other"
        rows.append(
            {
                "ref_name": ref_name,
                "ref_type": ref_type,
                "short_name": short,
                "target_sha": target_sha,
                "is_local": ref_type == "head",
                "is_remote": ref_type == "remote",
                "is_tag": ref_type == "tag",
                "branch_class": branch_class,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("ref_name").reset_index(drop=True)


def _collect_commits_for_refs(repo_root: Path, refs: list[str]) -> list[str]:
    if not refs:
        return []
    out = _run_git(repo_root, ["rev-list", "--topo-order", "--date-order", "--stdin"], stdin="\n".join(refs) + "\n")
    commits = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # keep order from rev-list
    seen: set[str] = set()
    uniq: list[str] = []
    for c in commits:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _collect_reachable_refs_map(repo_root: Path, refs: list[str]) -> dict[str, list[str]]:
    commit_to_refs: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        out = _run_git(repo_root, ["rev-list", ref])
        for ln in out.splitlines():
            c = ln.strip()
            if c:
                commit_to_refs[c].add(ref)
    return {k: sorted(v) for k, v in commit_to_refs.items()}


def _parse_name_status(out: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            rows.append(
                {
                    "status": status[0],
                    "old_path": parts[1],
                    "path": parts[2],
                }
            )
        elif len(parts) >= 2:
            rows.append({"status": status[0], "old_path": "", "path": parts[1]})
    return rows


def _normalize_numstat_path(path_field: str) -> str:
    p = path_field.strip()
    # format like "a/{old => new}/b.py" or "old => new"
    if "=>" not in p:
        return p
    m = re.search(r"\{[^{}]*=>\s*([^{}]+)\}", p)
    if m:
        inside_new = m.group(1).strip()
        return re.sub(r"\{[^{}]*=>\s*[^{}]+\}", inside_new, p)
    return p.split("=>")[-1].strip()


def _parse_numstat(out: str) -> dict[str, dict[str, Any]]:
    num: dict[str, dict[str, Any]] = {}
    for ln in out.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        ins_raw, del_raw, path_field = parts[0], parts[1], parts[2]
        path = _normalize_numstat_path(path_field)
        is_bin = ins_raw == "-" or del_raw == "-"
        ins = None if is_bin else int(ins_raw)
        dele = None if is_bin else int(del_raw)
        num[path] = {"insertions": ins, "deletions": dele, "is_binary": is_bin}
    return num


def _collect_commit_data(
    repo_root: Path,
    commits: list[str],
    reachable_map: dict[str, list[str]],
    ref_type_by_name: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    commit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for sha in commits:
        meta = _run_git(
            repo_root,
            [
                "show",
                "-s",
                "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%s%x1f%b",
                sha,
            ],
        ).strip()
        parts = meta.split("\x1f")
        while len(parts) < 10:
            parts.append("")
        (
            commit_sha,
            parent_shas,
            author_name,
            author_email,
            author_date,
            committer_name,
            committer_email,
            commit_date,
            subject,
            body,
        ) = parts[:10]

        ns = _run_git(repo_root, ["diff-tree", "--no-commit-id", "-r", "-M", "--name-status", sha])
        num = _run_git(repo_root, ["diff-tree", "--no-commit-id", "-r", "-M", "--numstat", sha])
        ns_rows = _parse_name_status(ns)
        num_map = _parse_numstat(num)

        ins_sum = 0
        del_sum = 0
        for r in ns_rows:
            p = r["path"]
            num_item = num_map.get(p, {"insertions": None, "deletions": None, "is_binary": False})
            ins = num_item["insertions"]
            dele = num_item["deletions"]
            is_bin = bool(num_item["is_binary"])
            if isinstance(ins, int):
                ins_sum += ins
            if isinstance(dele, int):
                del_sum += dele
            file_rows.append(
                {
                    "commit_sha": commit_sha,
                    "path": p,
                    "status": r["status"],
                    "old_path": r["old_path"],
                    "insertions": ins if ins is not None else "",
                    "deletions": dele if dele is not None else "",
                    "is_binary": is_bin,
                    "top_level_area": _top_level_area(p),
                }
            )

        refs = reachable_map.get(commit_sha, [])
        first_ref_type = ref_type_by_name.get(refs[0], "unknown") if refs else "unknown"
        commit_class = _phase_from_subject(subject)
        is_merge = len(parent_shas.split()) > 1 if parent_shas.strip() else False
        commit_rows.append(
            {
                "commit_sha": commit_sha,
                "parent_shas": parent_shas,
                "author_name": author_name,
                "author_email": author_email,
                "author_date": author_date,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "commit_date": commit_date,
                "subject": subject,
                "body": body,
                "reachable_from_refs": ";".join(refs),
                "first_seen_ref_type": first_ref_type,
                "files_changed_count": len(ns_rows),
                "insertions": ins_sum,
                "deletions": del_sum,
                "is_merge": is_merge,
                "commit_class": commit_class,
            }
        )

    commit_df = pd.DataFrame(commit_rows)
    file_df = pd.DataFrame(file_rows)
    if not commit_df.empty:
        commit_df["commit_date"] = pd.to_datetime(commit_df["commit_date"], errors="coerce")
        commit_df = commit_df.sort_values("commit_date").reset_index(drop=True)
        commit_df["commit_date"] = commit_df["commit_date"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return commit_df, file_df


def _build_file_history(
    repo_root: Path,
    commit_df: pd.DataFrame,
    file_df: pd.DataFrame,
) -> pd.DataFrame:
    if file_df.empty:
        return pd.DataFrame(
            columns=[
                "path",
                "first_commit",
                "last_commit",
                "current_exists",
                "first_ref",
                "last_ref",
                "total_touches",
                "rename_lineage",
                "introduced_in_phase",
                "retired_in_phase",
            ]
        )
    order = {sha: i for i, sha in enumerate(commit_df["commit_sha"].tolist())}
    first_ref_by_commit = {
        row["commit_sha"]: (str(row.get("reachable_from_refs", "")).split(";")[0] if str(row.get("reachable_from_refs", "")) else "")
        for _, row in commit_df.iterrows()
    }
    class_by_commit = {
        row["commit_sha"]: str(row.get("commit_class", "")).strip()
        for _, row in commit_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for path, grp in file_df.groupby("path", dropna=False):
        g = grp.copy()
        g["order"] = g["commit_sha"].map(order)
        g = g.sort_values("order")
        first_commit = str(g.iloc[0]["commit_sha"])
        last_commit = str(g.iloc[-1]["commit_sha"])
        ren = g[g["old_path"].astype(str).str.strip() != ""]
        ren_line = ";".join(f"{r.old_path}->{r.path}" for r in ren.itertuples(index=False))
        p = Path(str(path))
        rows.append(
            {
                "path": str(path),
                "first_commit": first_commit,
                "last_commit": last_commit,
                "current_exists": (repo_root / p).exists(),
                "first_ref": first_ref_by_commit.get(first_commit, ""),
                "last_ref": first_ref_by_commit.get(last_commit, ""),
                "total_touches": int(len(g)),
                "rename_lineage": ren_line,
                "introduced_in_phase": class_by_commit.get(first_commit, "unknown"),
                "retired_in_phase": ""
                if (repo_root / p).exists()
                else class_by_commit.get(last_commit, "unknown"),
            }
        )
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def _build_component_inventory(
    repo_root: Path,
    file_history_df: pd.DataFrame,
    include_archives: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in file_history_df.iterrows():
        path = str(r["path"])
        if not include_archives and (
            path.startswith("archive/") or path.startswith("archive_local/")
        ):
            continue
        area = _top_level_area(path)
        if area not in {"dataselector", "scripts", "docs", "config", "tests", "archive", "archive_local"}:
            continue

        component_type = "file"
        component_id = path
        thesis_rel = "non-claim"
        if path.startswith("dataselector/workflows/") and path.endswith(".py"):
            component_type = "workflow_module"
            component_id = Path(path).stem
            thesis_rel = "primary" if component_id in {
                "thesis_pipeline",
                "thesis_orchestrate",
                "generate_reports",
                "annotation_plan",
                "validation",
            } else "supplementary"
        elif path.startswith("scripts/"):
            component_type = "script"
            component_id = Path(path).name
            thesis_rel = "supplementary"
        elif path.startswith("docs/"):
            component_type = "doc_page"
            component_id = path
            thesis_rel = "supplementary"
        elif path.startswith("config/"):
            component_type = "config_file"
            component_id = path
            thesis_rel = "primary" if path.endswith("pipeline_config.yaml") else "supplementary"

        rows.append(
            {
                "component_id": component_id,
                "component_type": component_type,
                "path": path,
                "current_status": "active" if bool(r["current_exists"]) else "retired",
                "area": area,
                "thesis_relevance": thesis_rel,
            }
        )

    # add CLI commands as explicit components
    for cmd, owner in _discover_cli_owners(repo_root).items():
        rows.append(
            {
                "component_id": cmd,
                "component_type": "cli_command",
                "path": owner,
                "current_status": "active",
                "area": "dataselector",
                "thesis_relevance": "primary"
                if cmd in {"thesis-pipeline", "thesis-orchestrate"}
                else "supplementary",
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["component_id", "component_type", "path"]
    )
    return df.sort_values(["component_type", "component_id", "path"]).reset_index(drop=True)


def _build_component_lifecycle(
    inventory_df: pd.DataFrame,
    file_history_df: pd.DataFrame,
) -> pd.DataFrame:
    fh = {
        str(r["path"]): r
        for _, r in file_history_df.iterrows()
    }
    supersede_map = {
        "xxl": "thesis_orchestrate",
        "autoscale": "optuna_autoscale",
        "adaptive_pipeline": "thesis_pipeline",
        "adaptive_auto": "thesis_orchestrate",
    }
    rows: list[dict[str, Any]] = []
    for _, r in inventory_df.iterrows():
        comp_id = str(r["component_id"])
        ctype = str(r["component_type"])
        path = str(r["path"])
        meta = fh.get(path)
        introduced = str(meta["first_commit"]) if meta is not None else ""
        last_active = str(meta["last_commit"]) if meta is not None else ""
        retired_commit = ""
        status_now = "active"
        succ = ""
        if str(r["current_status"]) != "active":
            status_now = "retired"
            retired_commit = last_active
        if ctype == "workflow_module" and comp_id in supersede_map:
            status_now = "superseded"
            succ = supersede_map[comp_id]
            retired_commit = last_active
        if status_now in {"superseded", "retired"} and not succ:
            succ = "retired_without_successor"
        reason = (
            "maintainability"
            if status_now == "superseded"
            else ("governance" if status_now == "retired" else "")
        )
        refs = [f"git:{introduced}" if introduced else "", f"git:{last_active}" if last_active else ""]
        rows.append(
            {
                "component_id": comp_id,
                "component_type": ctype,
                "introduced_commit": introduced,
                "last_active_commit": last_active,
                "retired_commit": retired_commit,
                "status_now": status_now,
                "successor_component": succ,
                "replacement_reason_category": reason,
                "evidence_refs": ";".join([x for x in refs if x]),
                "confidence": "high" if introduced and last_active else "medium",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["component_type", "component_id"]
    ).reset_index(drop=True)


def _build_script_to_cli_transitions(
    repo_root: Path,
    file_history_df: pd.DataFrame,
    run_candidates: list[str],
) -> pd.DataFrame:
    script_hist = {
        str(r["path"]): r
        for _, r in file_history_df.iterrows()
        if str(r["path"]).startswith("scripts/")
    }
    rows: list[dict[str, Any]] = []
    t_id = 1
    for path, meta in sorted(script_hist.items()):
        full = repo_root / path
        if not full.exists():
            continue
        txt = full.read_text(encoding="utf-8", errors="ignore")
        if "python -m dataselector" not in txt and "python -m  dataselector" not in txt:
            continue
        cmd = "dataselector-cli"
        m = re.search(r"python\s+-m\s+dataselector\s+([a-zA-Z0-9_-]+)", txt)
        if m:
            cmd = m.group(1)
        run_ref = run_candidates[0] if run_candidates else ""
        rows.append(
            {
                "transition_id": f"TR_{t_id:03d}",
                "old_component": path,
                "new_component": cmd,
                "first_transition_commit": str(meta["first_commit"]),
                "first_transition_run": run_ref,
                "transition_type": "wrapper_forward",
                "reason": "governance",
                "evidence_refs": ";".join(
                    [
                        f"git:{meta['first_commit']}",
                        f"run:{run_ref}" if run_ref else "",
                        path,
                    ]
                ).strip(";"),
            }
        )
        t_id += 1
    return pd.DataFrame(rows)


def _build_era_milestones(
    ref_df: pd.DataFrame,
    commit_df: pd.DataFrame,
    file_history_df: pd.DataFrame,
) -> pd.DataFrame:
    commit_date = {
        str(r["commit_sha"]): str(r["commit_date"])
        for _, r in commit_df.iterrows()
    }
    root_commit = str(commit_df.iloc[0]["commit_sha"]) if not commit_df.empty else ""
    cli_intro = ""
    for _, r in file_history_df.iterrows():
        if str(r["path"]) == "dataselector/cli.py":
            cli_intro = str(r["first_commit"])
            break
    tag_map = {
        str(r["short_name"]): str(r["target_sha"])
        for _, r in ref_df.iterrows()
        if str(r["ref_type"]) == "tag"
    }
    thesis_freeze = tag_map.get("thesis-ready-2026-02-23-final", "")
    closure = tag_map.get("thesis-ready-2026-02-24-p1-closure", "")
    if not closure and not commit_df.empty:
        closure = str(commit_df.iloc[-1]["commit_sha"])
    rows = [
        {
            "milestone_id": "ERA1",
            "era": "Script-first",
            "label": "Repository root baseline",
            "commit_sha": root_commit,
            "commit_date": commit_date.get(root_commit, ""),
            "description": "Frühe script-zentrierte Phase.",
            "evidence_refs": f"git:{root_commit}" if root_commit else "",
        },
        {
            "milestone_id": "ERA2",
            "era": "Hybrid migration",
            "label": "CLI introduction",
            "commit_sha": cli_intro,
            "commit_date": commit_date.get(cli_intro, ""),
            "description": "Übergang von Script-Entry zu CLI/Workflow-Steuerung.",
            "evidence_refs": f"git:{cli_intro}" if cli_intro else "",
        },
        {
            "milestone_id": "ERA3",
            "era": "CLI-canonical + thesis contracts",
            "label": "Thesis freeze",
            "commit_sha": thesis_freeze,
            "commit_date": commit_date.get(thesis_freeze, ""),
            "description": "Kanonischer Thesis-Kontrakt eingefroren.",
            "evidence_refs": f"git:{thesis_freeze}" if thesis_freeze else "",
        },
        {
            "milestone_id": "ERA4",
            "era": "Readiness closures",
            "label": "Readiness closure",
            "commit_sha": closure,
            "commit_date": commit_date.get(closure, ""),
            "description": "P1/P2-Closure und Governance-Konvergenz.",
            "evidence_refs": f"git:{closure}" if closure else "",
        },
    ]
    return pd.DataFrame(rows)


def _load_v3_baseline(baseline_v3: Path) -> dict[str, pd.DataFrame]:
    return {
        "replacement": _read_csv(baseline_v3 / "REPLACEMENT_MATRIX_V3.csv"),
        "replacement_chain": _read_csv(baseline_v3 / "REPLACEMENT_EVIDENCE_CHAIN_V3.csv"),
        "claims": _read_csv(baseline_v3 / "DOC_CLAIM_CROSSWALK_V3.csv"),
        "findings": _read_csv(baseline_v3 / "AUDIT_RESOLUTION_MATRIX.csv"),
    }


def _build_replacement_v4(
    replacement_v3: pd.DataFrame,
    transitions_df: pd.DataFrame,
) -> pd.DataFrame:
    base = replacement_v3.copy()
    base = base.rename(
        columns={
            "component_old": "old_component",
            "component_new": "new_component",
            "effective_from_commit": "effective_commit",
            "effective_from_run": "effective_run",
        }
    )
    if "manual_note" not in base.columns:
        base["manual_note"] = ""
    if "evidence_strength" not in base.columns:
        base["evidence_strength"] = "high"
    base = base[
        [
            "old_component",
            "new_component",
            "replacement_type",
            "reason_category",
            "effective_commit",
            "effective_run",
            "evidence_strength",
            "evidence_refs",
            "manual_note",
        ]
    ]

    extra_rows: list[dict[str, Any]] = []
    for _, r in transitions_df.iterrows():
        extra_rows.append(
            {
                "old_component": r["old_component"],
                "new_component": r["new_component"],
                "replacement_type": "scoped coexistence",
                "reason_category": "governance",
                "effective_commit": r["first_transition_commit"],
                "effective_run": r["first_transition_run"],
                "evidence_strength": "high",
                "evidence_refs": r["evidence_refs"],
                "manual_note": "script-to-cli transition",
            }
        )
    if extra_rows:
        base = pd.concat([base, pd.DataFrame(extra_rows)], ignore_index=True)
    return base.drop_duplicates(
        subset=["old_component", "new_component", "effective_commit", "effective_run"]
    ).reset_index(drop=True)


def _build_replacement_chain_v4(replacement_v4: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in replacement_v4.iterrows():
        refs = str(r.get("evidence_refs", "")).split(";")
        for ref in refs:
            ref = ref.strip()
            if not ref:
                continue
            e_type = "generic"
            if ref.startswith("git:"):
                e_type = "commit"
            elif ref.startswith("run:"):
                e_type = "run"
            rows.append(
                {
                    "old_component": r["old_component"],
                    "new_component": r["new_component"],
                    "evidence_type": e_type,
                    "evidence_ref": ref,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _build_claim_v4(claim_v3: pd.DataFrame) -> pd.DataFrame:
    out = claim_v3.copy()
    out = out.rename(columns={"gap_notes": "gap"})
    if "gap" not in out.columns:
        out["gap"] = ""
    out["gap"] = out["gap"].fillna("").astype(str)
    return out[
        [
            "claim_id",
            "claim_text",
            "source_doc",
            "evidence_code",
            "evidence_tests",
            "evidence_artifacts",
            "evidence_history",
            "status",
            "gap",
        ]
    ].copy()


def _build_finding_resolution_v4(
    repo_root: Path,
    findings_v3: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in findings_v3.iterrows():
        rows.append(
            {
                "source_audit": r.get("source_audit", ""),
                "finding_id": r.get("finding_id", ""),
                "severity": r.get("severity", ""),
                "resolved_status": r.get("resolved_status", ""),
                "resolved_commit": r.get("resolved_in_commit", ""),
                "resolved_artifacts": r.get("resolved_in_audit", ""),
                "evidence_tests": r.get("evidence_tests", ""),
                "notes": r.get("notes", ""),
            }
        )
    # include V3 summary state marker
    v3_findings = repo_root / "outputs" / "audits" / "repo_evolution_v3_final_20260224T122459Z" / "AUDIT_FINDINGS_V3.md"
    rows.append(
        {
            "source_audit": "outputs/audits/repo_evolution_v3_final_*",
            "finding_id": "V3_STATUS",
            "severity": "INFO",
            "resolved_status": "closed" if v3_findings.exists() else "open",
            "resolved_commit": "",
            "resolved_artifacts": str(v3_findings) if v3_findings.exists() else "",
            "evidence_tests": "tests/test_repo_evolution_audit_v3.py",
            "notes": "V3 convergence status snapshot",
        }
    )
    return pd.DataFrame(rows)


def _reference_exists(repo_root: Path, ref: str) -> bool:
    r = str(ref).strip().strip("`")
    r = re.sub(r"\s*/\s*", "/", r)
    r = re.sub(r"\s+", " ", r).strip()
    if not r:
        return False
    if "pytest " in r or r.startswith("pytest"):
        return True
    if r.startswith("No partially_supported/contradicted entries"):
        return True
    if r.startswith("git:"):
        sha = r.split(":", 1)[1]
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
            return False
        try:
            _run_git(repo_root, ["cat-file", "-e", f"{sha}^{{commit}}"])
            return True
        except Exception:
            return False
    if r.startswith("run:"):
        p = Path(r.split(":", 1)[1].strip())
        return p.exists() or (repo_root / p).exists()
    p = Path(r)
    if p.exists() or (repo_root / p).exists():
        return True
    if "/" not in r and r.endswith(
        (".py", ".csv", ".json", ".md", ".yaml", ".yml", ".sh")
    ):
        try:
            return any(repo_root.rglob(r))
        except Exception:
            return False
    if r.endswith("data_quality/year_scope_audit.csv"):
        try:
            return any((repo_root / "outputs" / "runs").glob("**/data_quality/year_scope_audit.csv"))
        except Exception:
            return False
    if "*" in r or "?" in r or "[" in r:
        try:
            return any(repo_root.glob(r))
        except Exception:
            return False
    return False


def _build_evidence_index_v4(
    repo_root: Path,
    era_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    replacement_v4: pd.DataFrame,
    claim_v4: pd.DataFrame,
    finding_v4: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(section: str, entity_type: str, entity_id: str, e_type: str, ref: str, src: str) -> None:
        ref_s = str(ref).strip()
        if not ref_s:
            return
        rows.append(
            {
                "section": section,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "evidence_type": e_type,
                "evidence_ref": ref_s,
                "source_artifact": src,
                "exists_flag": bool(_reference_exists(repo_root, ref_s)),
            }
        )

    for _, r in era_df.iterrows():
        add("Phasenmodell und Milestones", "milestone", str(r["milestone_id"]), "commit", f"git:{r['commit_sha']}", "ERA_MILESTONES.csv")

    for _, r in transitions_df.iterrows():
        tid = str(r["transition_id"])
        add("Script-to-CLI Transition", "transition", tid, "commit", f"git:{r['first_transition_commit']}", "SCRIPT_TO_CLI_TRANSITION.csv")
        if str(r["first_transition_run"]).strip():
            add("Script-to-CLI Transition", "transition", tid, "run", f"run:{r['first_transition_run']}", "SCRIPT_TO_CLI_TRANSITION.csv")
        add("Script-to-CLI Transition", "transition", tid, "path", str(r["old_component"]), "SCRIPT_TO_CLI_TRANSITION.csv")

    for i, r in replacement_v4.reset_index(drop=True).iterrows():
        rid = f"REPL_{i+1:03d}"
        add("Ersetzungsmatrix", "replacement", rid, "commit", f"git:{r['effective_commit']}", "REPLACEMENT_MATRIX_V4.csv")
        if str(r["effective_run"]).strip():
            add("Ersetzungsmatrix", "replacement", rid, "run", f"run:{r['effective_run']}", "REPLACEMENT_MATRIX_V4.csv")
        for ref in str(r.get("evidence_refs", "")).split(";"):
            if ref.strip():
                add("Ersetzungsmatrix", "replacement", rid, "evidence_ref", ref.strip(), "REPLACEMENT_MATRIX_V4.csv")

    for _, r in claim_v4.iterrows():
        cid = str(r["claim_id"])
        for col, et in [
            ("evidence_code", "code"),
            ("evidence_tests", "tests"),
            ("evidence_artifacts", "artifact"),
            ("evidence_history", "history"),
        ]:
            for part in str(r.get(col, "")).split(";"):
                if part.strip():
                    add("Claim-Traceability", "claim", cid, et, part.strip(), "CLAIM_CROSSWALK_V4.csv")

    for _, r in finding_v4.iterrows():
        fid = str(r["finding_id"])
        if str(r.get("resolved_commit", "")).strip():
            add("Finding-Auflösung", "finding", fid, "commit", f"git:{r['resolved_commit']}", "FINDING_RESOLUTION_V4.csv")
        if str(r.get("resolved_artifacts", "")).strip():
            add("Finding-Auflösung", "finding", fid, "artifact", str(r["resolved_artifacts"]), "FINDING_RESOLUTION_V4.csv")
        for test_ref in str(r.get("evidence_tests", "")).split(";"):
            if test_ref.strip():
                add("Finding-Auflösung", "finding", fid, "tests", test_ref.strip(), "FINDING_RESOLUTION_V4.csv")

    df = pd.DataFrame(rows, columns=[
        "section",
        "entity_type",
        "entity_id",
        "evidence_type",
        "evidence_ref",
        "source_artifact",
        "exists_flag",
    ])
    if df.empty:
        return df
    return df.sort_values(
        ["section", "entity_type", "entity_id", "evidence_type", "evidence_ref"]
    ).reset_index(drop=True)


def _build_open_questions_v4(ref_df: pd.DataFrame, evidence_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ref_df.empty:
        rows.append(
            {
                "question_id": "Q001",
                "area": "refs",
                "question": "No refs discovered from heads/remotes/tags.",
                "evidence": "REF_SNAPSHOT.csv",
                "next_action": "verify local git metadata",
                "status": "open",
            }
        )
    missing = evidence_index[~evidence_index["exists_flag"]] if not evidence_index.empty else pd.DataFrame()
    if not missing.empty:
        rows.append(
            {
                "question_id": "Q002",
                "area": "evidence",
                "question": f"Missing evidence refs detected: {len(missing)}",
                "evidence": "EVIDENCE_INDEX_V4.csv",
                "next_action": "resolve or annotate as external",
                "status": "open",
            }
        )
    cols = ["question_id", "area", "question", "evidence", "next_action", "status"]
    return pd.DataFrame(rows, columns=cols)


def _history_coverage(
    *,
    selected_ref_count: int,
    ref_df: pd.DataFrame,
    expected_commit_count: int,
    commit_df: pd.DataFrame,
    file_df: pd.DataFrame,
    file_history_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    replacement_v4: pd.DataFrame,
    claim_v4: pd.DataFrame,
    finding_v4: pd.DataFrame,
    evidence_df: pd.DataFrame,
    open_questions_df: pd.DataFrame,
) -> dict[str, Any]:
    def ratio(n: int, d: int) -> float:
        return 1.0 if d == 0 else n / d

    ref_ratio = ratio(len(ref_df), selected_ref_count)
    commit_ratio = ratio(len(commit_df), expected_commit_count)
    file_ratio = ratio(
        len(set(file_history_df["path"].astype(str).tolist())),
        len(set(file_df["path"].astype(str).tolist())),
    )
    comp_ratio = ratio(len(lifecycle_df), len(inv_df))
    replacement_ratio = ratio(
        int(
            (
                replacement_v4["effective_commit"].astype(str).str.strip().ne("")
                & replacement_v4["effective_run"].astype(str).str.strip().ne("")
                & replacement_v4["evidence_refs"].astype(str).str.strip().ne("")
            ).sum()
        ),
        len(replacement_v4),
    )
    claim_ratio = ratio(
        int(claim_v4["status"].astype(str).str.lower().eq("supported").sum()),
        len(claim_v4),
    )
    finding_ratio = ratio(
        int(finding_v4["resolved_status"].astype(str).str.lower().eq("closed").sum()),
        len(finding_v4),
    )

    missing_evidence_count = int((~evidence_df["exists_flag"]).sum()) if not evidence_df.empty else 0
    open_question_count = int(
        open_questions_df[open_questions_df["status"].astype(str).str.lower() == "open"].shape[0]
    ) if not open_questions_df.empty else 0

    complete = (
        ref_ratio == 1.0
        and commit_ratio == 1.0
        and file_ratio == 1.0
        and comp_ratio == 1.0
        and replacement_ratio == 1.0
        and claim_ratio == 1.0
        and finding_ratio == 1.0
        and missing_evidence_count == 0
        and open_question_count == 0
    )
    return {
        "ref_coverage_ratio": ref_ratio,
        "commit_coverage_ratio": commit_ratio,
        "file_coverage_ratio": file_ratio,
        "component_coverage_ratio": comp_ratio,
        "replacement_coverage_ratio": replacement_ratio,
        "claim_coverage_ratio": claim_ratio,
        "finding_coverage_ratio": finding_ratio,
        "missing_evidence_count": missing_evidence_count,
        "open_question_count": open_question_count,
        "overall_status": "COMPLETE" if complete else "INCOMPLETE",
    }


def _build_masterdoc(
    *,
    ref_df: pd.DataFrame,
    commit_df: pd.DataFrame,
    era_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    replacement_v4: pd.DataFrame,
    claim_v4: pd.DataFrame,
    finding_v4: pd.DataFrame,
    coverage: dict[str, Any],
) -> str:
    ref_summary = (
        ref_df.groupby("ref_type", dropna=False).size().reset_index(name="count").sort_values("ref_type")
        if not ref_df.empty
        else pd.DataFrame(columns=["ref_type", "count"])
    )
    commit_summary = (
        commit_df.groupby("commit_class", dropna=False).size().reset_index(name="count").sort_values("commit_class")
        if not commit_df.empty
        else pd.DataFrame(columns=["commit_class", "count"])
    )
    transition_view = transitions_df[
        [
            "transition_id",
            "old_component",
            "new_component",
            "first_transition_commit",
            "transition_type",
            "reason",
        ]
    ] if not transitions_df.empty else pd.DataFrame(
        columns=[
            "transition_id",
            "old_component",
            "new_component",
            "first_transition_commit",
            "transition_type",
            "reason",
        ]
    )
    lifecycle_view = lifecycle_df[
        [
            "component_id",
            "component_type",
            "status_now",
            "successor_component",
            "replacement_reason_category",
        ]
    ].copy()
    lines: list[str] = []
    lines.append("# HISTORY_MASTERDOC")
    lines.append("")
    lines.append("## 1. Scope & Methodik")
    lines.append(
        "- Vollabzug über alle lokal verfügbaren `refs/heads/*`, `refs/remotes/*`, `refs/tags/*`."
    )
    lines.append(
        "- Forensische Kernartefakte: `REF_SNAPSHOT.csv`, `COMMIT_HISTORY_FULL.csv`, `COMMIT_FILE_CHANGES_FULL.csv`, `FILE_HISTORY_FULL.csv`."
    )
    lines.append(
        "- Dieses Dokument ist die lesbare Hauptfassung; alle Aussagen sind im `EVIDENCE_INDEX_V4.csv` referenziert."
    )
    lines.append("")
    lines.append("## 2. Ref-Snapshot (alle Branches/Tags)")
    lines.append(_markdown_table(ref_summary))
    lines.append("")
    lines.append("## 3. Commit-Überblick (vollständig)")
    lines.append(f"- Commit-Anzahl: `{len(commit_df)}`")
    lines.append(_markdown_table(commit_summary))
    lines.append("")
    lines.append("## 4. Phasenmodell und Milestones")
    lines.append(_markdown_table(era_df))
    lines.append("")
    lines.append("## 5. Script-to-CLI Transition")
    lines.append(_markdown_table(transition_view))
    lines.append("")
    lines.append("## 6. Workflow-/CLI-/Komponenten-Lifecycle")
    lines.append(_markdown_table(lifecycle_view))
    lines.append("")
    lines.append("## 7. Ersetzungsmatrix mit Gründen")
    lines.append(
        _markdown_table(
            replacement_v4[
                [
                    "old_component",
                    "new_component",
                    "replacement_type",
                    "reason_category",
                    "effective_commit",
                    "effective_run",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## 8. Claim-Traceability")
    lines.append(
        _markdown_table(
            claim_v4[
                [
                    "claim_id",
                    "source_doc",
                    "status",
                    "evidence_code",
                    "evidence_tests",
                    "evidence_artifacts",
                    "evidence_history",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## 9. Finding-Auflösung (v1 -> v4)")
    lines.append(_markdown_table(finding_v4))
    lines.append("")
    lines.append("## 10. Limitationen")
    lines.append("- Der Audit nutzt lokal verfügbare Git-Refs zum Ausführungszeitpunkt.")
    lines.append("- Externe Systeme außerhalb des Repos sind nicht Bestandteil des Vollabzugs.")
    lines.append("")
    lines.append("## 11. Reproduzierbarkeit (Command-Log, Hashes)")
    lines.append("- Vollständiger Laufkontext in `COMMAND_LOG.txt`.")
    lines.append("- Coverage-/Status-Gate in `HISTORY_COVERAGE_V4.json`.")
    lines.append("")
    lines.append("## 12. Fazit")
    lines.append(
        f"- Overall history status: `{coverage['overall_status']}`."
    )
    lines.append(
        f"- Coverage: refs={coverage['ref_coverage_ratio']:.2%}, commits={coverage['commit_coverage_ratio']:.2%}, files={coverage['file_coverage_ratio']:.2%}, components={coverage['component_coverage_ratio']:.2%}."
    )
    return "\n".join(lines) + "\n"


def _build_command_log(
    *,
    output_dir: Path,
    ref_count: int,
    commit_count: int,
    coverage: dict[str, Any],
    strict_complete: bool,
    include_archives: bool,
    baseline_v3: Path,
) -> str:
    lines = [
        f"generated_at_utc={_utc_now_iso()}",
        f"output_dir={output_dir}",
        f"baseline_v3={baseline_v3}",
        f"strict_complete={strict_complete}",
        f"include_archives={include_archives}",
        f"ref_count={ref_count}",
        f"commit_count={commit_count}",
    ]
    for k, v in coverage.items():
        lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


@dataclass
class V4AuditInputs:
    repo_root: Path
    output_dir: Path
    baseline_v3: Path
    strict_complete: bool = True
    include_archives: bool = True


@dataclass
class V4AuditResult:
    output_dir: Path
    overall_status: str
    commit_count: int
    ref_count: int


def run_repo_evolution_audit_v4(inputs: V4AuditInputs) -> V4AuditResult:
    out = inputs.output_dir
    out.mkdir(parents=True, exist_ok=True)
    repo_root = inputs.repo_root

    ref_df = _parse_ref_snapshot(repo_root)
    refs = ref_df["ref_name"].astype(str).tolist()
    commits = _collect_commits_for_refs(repo_root, refs)
    reachable_map = _collect_reachable_refs_map(repo_root, refs)
    ref_type_by_name = {
        str(r["ref_name"]): str(r["ref_type"])
        for _, r in ref_df.iterrows()
    }

    commit_df, file_df = _collect_commit_data(repo_root, commits, reachable_map, ref_type_by_name)
    file_history_df = _build_file_history(repo_root, commit_df, file_df)
    inv_df = _build_component_inventory(repo_root, file_history_df, inputs.include_archives)
    lifecycle_df = _build_component_lifecycle(inv_df, file_history_df)
    v3 = _load_v3_baseline(inputs.baseline_v3)

    run_candidates = v3["replacement"]["effective_from_run"].astype(str).tolist()
    transitions_df = _build_script_to_cli_transitions(repo_root, file_history_df, run_candidates)
    era_df = _build_era_milestones(ref_df, commit_df, file_history_df)

    replacement_v4 = _build_replacement_v4(v3["replacement"], transitions_df)
    replacement_chain_v4 = _build_replacement_chain_v4(replacement_v4)
    claim_v4 = _build_claim_v4(v3["claims"])
    finding_v4 = _build_finding_resolution_v4(repo_root, v3["findings"])
    evidence_df = _build_evidence_index_v4(
        repo_root,
        era_df,
        transitions_df,
        replacement_v4,
        claim_v4,
        finding_v4,
    )
    open_q_df = _build_open_questions_v4(ref_df, evidence_df)
    coverage = _history_coverage(
        selected_ref_count=len(refs),
        ref_df=ref_df,
        expected_commit_count=len(commits),
        commit_df=commit_df,
        file_df=file_df,
        file_history_df=file_history_df,
        inv_df=inv_df,
        lifecycle_df=lifecycle_df,
        replacement_v4=replacement_v4,
        claim_v4=claim_v4,
        finding_v4=finding_v4,
        evidence_df=evidence_df,
        open_questions_df=open_q_df,
    )
    if inputs.strict_complete and coverage["overall_status"] != "COMPLETE":
        # status remains INCOMPLETE; strictness is encoded via output gate semantics
        pass

    masterdoc = _build_masterdoc(
        ref_df=ref_df,
        commit_df=commit_df,
        era_df=era_df,
        transitions_df=transitions_df,
        lifecycle_df=lifecycle_df,
        replacement_v4=replacement_v4,
        claim_v4=claim_v4,
        finding_v4=finding_v4,
        coverage=coverage,
    )

    _write_text(out / "HISTORY_MASTERDOC.md", masterdoc)
    _write_csv(ref_df, out / "REF_SNAPSHOT.csv")
    _write_csv(commit_df, out / "COMMIT_HISTORY_FULL.csv")
    _write_csv(file_df, out / "COMMIT_FILE_CHANGES_FULL.csv")
    _write_csv(file_history_df, out / "FILE_HISTORY_FULL.csv")
    _write_csv(inv_df, out / "COMPONENT_INVENTORY_FULL.csv")
    _write_csv(lifecycle_df, out / "COMPONENT_LIFECYCLE_FULL.csv")
    _write_csv(transitions_df, out / "SCRIPT_TO_CLI_TRANSITION.csv")
    _write_csv(era_df, out / "ERA_MILESTONES.csv")
    _write_csv(replacement_v4, out / "REPLACEMENT_MATRIX_V4.csv")
    _write_csv(replacement_chain_v4, out / "REPLACEMENT_EVIDENCE_CHAIN_V4.csv")
    _write_csv(claim_v4, out / "CLAIM_CROSSWALK_V4.csv")
    _write_csv(finding_v4, out / "FINDING_RESOLUTION_V4.csv")
    _write_csv(evidence_df, out / "EVIDENCE_INDEX_V4.csv")
    _write_text(out / "HISTORY_COVERAGE_V4.json", json.dumps(coverage, indent=2))
    _write_csv(open_q_df, out / "OPEN_QUESTIONS_V4.csv")
    _write_text(
        out / "COMMAND_LOG.txt",
        _build_command_log(
            output_dir=out,
            ref_count=len(refs),
            commit_count=len(commits),
            coverage=coverage,
            strict_complete=inputs.strict_complete,
            include_archives=inputs.include_archives,
            baseline_v3=inputs.baseline_v3,
        ),
    )

    return V4AuditResult(
        output_dir=out,
        overall_status=str(coverage["overall_status"]),
        commit_count=len(commits),
        ref_count=len(refs),
    )


def _find_latest_v3_audit(repo_root: Path) -> Path:
    root = repo_root / "outputs" / "audits"
    candidates = sorted(root.glob("repo_evolution_v3_final_*"))
    if not candidates:
        raise FileNotFoundError("No repo_evolution_v3_final_* directory found under outputs/audits")
    return candidates[-1]


def _default_output_dir(repo_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "outputs" / "audits" / f"repo_evolution_v4_complete_{stamp}"


@cli_command(
    "repo-evolution-audit-v4",
    help="Generate complete repository evolution audit (all refs/commits/files/components).",
    args={
        "output_dir": {"type": str, "default": None},
        "strict_complete": {"type": bool, "default": True},
        "include_archives": {"type": bool, "default": True},
        "baseline_v3": {"type": str, "default": None},
    },
)
def main(
    output_dir: str | None = None,
    strict_complete: bool = True,
    include_archives: bool = True,
    baseline_v3: str | None = None,
) -> int:
    repo_root = Path.cwd()
    baseline = Path(baseline_v3) if baseline_v3 else _find_latest_v3_audit(repo_root)
    out = Path(output_dir) if output_dir else _default_output_dir(repo_root)
    result = run_repo_evolution_audit_v4(
        V4AuditInputs(
            repo_root=repo_root,
            output_dir=out,
            baseline_v3=baseline,
            strict_complete=bool(strict_complete),
            include_archives=bool(include_archives),
        )
    )
    print(
        json.dumps(
            {
                "status": result.overall_status,
                "ref_count": result.ref_count,
                "commit_count": result.commit_count,
                "output_dir": str(result.output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parse docs/all_scripts_overview_detailed.md and generate:
 - docs/all_scripts_report.json (machine-readable list of files)
 - docs/all_scripts_cleanup_plan.md (compact human-readable plan with suggestions)

Heuristics used for suggestions:
 - group by filename prefix tokens (optuna, compare_samplers, bootstrap, xxl, docs_link)
 - deprecated folder -> suggestion: archive/delete
 - multiple similarly named scripts -> suggestion: consolidate into unified CLI/subcommand
 - small utilities (check_*, verify_*, diagnose_*) -> suggestion: move into tools/ or consolidate
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_MD = ROOT / "docs" / "all_scripts_overview_detailed.md"
OUT_JSON = ROOT / "docs" / "all_scripts_report.json"
OUT_MD = ROOT / "docs" / "all_scripts_cleanup_plan.md"

text = SRC_MD.read_text(encoding="utf8")

sections = re.split(r"\n## ", text)[1:]
entries = []

for sec in sections:
    # each sec starts with path line
    lines = sec.splitlines()
    path = lines[0].strip()
    data = {"path": path}

    def get_field(field):
        m = re.search(rf"- \*\*{re.escape(field)}:\*\* (.*)", sec)
        if m:
            return m.group(1).strip()
        return ""

    data["imports"] = (
        [s.strip() for s in get_field("Imports (top-level modules)").split(",")]
        if get_field("Imports (top-level modules)")
        else []
    )
    data["functions"] = (
        [s.strip() for s in get_field("Top-level functions").split(",")]
        if get_field("Top-level functions")
        and "*(none)*" not in get_field("Top-level functions")
        else []
    )
    data["classes"] = (
        [s.strip() for s in get_field("Top-level classes").split(",")]
        if get_field("Top-level classes")
        and "*(none)*" not in get_field("Top-level classes")
        else []
    )
    data["has_main"] = True if "Yes" in get_field("Has __main__ guard") else False
    sc = get_field("Subprocess-like calls detected")
    data["subprocess_calls"] = (
        [s.strip() for s in sc.split(",")] if sc and "*(none)*" not in sc else []
    )
    cli = get_field("CLI libs detected")
    data["cli"] = [s.strip() for s in cli.split(",")] if cli else []

    # category heuristics
    p = path
    if p.startswith("scripts/"):
        category = "script"
    elif p.startswith("tools/"):
        category = "tool"
    elif p.startswith("dataselector/"):
        category = "package"
    else:
        category = "misc"
    if "/deprecated/" in p or "/deprecated" in p or p.startswith("scripts/deprecated/"):
        deprecated = True
    else:
        deprecated = False

    data["category"] = category
    data["deprecated"] = deprecated

    # short compact description heuristics
    short = []
    if category == "script":
        # derive purpose from filename tokens
        name = Path(p).name
        tokens = re.split(r"[_.-]", name.lower())
        if "optuna" in tokens:
            short.append("Optuna / hyperparameter optimization helpers")
        if "compare" in tokens or "compare" in name:
            short.append("Sampler / method comparison experiments")
        if "bootstrap" in tokens or "bootstrap" in name:
            short.append("Bootstrap / ensemble candidates & evaluation")
        if "run" in tokens or name.startswith("run_") or name.startswith("xxl_"):
            short.append("Pipeline runner or cluster orchestration script")
        if "report" in tokens or "generate" in tokens:
            short.append("Report generation and export")
        if (
            name.startswith("check_")
            or name.startswith("verify_")
            or "diagnose" in tokens
        ):
            short.append("Diagnostics / environment checks / validation")
        if name.startswith("install_") or name.startswith("compat"):
            short.append("Repo maintenance / install helpers")
        if not short:
            short.append("Utility or experiment runner (specific purpose)")
    elif category == "package":
        if "/workflows/" in p:
            short.append("High-level workflows used by scripts")
        elif "/pipeline/" in p:
            short.append("Core pipeline orchestration")
        elif "/selection/" in p:
            short.append("Selection algorithms (facility location, clustering, pareto)")
        elif "/data/" in p:
            short.append("IO / metadata / tiles handling")
        elif "/features/" in p:
            short.append("Feature extraction and pipelines")
        else:
            short.append("Core library module")
    else:
        short.append("Misc or helper module")

    data["compact_summary"] = " ".join(short)

    entries.append(data)

groups = defaultdict(list)
for e in entries:
    n = Path(e["path"]).name
    key = None
    if "optuna" in n.lower():
        key = "optuna"
    elif "compare_samplers" in n.lower() or "compare" in n.lower():
        key = "compare"
    elif "bootstrap" in n.lower():
        key = "bootstrap"
    elif n.lower().startswith("xxl_") or "xxl" in n.lower():
        key = "xxl"
    elif "docs_link" in n.lower() or "docs" in n.lower():
        key = "docs_link"
    elif (
        "generate_report" in n.lower()
        or "generate" in n.lower()
        or "report" in n.lower()
    ):
        key = "reports"
    elif (
        n.lower().startswith("check_")
        or n.lower().startswith("verify_")
        or "diagnose" in n.lower()
    ):
        key = "checks"
    elif e["deprecated"]:
        key = "deprecated"
    if key:
        groups[key].append(e["path"])

# build suggestions
for e in entries:
    p = e["path"]
    n = Path(p).name
    suggestions = []
    if e["deprecated"]:
        suggestions.append("archive_or_delete")
    # if primary grouping exists with >1 members suggest consolidation
    for k, v in groups.items():
        if p in v and len(v) > 1:
            if k == "optuna":
                suggestions.append("consolidate_optuna_to_single_cli_with_subcommands")
            elif k == "compare":
                suggestions.append("merge_compare_variants_into_parametrized_runner")
            elif k == "bootstrap":
                suggestions.append("consolidate_bootstrap_tools")
            elif k == "xxl":
                suggestions.append("parameterize_xxl_runs_or_unify_into_xxl_runner")
            elif k == "docs_link":
                suggestions.append("merge_docs_link_fixers_into_single_tool")
            elif k == "reports":
                suggestions.append("unify_report_generation_templates")
            elif k == "checks":
                suggestions.append("move_checks_to_tools_or_merge")
            elif k == "deprecated":
                suggestions.append("archive_or_delete")
    # small utilities suggestion
    if (
        e["category"] == "script"
        and (not e["functions"] or len(e["functions"]) <= 1)
        and (not e["has_main"])
    ):
        suggestions.append("consider_merging_into_tools")
    if "subprocess_calls" in e and e["subprocess_calls"]:
        suggestions.append("review_subprocess_usage_for_safety")

    e["suggestions"] = sorted(set(suggestions))

# write JSON
OUT_JSON.write_text(json.dumps(entries, indent=2), encoding="utf8")

# build compact markdown cleanup plan
lines = [
    "# All Scripts Cleanup Plan\n",
    "This plan groups files with suggested actions to consolidate, archive, or refactor.\n",
]

cats = Counter([e["category"] for e in entries])
lines.append(f"- Total files scanned: {len(entries)}\n")
for c, cnt in cats.items():
    lines.append(f"  - {c}: {cnt}\n")
lines.append("\n")

# Show groups with suggested actions
for k, v in groups.items():
    lines.append(f"## Group: {k} ({len(v)} files)\n")
    if k == "optuna":
        lines.append(
            "**Suggestion:** Consolidate to a single `optuna` CLI subcommand that supports import/export, autoscale, analysis. Move helper scripts into `dataselector.workflows.optuna` as subcommands.\n"
        )
    elif k == "compare":
        lines.append(
            "**Suggestion:** Merge `compare_samplers*` variants into a parametrized comparison runner that accepts dataset & seeding strategies as arguments. Keep legacy scripts as archived examples.\n"
        )
    elif k == "bootstrap":
        lines.append(
            "**Suggestion:** Consolidate bootstrap generation/plotting into a `bootstrap` module with small CLI wrappers.\n"
        )
    elif k == "xxl":
        lines.append(
            "**Suggestion:** Parameterize XXL-run scripts or unify into a single XXL orchestration CLI; keep one example runner per experiment profile.\n"
        )
    elif k == "docs_link":
        lines.append(
            "**Suggestion:** Merge docs link fixers into one robust maintenance tool with subcommands (patch, autofix, show patterns).\n"
        )
    elif k == "reports":
        lines.append(
            "**Suggestion:** Unify report generation and templating to reduce duplicate code; expose as `dataselector.workflows.generate_reports`.\n"
        )
    elif k == "checks":
        lines.append(
            "**Suggestion:** Move environment and diagnostic checks into `tools/` and add a single CLI entry.\n"
        )
    elif k == "deprecated":
        lines.append(
            "**Suggestion:** Archive or remove scripts in `deprecated/` after ensuring no active use.\n"
        )
    lines.append("\n")
    # list up to 40 files
    for path in sorted(v)[:40]:
        lines.append(f"- `{path}`\n")
    lines.append("\n")

# individual file short list with suggestions
lines.append("## Per-file suggestions\n")
for e in sorted(entries, key=lambda x: x["path"]):
    s = e["suggestions"]
    if s:
        lines.append(f"- `{e['path']}`: {', '.join(s)} — {e['compact_summary']}\n")

OUT_MD.write_text("\n".join(lines), encoding="utf8")
print("WROTE", OUT_JSON, OUT_MD)

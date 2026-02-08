#!/usr/bin/env python3
"""Generate an inventory of real Python `pass` statements.

This scanner is intentionally AST-based so comments/strings containing "pass"
are ignored. It emits JSON and Markdown artifacts for Phase 4E assessment.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_ROOTS = [
    "dataselector",
    "tests",
    "docs/reorganize_tests_plans/master_e2e_test.py",
    "check_image_assignment.py",
]


@dataclass
class PassEntry:
    file: str
    line: int
    col: int
    module_type: str
    parent_type: str
    except_type: str | None
    classification: str
    priority: str
    rationale: str
    code_line: str
    context_before: list[str]
    context_after: list[str]


def iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        if root.is_dir():
            yield from root.rglob("*.py")


def classify_module(path: Path) -> str:
    p = str(path).replace("\\", "/")
    if p.startswith("dataselector/workflows/"):
        return "workflow"
    if p.startswith("dataselector/tools/"):
        return "tool"
    if p.startswith("dataselector/"):
        return "runtime"
    if p.startswith("tests/archive/"):
        return "archive_test"
    if p.startswith("tests/"):
        return "test"
    if p.startswith("docs/reorganize_tests_plans/"):
        return "legacy_helper"
    return "helper_or_other"


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _except_type_name(except_node: ast.ExceptHandler) -> str | None:
    if except_node.type is None:
        return "Exception"
    t = except_node.type
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Tuple):
        names = []
        for elt in t.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(elt.attr)
        if names:
            return ",".join(names)
    return ast.dump(t, include_attributes=False)


def classify_pass(
    file_path: Path,
    module_type: str,
    parent_type: str,
    except_type: str | None,
    line: int,
) -> tuple[str, str, str]:
    p = str(file_path).replace("\\", "/")

    if p == "dataselector/compat.py" and line == 22:
        return (
            "runtime_error_swallow_risky",
            "P0",
            "Compatibility gate silently swallows runtime path and can misreport env safety.",
        )

    if module_type in {"archive_test", "legacy_helper"}:
        return (
            "legacy_or_archive_only",
            "P2",
            "Archived or historical helper/test path.",
        )

    if module_type == "test":
        return (
            "intentional_test_stub",
            "P2",
            "Test shim/stub/expected exception placeholder.",
        )

    if parent_type == "ExceptHandler":
        if except_type and (
            "ImportError" in except_type or "ModuleNotFoundError" in except_type
        ):
            return (
                "optional_dependency_guard",
                "P2",
                "Missing optional dependency is tolerated by design.",
            )
        if p == "dataselector/tools/audit.py":
            return (
                "runtime_error_swallow_risky",
                "P1",
                "Audit coordinate extraction can hide malformed geometry cases.",
            )
        if p == "dataselector/workflows/bootstrap.py":
            return (
                "runtime_error_swallow_risky",
                "P1",
                "Bootstrap clustering fallback may hide upstream clustering regressions.",
            )
        return (
            "best_effort_cleanup",
            "P2",
            "Best-effort cleanup/telemetry fallback; core flow continues.",
        )

    return (
        "best_effort_cleanup",
        "P2",
        "Intentional no-op control path.",
    )


def collect_entries(file_path: Path) -> tuple[list[PassEntry], str | None]:
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError as exc:
        return [], f"{file_path}: {exc}"

    parents = _build_parent_map(tree)
    entries: list[PassEntry] = []
    module_type = classify_module(file_path)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Pass):
            continue
        parent = parents.get(id(node))
        parent_type = type(parent).__name__ if parent is not None else "Unknown"
        except_type = (
            _except_type_name(parent) if isinstance(parent, ast.ExceptHandler) else None
        )

        line_idx = node.lineno - 1
        code_line = lines[line_idx].rstrip() if 0 <= line_idx < len(lines) else ""

        before_start = max(0, line_idx - 2)
        before = [ln.rstrip() for ln in lines[before_start:line_idx]]
        after = [ln.rstrip() for ln in lines[line_idx + 1 : line_idx + 3]]

        classification, priority, rationale = classify_pass(
            file_path=file_path,
            module_type=module_type,
            parent_type=parent_type,
            except_type=except_type,
            line=node.lineno,
        )

        entries.append(
            PassEntry(
                file=str(file_path).replace("\\", "/"),
                line=node.lineno,
                col=node.col_offset,
                module_type=module_type,
                parent_type=parent_type,
                except_type=except_type,
                classification=classification,
                priority=priority,
                rationale=rationale,
                code_line=code_line,
                context_before=before,
                context_after=after,
            )
        )
    return entries, None


def to_markdown(entries: list[PassEntry], parse_errors: list[str]) -> str:
    by_module: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    prio_counts: dict[str, int] = {}
    for e in entries:
        by_module[e.module_type] = by_module.get(e.module_type, 0) + 1
        class_counts[e.classification] = class_counts.get(e.classification, 0) + 1
        prio_counts[e.priority] = prio_counts.get(e.priority, 0) + 1

    out: list[str] = []
    out.append("# Phase 4E Pass Inventory")
    out.append("")
    out.append(f"- Total `pass` statements: {len(entries)}")
    out.append("")
    out.append("## Counts by Module Type")
    out.append("")
    for key in sorted(by_module):
        out.append(f"- `{key}`: {by_module[key]}")
    out.append("")

    out.append("## Classification Counts")
    out.append("")
    for key in sorted(class_counts):
        out.append(f"- `{key}`: {class_counts[key]}")
    out.append("")
    for key in sorted(prio_counts):
        out.append(f"- `{key}`: {prio_counts[key]}")
    out.append("")

    p0_p1 = [e for e in entries if e.priority in {"P0", "P1"}]
    if p0_p1:
        out.append("## Priority Findings (P0/P1)")
        out.append("")
        out.append("| Priority | File | Line | Classification | Rationale |")
        out.append("|---|---|---:|---|---|")
        for e in p0_p1:
            rationale = e.rationale.replace("|", "\\|")
            out.append(
                f"| `{e.priority}` | `{e.file}` | {e.line} | "
                f"`{e.classification}` | {rationale} |"
            )
        out.append("")

    if parse_errors:
        out.append("## Parse Errors")
        out.append("")
        for err in parse_errors:
            out.append(f"- {err}")
        out.append("")

    out.append("## Inventory")
    out.append("")
    out.append("| File | Line | Module | Parent | Except | Class | Prio | Code |")
    out.append("|---|---:|---|---|---|---|---|---|")
    for e in entries:
        safe_code = e.code_line.replace("|", "\\|").strip()
        out.append(
            f"| `{e.file}` | {e.line} | `{e.module_type}` | `{e.parent_type}` | "
            f"`{e.except_type or '-'}` | `{e.classification}` | `{e.priority}` | "
            f"`{safe_code}` |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pass-statement inventory")
    parser.add_argument(
        "--roots", nargs="*", default=DEFAULT_ROOTS, help="Root files/dirs to scan"
    )
    parser.add_argument("--json-out", required=True, help="Path to JSON output")
    parser.add_argument("--md-out", required=True, help="Path to Markdown output")
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    files = sorted(set(iter_python_files(roots)))
    entries: list[PassEntry] = []
    parse_errors: list[str] = []

    for file_path in files:
        found, err = collect_entries(file_path)
        entries.extend(found)
        if err:
            parse_errors.append(err)

    entries.sort(key=lambda e: (e.file, e.line, e.col))

    json_payload = {
        "roots": [str(r) for r in roots],
        "total": len(entries),
        "parse_errors": parse_errors,
        "entries": [
            {
                "file": e.file,
                "line": e.line,
                "col": e.col,
                "module_type": e.module_type,
                "parent_type": e.parent_type,
                "except_type": e.except_type,
                "classification": e.classification,
                "priority": e.priority,
                "rationale": e.rationale,
                "code_line": e.code_line,
                "context_before": e.context_before,
                "context_after": e.context_after,
            }
            for e in entries
        ],
    }

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(entries, parse_errors), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

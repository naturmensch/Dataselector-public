#!/usr/bin/env python3
"""Phase 4 feature-gap scanner with ownership governance checks."""

from __future__ import annotations

import argparse
import os
import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "artifacts" / "phase4b" / "feature_gap_scan.json"
DEFAULT_MD = ROOT / "docs" / "status" / f"phase4b_feature_gap_matrix_{date.today()}.md"
REGISTRY_PATH = ROOT / "docs" / "status" / "feature_ownership_registry.yaml"

ACTIVE_CALLSITE_PREFIXES = (
    "dataselector/workflows",
    "dataselector/tools",
    ".github/workflows",
)


@dataclass
class FeatureRow:
    feature: str
    owner_module: str
    registry_owner_module: str
    registry_type: str
    in_cli_registry: bool
    help_ok: bool
    callsite_count: int
    callsites: list[str]
    status: str
    proposed_action: str
    violations: list[str]
    notes: str = ""


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def parse_cli_commands(help_text: str) -> list[str]:
    commands: set[str] = set()

    usage_match = re.search(r"\{([a-z0-9,\-]+)\}\s*\.\.\.", help_text)
    if usage_match:
        for token in usage_match.group(1).split(","):
            token = token.strip()
            if re.fullmatch(r"[a-z0-9][a-z0-9-]*", token):
                commands.add(token)
        return sorted(commands)

    in_positional = False
    for line in help_text.splitlines():
        raw = line.rstrip()
        lower = raw.lower().strip()
        if lower == "positional arguments:":
            in_positional = True
            continue
        if not in_positional:
            continue
        if lower == "options:":
            break
        if not raw.startswith(" "):
            continue
        stripped = raw.strip()
        if not stripped or stripped.startswith("{"):
            continue
        token = stripped.split()[0]
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", token):
            commands.add(token)

    return sorted(commands)


def command_help_ok(cmd_name: str, python_bin: str) -> bool:
    proc = run([python_bin, "-m", "dataselector", cmd_name, "--help"])
    return proc.returncode == 0


def iter_source_files() -> Iterable[Path]:
    include_suffixes = {".py", ".md", ".yml", ".yaml", ".sh"}
    skip_prefixes = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "archive",
        "archive_local",
        "build",
        "dist",
        "venv",
        ".venv",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        rel = path.relative_to(ROOT)
        first = rel.parts[0] if rel.parts else ""
        if first in skip_prefixes:
            continue
        yield path


def find_callsites(feature: str) -> list[str]:
    patterns = [
        re.compile(rf"\bpython\s+-m\s+dataselector\s+{re.escape(feature)}\b"),
        re.compile(rf"\bdataselector\s+{re.escape(feature)}\b"),
        re.compile(rf"['\"]{re.escape(feature)}['\"]"),
    ]
    hits: set[str] = set()
    for path in iter_source_files():
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(p.search(text) for p in patterns):
            hits.add(str(path.relative_to(ROOT)))
    return sorted(hits)


def _extract_commands_from_text(text: str) -> set[str]:
    patterns = [
        re.compile(r"\bpython\s+-m\s+dataselector\s+([a-z0-9][a-z0-9-]*)\b"),
        re.compile(
            r"['\"]-m['\"]\s*,\s*['\"]dataselector['\"]\s*,\s*['\"]([a-z0-9][a-z0-9-]*)['\"]"
        ),
    ]
    commands: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            commands.add(match.group(1))
    return commands


def find_active_command_callsites() -> dict[str, list[str]]:
    hits: dict[str, set[str]] = {}
    for path in iter_source_files():
        rel = str(path.relative_to(ROOT))
        if not rel.startswith(ACTIVE_CALLSITE_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for cmd in _extract_commands_from_text(text):
            hits.setdefault(cmd, set()).add(rel)
    return {cmd: sorted(paths) for cmd, paths in hits.items()}


def collect_cli_owners() -> dict[str, list[str]]:
    owners: dict[str, set[str]] = {}
    for path in ROOT.joinpath("dataselector").rglob("*.py"):
        if path.name == "cli_decorators.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "cli_command"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                owners.setdefault(first.value, set()).add(str(path.relative_to(ROOT)))
    return {cmd: sorted(paths) for cmd, paths in owners.items()}


def load_ownership_registry() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    commands = data.get("commands", {})
    if not isinstance(commands, dict):
        raise ValueError(f"Invalid ownership registry format in {REGISTRY_PATH}")
    return commands


def _alias_behavior_ok(entry: dict, owner_module: str) -> tuple[bool, str]:
    expected = str(entry.get("expected_behavior", "")).strip()
    if expected != "forward":
        return True, ""

    owner_path = ROOT / owner_module
    if not owner_path.exists():
        return False, f"alias owner module not found: {owner_module}"

    text = owner_path.read_text(encoding="utf-8", errors="ignore")
    target = str(entry.get("forward_target", "")).strip()

    if target and target not in text:
        return False, f"alias forward target '{target}' not referenced in owner module"

    if not re.search(r"return\s+.*\.main\(", text):
        return False, "alias wrapper does not forward via return <target>.main(...)"

    return True, ""


def classify(
    in_cli: bool,
    help_ok: bool,
    has_active_callsites: bool,
    has_any_callsites: bool,
) -> tuple[str, str]:
    if in_cli and help_ok:
        return ("implemented_and_valid", "No action.")
    if in_cli and not help_ok:
        return ("implemented_but_misaligned", "Fix command help/argument parsing.")
    if (not in_cli) and has_active_callsites:
        return (
            "missing_required",
            "Command is referenced in active workflow paths but not registered in CLI.",
        )
    if (not in_cli) and has_any_callsites:
        return (
            "intentionally_deferred",
            "Referenced only in non-active paths (docs/tests/archive); verify if this is intended.",
        )
    return (
        "obsolete_candidate_for_removal",
        "No active references and not registered in CLI.",
    )


def build_rows(cli_commands: list[str], python_bin: str) -> list[FeatureRow]:
    owner_map = collect_cli_owners()
    registry = load_ownership_registry()
    active_callsites = find_active_command_callsites()

    observed = sorted(
        set(cli_commands)
        | set(owner_map.keys())
        | set(registry.keys())
        | set(active_callsites.keys())
    )

    rows: list[FeatureRow] = []
    for feature in observed:
        in_cli = feature in cli_commands
        help_ok = command_help_ok(feature, python_bin) if in_cli else False

        callsites = find_callsites(feature)
        active_hits = active_callsites.get(feature, [])
        has_active = len(active_hits) > 0

        status, action = classify(
            in_cli=in_cli,
            help_ok=help_ok,
            has_active_callsites=has_active,
            has_any_callsites=len(callsites) > 0,
        )

        owners = owner_map.get(feature, [])
        owner_module = owners[0] if owners else ""
        registry_entry = registry.get(feature, {}) if isinstance(registry, dict) else {}
        registry_owner = str(registry_entry.get("canonical_owner_module", ""))
        registry_type = str(registry_entry.get("type", ""))

        violations: list[str] = []
        notes: list[str] = []

        if in_cli and feature not in registry:
            violations.append("missing ownership registry entry for CLI command")

        if feature in registry and not in_cli and has_active:
            violations.append("command in ownership registry is not registered in CLI")

        if in_cli and len(owners) == 0:
            violations.append("no canonical owner module found for CLI command")

        if len(owners) > 1:
            violations.append(
                "multiple canonical owners found: " + ", ".join(sorted(owners))
            )

        if registry_owner and len(owners) == 1 and owners[0] != registry_owner:
            violations.append(
                f"registry owner mismatch (registry={registry_owner}, detected={owners[0]})"
            )

        if has_active and feature not in cli_commands and feature not in registry:
            violations.append("unknown command referenced in active workflow paths")
            notes.append("active callsites: " + ", ".join(active_hits[:5]))

        if registry_type == "alias":
            ok, msg = _alias_behavior_ok(registry_entry, registry_owner or owner_module)
            if not ok:
                violations.append(msg)

        if violations and status == "implemented_and_valid":
            status = "implemented_but_misaligned"
            action = "Resolve ownership/contract violations listed in this row."

        rows.append(
            FeatureRow(
                feature=feature,
                owner_module=owner_module or registry_owner or "dataselector/cli.py",
                registry_owner_module=registry_owner,
                registry_type=registry_type,
                in_cli_registry=in_cli,
                help_ok=help_ok,
                callsite_count=len(callsites),
                callsites=callsites[:10],
                status=status,
                proposed_action=action,
                violations=violations,
                notes="; ".join(notes),
            )
        )

    return rows


def render_markdown(rows: list[FeatureRow], sha: str, python_bin: str) -> str:
    lines = [
        f"# Phase 4 Feature Gap Matrix ({date.today()})",
        "",
        "## Baseline",
        f"- Commit: `{sha}`",
        f"- Python: `{python_bin}`",
        f"- Ownership registry: `{REGISTRY_PATH.relative_to(ROOT)}`",
        "",
        "## Status Classes",
        "- `implemented_and_valid`",
        "- `implemented_but_misaligned`",
        "- `missing_required`",
        "- `intentionally_deferred`",
        "- `obsolete_candidate_for_removal`",
        "",
        "## Matrix",
        "",
        "| Feature | Owner | Registry Owner | Type | CLI | Help | Callsites | Status | Violations | Action |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | `{owner}` | `{registry_owner}` | `{reg_type}` | {cli} | {help_ok} | {calls} | `{status}` | {violations} | {action} |".format(
                feature=row.feature,
                owner=row.owner_module,
                registry_owner=row.registry_owner_module or "-",
                reg_type=row.registry_type or "-",
                cli="yes" if row.in_cli_registry else "no",
                help_ok="yes" if row.help_ok else "no",
                calls=row.callsite_count,
                status=row.status,
                violations=("; ".join(row.violations) if row.violations else "-"),
                action=row.proposed_action,
            )
        )

    lines.append("")
    lines.append("## Callsite Samples")
    for row in rows:
        if not row.callsites:
            continue
        lines.append(f"### {row.feature}")
        for hit in row.callsites:
            lines.append(f"- `{hit}`")
        if row.notes:
            lines.append(f"- note: {row.notes}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate feature gap scan JSON + markdown matrix."
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("DATASELECTOR_PYTHON_BIN", "python"),
        help="Python executable used for command help checks",
    )
    parser.add_argument(
        "--json-out",
        default=str(DEFAULT_JSON),
        help="Output JSON path",
    )
    parser.add_argument(
        "--md-out",
        default=str(DEFAULT_MD),
        help="Output markdown path",
    )
    args = parser.parse_args()

    help_proc = run([args.python_bin, "-m", "dataselector", "--help"])
    if help_proc.returncode != 0:
        raise SystemExit(
            f"Failed to read CLI registry via --help (rc={help_proc.returncode}):\n{help_proc.stderr}"
        )

    cli_commands = parse_cli_commands(help_proc.stdout)
    sha = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()

    rows = build_rows(cli_commands=cli_commands, python_bin=args.python_bin)
    rows_dict = [asdict(r) for r in rows]

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(
        json.dumps(rows_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_out.write_text(
        render_markdown(rows=rows, sha=sha, python_bin=args.python_bin),
        encoding="utf-8",
    )

    print(f"Wrote JSON: {json_out}")
    print(f"Wrote Markdown: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

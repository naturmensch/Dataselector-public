#!/usr/bin/env python3
"""Check documentation drift against current dataselector CLI commands.

The checker classifies docs as authoritative/historical/generated via
docs/status/docs_registry.yaml and can fail on blocking findings only for
authoritative docs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = ROOT / "artifacts" / "docs_cli_drift_report.json"
DEFAULT_MD_OUT = ROOT / "docs" / "status" / f"docs_cli_drift_report_{date.today()}.md"
DEFAULT_REGISTRY = ROOT / "docs" / "status" / "docs_registry.yaml"

PY_M_DATSELECTOR_RE = re.compile(
    r"\bpython(?:3)?\s+-m\s+dataselector\s+([a-z0-9][a-z0-9-]*)\b", re.IGNORECASE
)
# Only match standalone shell-like command lines to avoid prose false positives.
BARE_CMD_RE = re.compile(
    r"(?m)^\s*(?:[$]\s*)?dataselector\s+([a-z0-9][a-z0-9-]*)\b",
    re.IGNORECASE,
)
SCRIPT_ERA_RE = re.compile(
    r"(?:(?:^|[\s`])(?:\./)?scripts/[^\s`]+|(?:^|[\s`])python(?:3)?\s+scripts/[^\s`]+)",
    re.IGNORECASE,
)
LEGACY_TOOLS_RE = re.compile(
    r"\bpython(?:3)?\s+-m\s+dataselector\s+tools\b", re.IGNORECASE
)


@dataclass
class DriftRow:
    file: str
    doc_class: str
    owner: str
    referenced_commands: list[str]
    unknown_commands: list[str]
    script_era_calls: list[str]
    legacy_tools_syntax: bool
    blocking: bool


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def parse_cli_commands(help_text: str) -> set[str]:
    m = re.search(r"\{([^}]+)\}\s*\.\.\.", help_text)
    if not m:
        return set()
    commands = set()
    for token in m.group(1).split(","):
        token = token.strip()
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", token):
            commands.add(token)
    return commands


def load_registry(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry format in {path}")
    return data


def _rule_matches(rel: str, rule: dict) -> bool:
    if "path" in rule and rel == str(rule["path"]):
        return True
    if "prefix" in rule and rel.startswith(str(rule["prefix"])):
        return True
    if "glob" in rule and Path(rel).match(str(rule["glob"])):
        return True
    return False


def classify_doc(rel: str, registry: dict) -> tuple[str, str]:
    for doc_class in ("authoritative", "historical", "generated"):
        for rule in registry.get(doc_class, []):
            if isinstance(rule, str):
                rule = {"path": rule}
            if not isinstance(rule, dict):
                continue
            if _rule_matches(rel, rule):
                return doc_class, str(rule.get("owner", "unknown"))

    # Conservative default: treat unknown docs as historical so they are visible
    # in report but do not block authoritative checks.
    return "historical", "unassigned"


def iter_docs() -> list[Path]:
    docs: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if rel.startswith((".git/", ".venv/", "venv/", "__pycache__/")):
            continue
        docs.append(path)
    return sorted(docs)


def extract_references(text: str) -> tuple[list[str], list[str], bool]:
    cmd_refs = {m.group(1).lower() for m in PY_M_DATSELECTOR_RE.finditer(text)}
    cmd_refs.update(m.group(1).lower() for m in BARE_CMD_RE.finditer(text))
    script_refs = sorted({m.group(0).strip() for m in SCRIPT_ERA_RE.finditer(text)})
    legacy_tools = bool(LEGACY_TOOLS_RE.search(text))
    return sorted(cmd_refs), script_refs, legacy_tools


def render_markdown(rows: list[DriftRow], commands: set[str]) -> str:
    total = len(rows)
    blocking = [r for r in rows if r.blocking]
    with_unknown = [r for r in rows if r.unknown_commands]
    with_script = [r for r in rows if r.script_era_calls]
    with_tools = [r for r in rows if r.legacy_tools_syntax]

    lines = [
        "# Docs CLI Drift Report",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- CLI commands in registry: {len(commands)}",
        f"- Docs scanned: {total}",
        f"- Blocking findings (authoritative only): {len(blocking)}",
        f"- Files with unknown CLI commands: {len(with_unknown)}",
        f"- Files with script-era calls: {len(with_script)}",
        f"- Files with legacy `dataselector tools` syntax: {len(with_tools)}",
        "",
        "## Blocking Findings",
        "",
    ]

    if not blocking:
        lines.append("No blocking findings in authoritative docs.")
    else:
        for row in blocking:
            lines.append(f"- `{row.file}`")
            if row.unknown_commands:
                lines.append(
                    f"  - unknown commands: `{', '.join(row.unknown_commands)}`"
                )
            if row.script_era_calls:
                lines.append(
                    f"  - script-era calls: `{', '.join(row.script_era_calls[:3])}`"
                )
            if row.legacy_tools_syntax:
                lines.append("  - legacy `python -m dataselector tools ...` syntax")

    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| File | Class | Unknown CLI | Script-era | Legacy tools | Blocking |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.file}` | `{row.doc_class}` | {len(row.unknown_commands)} | "
            f"{len(row.script_era_calls)} | {'yes' if row.legacy_tools_syntax else 'no'} | "
            f"{'yes' if row.blocking else 'no'} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check docs for CLI drift and script-era commands."
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python binary used for `-m dataselector --help`.",
    )
    parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY), help="Path to docs registry yaml."
    )
    parser.add_argument(
        "--json-out", default=str(DEFAULT_JSON_OUT), help="Output JSON report path."
    )
    parser.add_argument(
        "--md-out", default=str(DEFAULT_MD_OUT), help="Output Markdown report path."
    )
    parser.add_argument(
        "--strict-authoritative",
        action="store_true",
        help="Exit non-zero if authoritative docs contain blocking findings.",
    )
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    help_proc = run([args.python_bin, "-m", "dataselector", "--help"])
    if help_proc.returncode != 0:
        raise RuntimeError(f"Failed to load dataselector help:\n{help_proc.stderr}")
    commands = parse_cli_commands(help_proc.stdout)
    if not commands:
        raise RuntimeError(
            "Could not parse CLI commands from `python -m dataselector --help`."
        )

    rows: list[DriftRow] = []
    for path in iter_docs():
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8", errors="ignore")
        refs, script_calls, legacy_tools = extract_references(text)
        unknown = sorted([cmd for cmd in refs if cmd not in commands])
        doc_class, owner = classify_doc(rel, registry)
        blocking = bool(
            doc_class == "authoritative" and (unknown or script_calls or legacy_tools)
        )
        rows.append(
            DriftRow(
                file=rel,
                doc_class=doc_class,
                owner=owner,
                referenced_commands=refs,
                unknown_commands=unknown,
                script_era_calls=script_calls,
                legacy_tools_syntax=legacy_tools,
                blocking=blocking,
            )
        )

    rows_sorted = sorted(rows, key=lambda r: (not r.blocking, r.doc_class, r.file))
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps([asdict(r) for r in rows_sorted], indent=2), encoding="utf-8"
    )
    md_out.write_text(render_markdown(rows_sorted, commands), encoding="utf-8")

    blocking_count = sum(1 for row in rows_sorted if row.blocking)
    print(f"Scanned {len(rows_sorted)} docs. Blocking findings: {blocking_count}.")
    print(f"JSON report: {json_out}")
    print(f"Markdown report: {md_out}")

    if args.strict_authoritative and blocking_count:
        print("Blocking findings in authoritative docs detected.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

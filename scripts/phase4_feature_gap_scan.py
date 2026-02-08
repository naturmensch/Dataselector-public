#!/usr/bin/env python3
"""Phase 4B feature gap scanner.

Builds a repository-level feature readiness matrix to find:
- implemented commands
- contract mismatches
- missing but referenced features
- intentionally deferred items
- obsolete references
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "artifacts" / "phase4b" / "feature_gap_scan.json"
DEFAULT_MD = ROOT / "docs" / "status" / f"phase4b_feature_gap_matrix_{date.today()}.md"


@dataclass
class FeatureRow:
    feature: str
    owner_module: str
    in_cli_registry: bool
    help_ok: bool
    callsite_count: int
    callsites: list[str]
    status: str
    proposed_action: str
    notes: str = ""


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)


def parse_cli_commands(help_text: str) -> list[str]:
    commands: set[str] = set()

    # Argparse main usage style:
    # usage: dataselector [-h] {cmd-a,cmd-b,...} ...
    usage_match = re.search(r"\{([a-z0-9,\-]+)\}\s*\.\.\.", help_text)
    if usage_match:
        for token in usage_match.group(1).split(","):
            token = token.strip()
            if re.fullmatch(r"[a-z0-9][a-z0-9-]*", token):
                commands.add(token)
        return sorted(commands)

    # Fallback for alternate help layout where commands are listed line-by-line.
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


def detect_contract_mismatch(feature: str) -> str:
    if feature == "bootstrap-pareto":
        boot = ROOT / "dataselector" / "workflows" / "bootstrap.py"
        text = boot.read_text(encoding="utf-8", errors="ignore")
        if "Ensemble mode is not implemented yet" in text:
            return "ensemble_uq_not_implemented"
    if feature == "adaptive-auto":
        cli = ROOT / "dataselector" / "cli.py"
        text = cli.read_text(encoding="utf-8", errors="ignore")
        if "adaptive_auto" not in text:
            return "command_not_registered"
    return ""


def classify(
    feature: str, in_cli: bool, help_ok: bool, callsites: list[str]
) -> tuple[str, str]:
    mismatch = detect_contract_mismatch(feature)
    has_callsites = len(callsites) > 0

    if mismatch == "ensemble_uq_not_implemented":
        return (
            "implemented_but_misaligned",
            "Implement ensemble path in bootstrap-pareto using existing uncertainty_quantification module.",
        )
    if mismatch == "command_not_registered":
        return (
            "missing_required",
            "Implement and register adaptive-auto as thin orchestrator over autoscale + adaptive-pipeline.",
        )
    if in_cli and help_ok:
        return ("implemented_and_valid", "No action.")
    if in_cli and not help_ok:
        return ("implemented_but_misaligned", "Fix command help/arg parsing contract.")
    if (not in_cli) and has_callsites:
        return (
            "missing_required",
            "Feature referenced in active paths but absent from CLI registry; implement or remove active reference.",
        )
    if (not in_cli) and (not has_callsites):
        return (
            "obsolete_candidate_for_removal",
            "Not registered and not referenced in active paths; remove dead references if any appear.",
        )
    return ("intentionally_deferred", "Document defer reason and ownership.")


def build_rows(cli_commands: list[str], python_bin: str) -> list[FeatureRow]:
    owners = {
        "autoscale": "dataselector/workflows/autoscale.py",
        "optuna-optimize": "dataselector/workflows/optuna_optimize.py",
        "bootstrap-pareto": "dataselector/workflows/bootstrap.py",
        "bootstrap-final": "dataselector/workflows/bootstrap.py",
        "compare-samplers": "dataselector/workflows/compare_samplers.py",
        "thesis-pipeline": "dataselector/workflows/thesis_pipeline.py",
        "thesis-sampler-suite": "dataselector/workflows/thesis_sampler_suite.py",
        "sampler-suite": "dataselector/workflows/sampler_suite.py",
        "xxl": "dataselector/workflows/xxl.py",
        "adaptive-auto": "dataselector/workflows/adaptive_auto.py",
    }

    priority_features = list(owners.keys())
    observed = sorted(set(priority_features + cli_commands))

    rows: list[FeatureRow] = []
    for feature in observed:
        in_cli = feature in cli_commands
        help_ok = command_help_ok(feature, python_bin) if in_cli else False
        callsites = find_callsites(feature)
        status, action = classify(feature, in_cli, help_ok, callsites)
        rows.append(
            FeatureRow(
                feature=feature,
                owner_module=owners.get(feature, "dataselector/cli.py"),
                in_cli_registry=in_cli,
                help_ok=help_ok,
                callsite_count=len(callsites),
                callsites=callsites[:10],
                status=status,
                proposed_action=action,
            )
        )
    return rows


def render_markdown(rows: list[FeatureRow], sha: str, python_bin: str) -> str:
    lines = [
        f"# Phase 4B Feature Gap Matrix ({date.today()})",
        "",
        "## Baseline",
        f"- Commit: `{sha}`",
        f"- Python: `{python_bin}`",
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
        "| Feature | Owner | CLI | Help | Callsites | Status | Action |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | `{owner}` | {cli} | {help_ok} | {calls} | `{status}` | {action} |".format(
                feature=row.feature,
                owner=row.owner_module,
                cli="yes" if row.in_cli_registry else "no",
                help_ok="yes" if row.help_ok else "no",
                calls=row.callsite_count,
                status=row.status,
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
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase 4B feature gap scan JSON + markdown matrix."
    )
    parser.add_argument(
        "--python-bin",
        default="/opt/miniconda3/envs/dataselector/bin/python",
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

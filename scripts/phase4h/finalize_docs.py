#!/usr/bin/env python3
"""Finalize Phase4H docs by aligning evidence links and writing a consistency summary."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def newest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[-1]


def update_howto(howto: Path, latest_distance_summary: Path | None) -> bool:
    text = howto.read_text(encoding="utf-8")
    original = text

    if latest_distance_summary is not None:
        repl = (
            f"docs/06_REFERENCE/thesis_decision_evidence/{latest_distance_summary.name}"
        )
        text = re.sub(
            r"docs/06_REFERENCE/thesis_decision_evidence/min_distance_policy_summary_[0-9TZ]+\.md",
            repl,
            text,
        )

    if text != original:
        howto.write_text(text, encoding="utf-8")
        return True
    return False


def update_closeout(
    closeout: Path,
    *,
    n_samples: int | None,
    min_distance: float | None,
    distance_decision: Path | None,
    nsamples_decision: Path | None,
    policy24_evidence: Path | None,
) -> bool:
    text = closeout.read_text(encoding="utf-8")
    original = text

    marker = "## Scientific Final Freeze (Automation)"
    block = "\n".join(
        [
            marker,
            f"- Frozen at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
            f"- `selection.n_samples`: `{n_samples}`",
            f"- `selection.min_distance_km`: `{min_distance}`",
            "- Policy/reference split:",
            "  - geometric reference: `45.0`",
            f"  - operational policy: `{min_distance}`",
            "- Decision artifacts:",
            (
                f"  - `{distance_decision}`"
                if distance_decision
                else "  - missing distance decision artifact"
            ),
            (
                f"  - `{nsamples_decision}`"
                if nsamples_decision
                else "  - missing n_samples decision artifact"
            ),
            (
                f"- Latest policy-24 go-live evidence: `{policy24_evidence}`"
                if policy24_evidence
                else "- Latest policy-24 go-live evidence: not generated yet"
            ),
            "",
        ]
    )

    if marker in text:
        text = re.sub(
            r"## Scientific Final Freeze \(Automation\)\n(?:.*\n)*?(?=\n## |\Z)",
            block,
            text,
            flags=re.MULTILINE,
        )
    else:
        text = text.rstrip() + "\n\n" + block

    if text != original:
        closeout.write_text(text, encoding="utf-8")
        return True
    return False


def collect_missing_doc_refs(paths: list[Path], root: Path) -> list[tuple[Path, str]]:
    missing: list[tuple[Path, str]] = []
    pattern = re.compile(
        r"(docs/06_REFERENCE/thesis_decision_evidence/[A-Za-z0-9_./-]+\.md)"
    )
    for p in paths:
        text = p.read_text(encoding="utf-8")
        for rel in pattern.findall(text):
            target = root / rel
            if not target.exists():
                missing.append((p, rel))
    return missing


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    reports = root / "docs/06_REFERENCE/thesis_decision_evidence"
    docs = root / "docs"
    status = docs / "status"
    howto = docs / "03_USER_GUIDES" / "THESIS_PIPELINE_HOWTO.md"
    closeout = status / "phase4h_masterarbeit_closeout_2026-02-09.md"
    config = root / "config" / "pipeline_config.yaml"

    latest_distance_summary = newest_path(
        list(reports.glob("min_distance_policy_summary_*.md"))
    )
    latest_policy24_evidence = newest_path(
        list(reports.glob("GO_LIVE_EVIDENCE_POLICY24_*.md"))
    )
    distance_decision = reports / "MIN_DISTANCE_DECISION_2026-02-09.md"
    nsamples_decision = reports / "N_SAMPLES_DECISION_2026-02-09.md"

    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    sel = cfg.get("selection", {})
    n_samples = sel.get("n_samples")
    min_distance = sel.get("min_distance_km")

    changed_howto = update_howto(howto, latest_distance_summary)
    changed_closeout = update_closeout(
        closeout,
        n_samples=n_samples,
        min_distance=min_distance,
        distance_decision=distance_decision if distance_decision.exists() else None,
        nsamples_decision=nsamples_decision if nsamples_decision.exists() else None,
        policy24_evidence=latest_policy24_evidence,
    )

    key_docs = [
        howto,
        docs / "MIN_DISTANCE_CALCULATION.md",
        reports / "GO_LIVE_EVIDENCE_2026-02-09.md",
        closeout,
    ]
    missing_refs = collect_missing_doc_refs(key_docs, root)

    summary = reports / "FINAL_CONSISTENCY_SUMMARY_2026-02-09.md"
    rows = len(pd.read_csv(root / "data/new_all_tiles.csv"))
    summary_lines = [
        "# Final Consistency Summary (2026-02-09)",
        "",
        "## Frozen Contract Values",
        f"- selection.n_samples: `{n_samples}`",
        f"- selection.min_distance_km: `{min_distance}`",
        "- min_distance geometric reference: `45.0`",
        f"- canonical rows: `{rows}`",
        "",
        "## Evidence Files",
        f"- distance decision: `{distance_decision}` ({'exists' if distance_decision.exists() else 'missing'})",
        f"- n_samples decision: `{nsamples_decision}` ({'exists' if nsamples_decision.exists() else 'missing'})",
        f"- latest distance summary: `{latest_distance_summary}`",
        f"- latest policy24 evidence: `{latest_policy24_evidence}`",
        "",
        "## Doc Mutation",
        f"- THESIS_PIPELINE_HOWTO updated: `{changed_howto}`",
        f"- phase4h_masterarbeit_closeout updated: `{changed_closeout}`",
        "",
        "## Missing Report References in Key Docs",
    ]

    if missing_refs:
        summary_lines.extend([f"- `{doc}` -> `{rel}`" for doc, rel in missing_refs])
    else:
        summary_lines.append("- none")

    summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {summary}")

    if missing_refs:
        raise SystemExit(
            "Missing report references found in key docs; see FINAL_CONSISTENCY_SUMMARY."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

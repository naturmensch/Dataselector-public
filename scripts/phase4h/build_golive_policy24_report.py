#!/usr/bin/env python3
"""Build policy-24 go-live evidence report with deterministic hash comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_selection_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for p in sorted(run_dir.rglob("selection_*.csv")):
        rel = str(p.relative_to(run_dir))
        hashes[rel] = sha256_file(p)
    return hashes


def load_run_metadata(run_dir: Path) -> dict:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return {"_missing": True}
    return json.loads(p.read_text(encoding="utf-8"))


def _meta_value(meta: dict, key: str):
    """Read key from top-level metadata, falling back to `extra` payload."""
    if key in meta:
        return meta.get(key)
    extra = meta.get("extra", {})
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--run-h", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--append-to", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--min-distance", type=float, default=28.5)
    parser.add_argument("--run-tag", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for p in (args.run_a, args.run_b, args.run_h):
        if not p.exists():
            raise SystemExit(f"Run dir does not exist: {p}")

    hashes_a = collect_selection_hashes(args.run_a)
    hashes_b = collect_selection_hashes(args.run_b)
    common = sorted(set(hashes_a) & set(hashes_b))
    only_a = sorted(set(hashes_a) - set(hashes_b))
    only_b = sorted(set(hashes_b) - set(hashes_a))
    mismatched = [rel for rel in common if hashes_a[rel] != hashes_b[rel]]
    matched = len(common) - len(mismatched)

    meta_a = load_run_metadata(args.run_a)
    meta_b = load_run_metadata(args.run_b)
    meta_h = load_run_metadata(args.run_h)

    report_lines = [
        f"# Go-Live Evidence (Policy n=24) — {args.run_tag}",
        "",
        "## Execution Context",
        f"- Generated at (UTC): `{ts}`",
        f"- Seed: `{args.seed}`",
        f"- n_samples: `{args.n_samples}`",
        f"- validation_min_distances: `{args.min_distance}`",
        "",
        "## Run Directories",
        f"- Twin A: `{args.run_a}`",
        f"- Twin B: `{args.run_b}`",
        f"- Hamburg: `{args.run_h}`",
        "",
        "## Determinism Check (A vs B)",
        f"- selection files in A: `{len(hashes_a)}`",
        f"- selection files in B: `{len(hashes_b)}`",
        f"- common relative paths: `{len(common)}`",
        f"- exact hash matches: `{matched}`",
        f"- mismatched hashes: `{len(mismatched)}`",
        f"- only in A: `{len(only_a)}`",
        f"- only in B: `{len(only_b)}`",
    ]

    if mismatched:
        report_lines.extend(
            [
                "",
                "### Mismatched files (A vs B)",
                *[f"- `{rel}`" for rel in mismatched[:20]],
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "### Result",
                "- All common `selection_*.csv` artifacts are byte-identical across A/B.",
            ]
        )

    def fmt_meta(tag: str, meta: dict) -> list[str]:
        if meta.get("_missing"):
            return [f"- {tag}: `run_metadata.json` missing"]
        return [
            f"- {tag}: `n_samples={_meta_value(meta, 'n_samples')}` "
            f"`n_samples_source={_meta_value(meta, 'n_samples_source')}` "
            f"`validation_min_distances={_meta_value(meta, 'validation_min_distances')}` "
            f"`validation_seeds={_meta_value(meta, 'validation_seeds')}` "
            f"`hamburg_shortcut={_meta_value(meta, 'hamburg_shortcut')}`",
        ]

    report_lines.extend(
        [
            "",
            "## run_metadata Summary",
            *fmt_meta("A", meta_a),
            *fmt_meta("B", meta_b),
            *fmt_meta("Hamburg", meta_h),
            "",
            "## Key Artifacts",
            f"- `{args.run_a / 'run_metadata.json'}`",
            f"- `{args.run_b / 'run_metadata.json'}`",
            f"- `{args.run_h / 'run_metadata.json'}`",
            f"- `{args.run_a / 'validation/validation_results.csv'}`",
            f"- `{args.run_b / 'validation/validation_results.csv'}`",
            f"- `{args.run_h / 'validation/validation_results.csv'}`",
            "",
            "## Scientific Note",
            "- This report updates go-live evidence for the current policy baseline "
            f"`n_samples={args.n_samples}` while keeping historical `n=34` evidence intact.",
            "",
        ]
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {args.output_md}")

    if args.append_to is not None:
        section = [
            "",
            f"## Policy-24 Evidence Refresh ({args.run_tag})",
            f"- Evidence report: `{args.output_md}`",
            f"- Twin A: `{args.run_a}`",
            f"- Twin B: `{args.run_b}`",
            f"- Hamburg: `{args.run_h}`",
            f"- A/B common selection files: `{len(common)}`",
            f"- A/B mismatched hashes: `{len(mismatched)}`",
        ]
        with open(args.append_to, "a", encoding="utf-8") as f:
            f.write("\n".join(section) + "\n")
        print(f"appended summary to {args.append_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

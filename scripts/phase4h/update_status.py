#!/usr/bin/env python3
"""Append lightweight status lines to the phase4h scientific plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True, help="Plan markdown path")
    parser.add_argument("--wave", required=True, help="Wave identifier")
    parser.add_argument(
        "--state", required=True, choices=["running", "completed", "failed", "paused"]
    )
    parser.add_argument("--details", required=True, help="Short status details")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan_file)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not plan_path.exists():
        raise SystemExit(f"Plan file does not exist: {plan_path}")

    marker = "## Automation Status Updates"
    content = plan_path.read_text(encoding="utf-8")
    if marker not in content:
        content = content.rstrip() + f"\n\n{marker}\n\n"

    line = f"- `{ts}` | `{args.wave}` | `{args.state}` | {args.details}\n"
    content = content.rstrip() + "\n" + line
    plan_path.write_text(content, encoding="utf-8")
    print(line.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

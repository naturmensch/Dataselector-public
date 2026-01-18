#!/usr/bin/env python3
"""Minimal reports generator.

This script provides a stable entry point expected by tests.
It scans the global outputs folder and writes a simple summary report.
"""
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT_DIR = OUTPUTS / "reports"
from datetime import datetime
DATE_STR = datetime.now().strftime("%Y%m%d")
EXPECTED_REPORT_PATH = OUTPUTS / f"report_{DATE_STR}.md"
REPORT_PATH = REPORT_DIR / "workspace_report.md"


def main():
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Workspace Report\n")
        lines.append(f"Root: {ROOT}\n\n")
        if OUTPUTS.exists():
            lines.append("## Outputs contents\n")
            for p in sorted(OUTPUTS.rglob('*')):
                if p.is_file():
                    rel = p.relative_to(ROOT)
                    lines.append(f"- {rel}\n")
        else:
            lines.append("No outputs directory found.\n")
        text = ''.join(lines)
        # Write both the workspace report and the expected test target
        REPORT_PATH.write_text(text)
        EXPECTED_REPORT_PATH.write_text(text)
        print(f"Report written: {REPORT_PATH} and {EXPECTED_REPORT_PATH}")
        return 0
    except Exception as e:
        print(f"Failed to generate report: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

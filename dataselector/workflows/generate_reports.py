from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    """Generate summary plots and reports via the legacy script.

    Canonical usage:
        python -m dataselector generate-reports -- <script args>
    """

    argv = list(argv or [])
    return run_script("scripts/generate_reports.py", argv)

from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    """Run the XXL full-run monitor (scripts/xxl_full_run_monitor.py) via subprocess."""
    argv = list(argv or [])
    return run_script("scripts/xxl_full_run_monitor.py", argv)

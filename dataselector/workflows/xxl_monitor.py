"""XXL Monitor - Legacy subprocess wrapper (NOT a CLI command).

⚠️ DEPRECATED: This module wraps scripts/xxl_full_run_monitor.py (a shell script).
It is NOT registered as a @cli_command decorator, as it would require full
refactoring to work as a proper Python workflow command.

Status: Kept for backward compatibility if scripts/ is still needed.
         Not exposed via CLI. Internal use only.
"""

from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    """Run the XXL full-run monitor (scripts/xxl_full_run_monitor.py) via subprocess.

    ⚠️ Internal use only - NOT exposed via CLI.
    """
    argv = list(argv or [])
    return run_script("scripts/xxl_full_run_monitor.py", argv)

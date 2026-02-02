from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    return run_script("scripts/optuna_autoscale.py", argv)

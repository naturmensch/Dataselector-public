from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    """Run Optuna optimization runner (legacy script) via canonical CLI.

    This is intentionally a thin wrapper that preserves current behavior while
    we migrate logic inward into the `dataselector` package.
    """

    argv = list(argv or [])
    return run_script("scripts/optuna_optimize.py", argv)

from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    return run_script("scripts/run_thesis_sampler_suite.py", argv)

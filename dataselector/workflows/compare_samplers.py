from __future__ import annotations

from dataselector.workflows._subprocess import run_script


def main(argv: list[str] | None = None) -> int:
    """Workflow wrapper for `scripts/compare_samplers_multi_seed.py`.

    Canonical entrypoint:
        python -m dataselector compare-samplers -- <script args>
    """

    argv = list(argv or [])
    return run_script("scripts/compare_samplers_multi_seed.py", argv)

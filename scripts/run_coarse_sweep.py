"""
DEPRECATED: Coarse Grid Sweep (moved to `scripts/deprecated/run_coarse_sweep.py`).

This script is retained at its original path as a convenience wrapper but is deprecated.
Please use the adaptive LHS-based Exploration pipeline instead:

  - `python scripts/tune_weights_and_run.py --n-lhs <n>`
  - or run the full adaptive pipeline: `python scripts/run_adaptive_pipeline.py --yes`

The new approach (LHS/Sobol) is adaptive, more reproducible, and scientifically preferable.
"""

import sys
from pathlib import Path


def main() -> int:
    """Deprecation wrapper: prints a helpful message and exits with code 0 when run as a script.

    Importing this module will no longer print or exit; the legacy implementation is
    preserved in `scripts/deprecated/run_coarse_sweep.py` for archival purposes.
    """
    print(
        "DEPRECATED: `scripts/run_coarse_sweep.py` has been moved to `scripts/deprecated/` and will be removed in a future release."
    )
    print(
        "Use `scripts/tune_weights_and_run.py` or `scripts/run_adaptive_pipeline.py --yes` instead."
    )
    # Exit early to avoid running legacy code by accident
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Compatibility helpers for runtime dependency checks.

Simple runtime check to detect NumPy <-> Numba incompatibilities and fail fast
with a helpful message.
"""

from __future__ import annotations

from packaging.version import Version, parse


def check_numba_numpy_compatibility(raise_on_error: bool = True) -> bool:
    """Return True if compatible, False otherwise.

    - If numba is not installed, returns True (we don't enforce it).
    - If numba is installed, verifies NumPy is in a supported range (<2.4).
    """
    try:
        import numba  # type: ignore
    except Exception:  # ImportError, etc.
        return True

    try:
        import numpy as np  # type: ignore
    except Exception:
        # If NumPy cannot be imported, leave the error to the importer
        if raise_on_error:
            raise
        return False

    npv = parse(np.__version__)
    # Compatibility policy: numba in the tested range requires numpy < 2.4
    if npv >= Version("2.4"):
        msg = (
            "Detected installed Numba and NumPy >= 2.4, which is known to cause "
            "compatibility issues in our environment.\n"
            "Options: (1) pin NumPy to '<2.4' (recommended), or (2) upgrade Numba "
            "to a release that explicitly supports NumPy >=2.4. See project docs "
            "for details."
        )
        if raise_on_error:
            raise RuntimeError(msg)
        return False

    return True

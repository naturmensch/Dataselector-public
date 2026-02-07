"""Compatibility helpers for runtime dependency checks.

Simple runtime check to detect NumPy <-> Numba incompatibilities and fail fast
with a helpful message.
"""

from __future__ import annotations

import importlib
import sys

from packaging.version import Version, parse


def check_numba_numpy_compatibility(raise_on_error: bool = True) -> bool:
    """Return True if compatible, False otherwise.

    - If numba is not installed, returns True (we don't enforce it).
    - If numba is installed, verifies NumPy is in a supported range (<2.4).
    """
    try:
        pass  # type: ignore
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


def validate_environment_full(raise_on_error: bool = True) -> dict[str, bool]:
    """Validate runtime environment contracts used by tests/CI."""
    result: dict[str, bool] = {
        "numpy": False,
        "numba": False,
        "umap": False,
        "apricot": False,
        "env_name": False,
    }
    errors: list[str] = []

    # NumPy
    try:
        import numpy as np  # type: ignore

        npv = parse(np.__version__)
        if npv >= Version("2.4"):
            errors.append(
                f"NumPy version mismatch: expected <2.4 for current numba policy, got {np.__version__}"
            )
            result["numpy"] = False
        else:
            result["numpy"] = True
    except Exception as exc:
        errors.append(f"NumPy import failed: {exc}")
        result["numpy"] = False

    # Numba
    try:
        import numba  # type: ignore  # noqa: F401

        result["numba"] = True
    except Exception as exc:
        errors.append(f"Numba import failed: {exc}")
        result["numba"] = False

    # Optional deps
    try:
        result["umap"] = importlib.import_module("umap") is not None
    except Exception:
        result["umap"] = False

    try:
        result["apricot"] = importlib.import_module("apricot") is not None
    except Exception:
        result["apricot"] = False

    # Expected conda env name
    prefix = str(getattr(sys, "prefix", ""))
    result["env_name"] = "/envs/dataselector" in prefix.replace("\\", "/")
    if not result["env_name"]:
        errors.append(
            f"Not running in required conda env 'dataselector' (sys.prefix={prefix})"
        )

    if errors and raise_on_error:
        raise RuntimeError("\n".join(errors))

    return result

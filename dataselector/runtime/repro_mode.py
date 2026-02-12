"""Execution-profile helpers for reproducible thesis runs."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}

ALLOWED_PROFILES = {"default", "thesis_repro"}


def _is_expected_interop_reinit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "cannot set number of interop threads after parallel work has started" in text
        or "set_num_interop_threads called" in text
    )


def activate_repro_mode(profile: str = "default", seed: int = 42) -> dict[str, Any]:
    """Activate runtime settings for a selected execution profile.

    Parameters
    ----------
    profile : str
        One of "default" or "thesis_repro".
    seed : int
        Global seed used for Python/NumPy/Torch seeding.

    Returns
    -------
    dict[str, Any]
        Applied settings and capability information for metadata logging.
    """
    if profile not in ALLOWED_PROFILES:
        allowed = ", ".join(sorted(ALLOWED_PROFILES))
        raise ValueError(
            f"Unknown execution profile '{profile}'. Expected one of: {allowed}"
        )

    result: dict[str, Any] = {
        "profile": profile,
        "seed": int(seed),
        "thread_env": {},
        "torch": {"available": False},
        "numpy": {"available": False},
        "repro_degraded": False,
        "parallelism_degraded": False,
        "repro_warnings": [],
    }

    # Export profile/seed for child processes started via subprocess.
    os.environ["DATASELECTOR_EXECUTION_PROFILE"] = profile
    os.environ["DATASELECTOR_EXECUTION_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if profile == "thesis_repro":
        for key, value in THREAD_ENV_VARS.items():
            os.environ[key] = value
            result["thread_env"][key] = value
        shm_dir = Path("/dev/shm")
        if not shm_dir.exists():
            result["parallelism_degraded"] = True
            result["repro_degraded"] = True
            result["repro_warnings"].append("dev_shm_missing")
        else:
            try:
                with tempfile.NamedTemporaryFile(dir=shm_dir, delete=True):
                    pass
            except Exception as exc:
                result["parallelism_degraded"] = True
                result["repro_degraded"] = True
                result["repro_warnings"].append(f"dev_shm_not_writable:{exc}")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        result["numpy"] = {"available": True, "seed": int(seed)}
    except Exception as exc:  # pragma: no cover - numpy is expected in runtime env
        result["numpy"] = {"available": False, "error": str(exc)}

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if profile == "thesis_repro":
            try:
                torch.set_num_threads(1)
            except Exception as exc:
                result["repro_degraded"] = True
                result["repro_warnings"].append(f"set_num_threads_failed:{exc}")
            try:
                torch.set_num_interop_threads(1)
            except Exception as exc:
                # Idempotent behavior: a second profile activation can hit this path
                # even when the effective interop thread count is already correct.
                if _is_expected_interop_reinit_error(exc):
                    current_interop = None
                    try:
                        current_interop = int(torch.get_num_interop_threads())
                    except Exception:
                        current_interop = None
                    if current_interop == 1:
                        result["repro_warnings"].append(
                            "set_num_interop_threads_already_initialized"
                        )
                    else:
                        result["repro_degraded"] = True
                        result["repro_warnings"].append(
                            f"set_num_interop_threads_failed:{exc}"
                        )
                else:
                    result["repro_degraded"] = True
                    result["repro_warnings"].append(f"set_num_interop_threads_failed:{exc}")

            try:
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except Exception as exc:
                result["repro_degraded"] = True
                result["repro_warnings"].append(f"cudnn_flags_failed:{exc}")

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:
                result["repro_degraded"] = True
                result["repro_warnings"].append(
                    f"use_deterministic_algorithms_failed:{exc}"
                )

        result["torch"] = {
            "available": True,
            "seed": int(seed),
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            )
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None,
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False))
            if hasattr(torch.backends, "cudnn")
            else None,
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False))
            if hasattr(torch.backends, "cudnn")
            else None,
        }
    except Exception as exc:  # pragma: no cover - torch may be optional in some envs
        result["torch"] = {"available": False, "error": str(exc)}

    return result

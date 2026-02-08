"""Execution-profile helpers for reproducible thesis runs."""

from __future__ import annotations

import os
import random
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
    }

    # Export profile/seed for child processes started via subprocess.
    os.environ["DATASELECTOR_EXECUTION_PROFILE"] = profile
    os.environ["DATASELECTOR_EXECUTION_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if profile == "thesis_repro":
        for key, value in THREAD_ENV_VARS.items():
            os.environ[key] = value
            result["thread_env"][key] = value

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
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        result["torch"] = {
            "available": True,
            "seed": int(seed),
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
        }
    except Exception as exc:  # pragma: no cover - torch may be optional in some envs
        result["torch"] = {"available": False, "error": str(exc)}

    return result

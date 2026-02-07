"""Canonical spatial schema helpers.

The canonical tile schema uses bounding coordinates:
`ul_x`, `ul_y`, `lr_x`, `lr_y`.
Derived center coordinates are exposed as `center_x`, `center_y`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL_BOUNDS = ("ul_x", "ul_y", "lr_x", "lr_y")
LEGACY_BOUNDS = ("left", "top", "right", "bottom")
CENTER_COLUMNS = ("center_x", "center_y")


def _find_case_insensitive(columns: Iterable[str], target: str) -> str | None:
    target_lc = target.lower()
    for col in columns:
        if str(col).lower() == target_lc:
            return str(col)
    return None


def normalize_spatial_schema(
    df: pd.DataFrame,
    *,
    require_bounds: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Normalize a DataFrame to canonical spatial columns.

    Accepts either canonical bounds (`ul_x`, `ul_y`, `lr_x`, `lr_y`) or legacy
    bounds (`left`, `top`, `right`, `bottom`). Legacy lat/lon center columns are
    intentionally not accepted to keep the hard-cut strict.
    """
    work = df.copy() if copy else df
    rename_map: dict[str, str] = {}

    for name in (*CANONICAL_BOUNDS, *LEGACY_BOUNDS, *CENTER_COLUMNS):
        found = _find_case_insensitive(work.columns, name)
        if found is not None and found != name:
            rename_map[found] = name
    if rename_map:
        work = work.rename(columns=rename_map)

    has_canonical = all(col in work.columns for col in CANONICAL_BOUNDS)
    has_legacy_bounds = all(col in work.columns for col in LEGACY_BOUNDS)

    if has_canonical:
        pass
    elif has_legacy_bounds:
        work["ul_x"] = work["left"]
        work["ul_y"] = work["top"]
        work["lr_x"] = work["right"]
        work["lr_y"] = work["bottom"]
    elif require_bounds:
        raise ValueError(
            "Spatial schema requires ul_x/ul_y/lr_x/lr_y columns "
            "(or left/top/right/bottom for conversion)."
        )

    for col in CANONICAL_BOUNDS:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if all(col in work.columns for col in CANONICAL_BOUNDS):
        work["center_x"] = (work["ul_x"] + work["lr_x"]) / 2.0
        work["center_y"] = (work["ul_y"] + work["lr_y"]) / 2.0
    elif all(col in work.columns for col in CENTER_COLUMNS):
        work["center_x"] = pd.to_numeric(work["center_x"], errors="coerce")
        work["center_y"] = pd.to_numeric(work["center_y"], errors="coerce")
    elif require_bounds:
        raise ValueError(
            "Spatial schema normalization could not derive center_x/center_y."
        )

    return work


def center_array(df: pd.DataFrame) -> np.ndarray:
    """Return center coordinates as `[[x, y], ...]` float array."""
    if not all(c in df.columns for c in CENTER_COLUMNS):
        raise ValueError("Missing center_x/center_y columns")
    return df[["center_x", "center_y"]].to_numpy(dtype=float)


def coordinates_look_projected(df: pd.DataFrame) -> bool:
    """Heuristic CRS-unit check based on center magnitude."""
    coords = center_array(df)
    if coords.size == 0:
        return False
    x = coords[:, 0]
    y = coords[:, 1]
    return bool(np.nanmax(np.abs(x)) > 180.0 or np.nanmax(np.abs(y)) > 90.0)


def spatial_spread(df: pd.DataFrame, indices: np.ndarray | list[int]) -> float:
    """Compute mean std-dev across center_x/center_y for selected rows."""
    if len(indices) == 0:
        return 0.0
    subset = df.iloc[np.asarray(indices, dtype=int)]
    if not all(c in subset.columns for c in CENTER_COLUMNS):
        subset = normalize_spatial_schema(subset, require_bounds=True, copy=True)
    return float(subset[["center_x", "center_y"]].std().mean())

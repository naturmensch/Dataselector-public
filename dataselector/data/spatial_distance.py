"""Spatial distance helpers for leakage-safe splitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dataselector.data.spatial_schema import normalize_spatial_schema


def _parse_epsg(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text.startswith("EPSG:"):
        text = text.split(":", 1)[1]
    try:
        return int(text)
    except Exception:
        return None


def _infer_source_epsg(metadata: pd.DataFrame) -> int | None:
    source = metadata.attrs.get("source_crs") if hasattr(metadata, "attrs") else None
    epsg = _parse_epsg(source)
    if epsg is not None:
        return epsg

    # Pragmatic fallback used in current pipeline: projected center magnitudes.
    if "center_x" in metadata.columns and "center_y" in metadata.columns:
        x_abs = float(np.nanmax(np.abs(metadata["center_x"].to_numpy(dtype=float))))
        y_abs = float(np.nanmax(np.abs(metadata["center_y"].to_numpy(dtype=float))))
        if x_abs > 180.0 or y_abs > 90.0:
            return 3857
        return 4326
    return None


def tile_bounds_to_metric(
    metadata: pd.DataFrame,
    *,
    target_epsg: int = 25832,
    strict: bool = True,
) -> pd.DataFrame:
    """Convert tile bounds to metric rectangles in target CRS.

    Returns dataframe with:
    - `_minx_m`, `_maxx_m`, `_miny_m`, `_maxy_m`
    - `_center_x_m`, `_center_y_m`
    """
    work = normalize_spatial_schema(metadata, require_bounds=True, copy=True)
    source_epsg = _infer_source_epsg(work)
    if source_epsg is None:
        if strict:
            raise RuntimeError("Unable to infer source CRS for metric distance conversion.")
        # Fallback: treat as already metric.
        source_epsg = target_epsg

    ul_x = work["ul_x"].to_numpy(dtype=float)
    ul_y = work["ul_y"].to_numpy(dtype=float)
    lr_x = work["lr_x"].to_numpy(dtype=float)
    lr_y = work["lr_y"].to_numpy(dtype=float)
    ur_x = lr_x
    ur_y = ul_y
    ll_x = ul_x
    ll_y = lr_y

    if source_epsg != target_epsg:
        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                f"EPSG:{source_epsg}",
                f"EPSG:{target_epsg}",
                always_xy=True,
            )
            ul_x, ul_y = transformer.transform(ul_x, ul_y)
            lr_x, lr_y = transformer.transform(lr_x, lr_y)
            ur_x, ur_y = transformer.transform(ur_x, ur_y)
            ll_x, ll_y = transformer.transform(ll_x, ll_y)
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"Could not transform bounds to EPSG:{target_epsg}: {exc}"
                ) from exc

    minx = np.minimum.reduce([ul_x, lr_x, ur_x, ll_x])
    maxx = np.maximum.reduce([ul_x, lr_x, ur_x, ll_x])
    miny = np.minimum.reduce([ul_y, lr_y, ur_y, ll_y])
    maxy = np.maximum.reduce([ul_y, lr_y, ur_y, ll_y])

    out = work.copy()
    out["_minx_m"] = minx
    out["_maxx_m"] = maxx
    out["_miny_m"] = miny
    out["_maxy_m"] = maxy
    out["_center_x_m"] = (minx + maxx) / 2.0
    out["_center_y_m"] = (miny + maxy) / 2.0
    if hasattr(out, "attrs"):
        out.attrs["source_crs"] = f"EPSG:{source_epsg}"
        out.attrs["metric_crs"] = f"EPSG:{target_epsg}"
        out.attrs["transform_applied"] = bool(source_epsg != target_epsg)
    return out


def edge_distance_km(
    a_minx: float,
    a_maxx: float,
    a_miny: float,
    a_maxy: float,
    b_minx: float,
    b_maxx: float,
    b_miny: float,
    b_maxy: float,
) -> float:
    """Rectangle edge-to-edge distance in kilometers."""
    dx = max(a_minx - b_maxx, b_minx - a_maxx, 0.0)
    dy = max(a_miny - b_maxy, b_miny - a_maxy, 0.0)
    return float(np.sqrt(dx * dx + dy * dy) / 1000.0)


def center_distance_km(ax: float, ay: float, bx: float, by: float) -> float:
    """Center-to-center distance in kilometers."""
    dx = float(ax) - float(bx)
    dy = float(ay) - float(by)
    return float(np.sqrt(dx * dx + dy * dy) / 1000.0)


def pairwise_edge_distance_matrix(metric_bounds: pd.DataFrame) -> np.ndarray:
    """Compute symmetric pairwise edge-distance matrix (km)."""
    n = len(metric_bounds)
    dist = np.zeros((n, n), dtype=float)
    minx = metric_bounds["_minx_m"].to_numpy(dtype=float)
    maxx = metric_bounds["_maxx_m"].to_numpy(dtype=float)
    miny = metric_bounds["_miny_m"].to_numpy(dtype=float)
    maxy = metric_bounds["_maxy_m"].to_numpy(dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = edge_distance_km(
                minx[i],
                maxx[i],
                miny[i],
                maxy[i],
                minx[j],
                maxx[j],
                miny[j],
                maxy[j],
            )
            dist[i, j] = d
            dist[j, i] = d
    return dist

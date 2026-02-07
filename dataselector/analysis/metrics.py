from typing import Dict

import numpy as np
import pandas as pd

from dataselector.data.io import get_metric_gdf
from dataselector.data.spatial_schema import (
    coordinates_look_projected,
    normalize_spatial_schema,
)


def compute_metrics(
    selected_idx,
    metadata: pd.DataFrame,
    cluster_labels: np.ndarray,
    features: np.ndarray,
) -> Dict:
    # Make selection indices safe: ensure integer dtype and handle empty selections
    sel_idx_arr = (
        np.asarray(selected_idx, dtype=int)
        if getattr(selected_idx, "__len__", None) is not None and len(selected_idx) > 0
        else np.array([], dtype=int)
    )
    selected = metadata.iloc[sel_idx_arr]
    if "year" in selected.columns:
        years = pd.to_numeric(selected["year"], errors="coerce").dropna()
    else:
        years = pd.Series(dtype=float)
    temporal_std = float(years.std()) if len(years) > 1 else 0.0
    temporal_range = int(years.max() - years.min()) if len(years) > 1 else 0
    wwi_frac = float(years.between(1914, 1918).mean() * 100) if len(years) > 0 else 0.0

    # Spatial mean/min distance
    gdf_metric = None
    try:
        gdf_metric = get_metric_gdf(metadata)
    except Exception:
        gdf_metric = getattr(metadata, "gdf_metric", None)

    if gdf_metric is not None:
        # Prefer projected metric coordinates if available.
        proj = gdf_metric.loc[selected.index, ["_proj_x", "_proj_y"]].values
        if len(proj) > 1:
            diffs = proj[:, None, :] - proj[None, :, :]
            dmat = np.sqrt((diffs**2).sum(axis=2))
            i, j = np.triu_indices(len(proj), k=1)
            dists_m = dmat[i, j]
            spatial_mean = float(np.mean(dists_m) / 1000.0)
            spatial_min = float(np.min(dists_m) / 1000.0)
        else:
            spatial_mean = 0.0
            spatial_min = 0.0
    else:
        # Fallback to center coordinates from canonical spatial schema.
        from dataselector.data.metadata_processor import MetadataProcessor

        selected = normalize_spatial_schema(selected, require_bounds=True, copy=True)
        coords = selected[["center_y", "center_x"]].to_numpy(dtype=float)
        is_projected = coordinates_look_projected(selected)

        if len(coords) > 1:
            dists = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    if is_projected:
                        dx = coords[i, 1] - coords[j, 1]
                        dy = coords[i, 0] - coords[j, 0]
                        dists.append(float(np.sqrt(dx * dx + dy * dy) / 1000.0))
                    else:
                        dists.append(
                            MetadataProcessor.calculate_spatial_distance(
                                coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                            )
                        )
            spatial_mean = float(np.mean(dists))
            spatial_min = float(np.min(dists))
        else:
            spatial_mean = 0.0
            spatial_min = 0.0

    clusters_covered = (
        int(len(np.unique(cluster_labels[sel_idx_arr]))) if sel_idx_arr.size > 0 else 0
    )

    return {
        "n_selected": int(sel_idx_arr.size),
        "temporal_std": temporal_std,
        "temporal_range": temporal_range,
        "wwi_percent": wwi_frac,
        "spatial_mean_km": spatial_mean,
        "spatial_min_km": spatial_min,
        "clusters_covered": clusters_covered,
    }

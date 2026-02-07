from typing import Dict

import numpy as np
import pandas as pd

from dataselector.data.io import get_metric_gdf


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
    temporal_std = float(selected["year"].std()) if len(selected) > 1 else 0.0
    temporal_range = (
        int(selected["year"].max() - selected["year"].min()) if len(selected) > 1 else 0
    )
    wwi_frac = float((selected["year"].between(1914, 1918)).mean() * 100)

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
        # Fallback to Haversine distance on N/left columns.
        from dataselector.data.metadata_processor import MetadataProcessor

        if "N" in selected.columns and "left" in selected.columns:
            coords = selected[["N", "left"]].to_numpy(dtype=float)
        else:
            coords = np.empty((0, 2), dtype=float)

        mp = MetadataProcessor("")
        if len(coords) > 1:
            dists = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dists.append(
                        mp.calculate_spatial_distance(
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

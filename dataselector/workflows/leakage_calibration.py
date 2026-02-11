"""Data-driven leakage buffer calibration for spatial splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataselector.data.spatial_distance import (
    pairwise_edge_distance_matrix,
    tile_bounds_to_metric,
)
from dataselector.runtime.parameter_snapshot import compute_file_sha256


@dataclass(frozen=True)
class LeakageCalibrationResult:
    d_leak_km: float
    calibration_csv: Path
    policy_json: Path
    method: str
    threshold_similarity: float


def _pairwise_similarity(features: np.ndarray) -> np.ndarray:
    feats = np.asarray(features, dtype=float)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    feats = feats / norms
    sim = feats @ feats.T
    np.fill_diagonal(sim, np.nan)
    return sim


def calibrate_leakage_buffer(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    split_policy: dict[str, Any],
    leakage_buffer_km: float | str = "auto",
) -> LeakageCalibrationResult:
    """Calibrate leakage buffer distance (d_leak) from feature similarity decay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    metric_bounds = tile_bounds_to_metric(metadata, target_epsg=25832, strict=True)
    distance_mat = pairwise_edge_distance_matrix(metric_bounds)
    similarity_mat = _pairwise_similarity(features)

    tri = np.triu_indices(len(metric_bounds), k=1)
    distances = distance_mat[tri]
    similarities = similarity_mat[tri]
    valid_mask = np.isfinite(distances) & np.isfinite(similarities)
    distances = distances[valid_mask]
    similarities = similarities[valid_mask]

    if len(distances) == 0:
        raise RuntimeError("Leakage calibration requires at least one pairwise distance.")

    leakage_cfg = (split_policy.get("leakage", {}) or {}).get("calibration", {}) or {}
    bin_width = float(leakage_cfg.get("bin_width_km", 5.0))
    min_pairs = int(leakage_cfg.get("min_pairs_per_bin", 30))
    stability_bins = int(leakage_cfg.get("stability_bins", 2))
    far_percentile = float(leakage_cfg.get("far_percentile", 75))
    epsilon = float(leakage_cfg.get("similarity_epsilon", 0.02))

    max_dist = float(np.nanmax(distances))
    bins = np.arange(0.0, max_dist + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([0.0, max(1.0, max_dist + 1e-6)])

    records: list[dict[str, Any]] = []
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (distances >= start) & (distances < end)
        vals = similarities[mask]
        if len(vals) == 0:
            continue
        records.append(
            {
                "bin_start_km": float(start),
                "bin_end_km": float(end),
                "n_pairs": int(len(vals)),
                "mean_similarity": float(np.mean(vals)),
                "std_similarity": float(np.std(vals)),
            }
        )

    calib_df = pd.DataFrame(records)
    if calib_df.empty:
        raise RuntimeError("Leakage calibration produced no populated distance bins.")

    far_mask = distances >= np.percentile(distances, far_percentile)
    far_vals = similarities[far_mask]
    if len(far_vals) == 0:
        far_vals = similarities
    threshold = float(np.mean(far_vals) + epsilon)

    if isinstance(leakage_buffer_km, str) and leakage_buffer_km.strip().lower() == "auto":
        d_leak_km = None
        means = calib_df["mean_similarity"].to_numpy(dtype=float)
        starts = calib_df["bin_start_km"].to_numpy(dtype=float)
        counts = calib_df["n_pairs"].to_numpy(dtype=int)
        for i in range(len(calib_df)):
            if counts[i] < min_pairs:
                continue
            if means[i] > threshold:
                continue
            window_end = min(len(calib_df), i + 1 + max(0, stability_bins))
            window_means = means[i:window_end]
            window_counts = counts[i:window_end]
            if len(window_means) < max(1, stability_bins):
                continue
            if np.all(window_counts >= min_pairs) and np.all(window_means <= threshold):
                d_leak_km = float(starts[i])
                break
        if d_leak_km is None:
            # Conservative fallback if no stable threshold crossing is found.
            d_leak_km = float(np.percentile(distances, 25))
        method = "auto_similarity_decay"
    else:
        d_leak_km = float(leakage_buffer_km)
        method = "explicit_override"

    calibration_csv = policy_dir / "leakage_calibration.csv"
    calib_df.to_csv(calibration_csv, index=False)

    distance_policy = {
        "method": method,
        "distance_metric": "edge_to_edge_km",
        "d_leak_km": float(d_leak_km),
        "threshold_similarity": float(threshold),
        "calibration": {
            "bin_width_km": bin_width,
            "min_pairs_per_bin": min_pairs,
            "stability_bins": stability_bins,
            "far_percentile": far_percentile,
            "similarity_epsilon": epsilon,
        },
        "inputs": {
            "n_tiles": int(len(metadata)),
            "n_pairs": int(len(distances)),
            "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
            "source_crs": metric_bounds.attrs.get("source_crs"),
            "metric_crs": metric_bounds.attrs.get("metric_crs"),
        },
    }
    policy_json = policy_dir / "distance_policy.json"
    policy_json.write_text(
        json.dumps(distance_policy, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    distance_policy["policy_sha256"] = compute_file_sha256(policy_json)
    policy_json.write_text(
        json.dumps(distance_policy, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return LeakageCalibrationResult(
        d_leak_km=float(d_leak_km),
        calibration_csv=calibration_csv,
        policy_json=policy_json,
        method=method,
        threshold_similarity=threshold,
    )

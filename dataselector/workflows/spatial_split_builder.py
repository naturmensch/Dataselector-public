"""Leakage-safe spatial split builder."""

from __future__ import annotations

import json
import math
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
class SpatialSplitBuildResult:
    split_manifest_path: Path
    split_manifest_sha256: str
    component_count: int
    split_sizes: dict[str, int]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _component_labels(distance_mat: np.ndarray, threshold_km: float) -> np.ndarray:
    n = distance_mat.shape[0]
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if float(distance_mat[i, j]) < float(threshold_km):
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    root_to_label: dict[int, int] = {}
    labels = np.zeros(n, dtype=int)
    next_label = 0
    for idx, root in enumerate(roots):
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[idx] = root_to_label[root]
    return labels


def _resolve_tile_ids(metadata: pd.DataFrame) -> list[str]:
    if "shortName" in metadata.columns:
        ids = metadata["shortName"].astype(str).tolist()
    elif "longName" in metadata.columns:
        ids = metadata["longName"].astype(str).tolist()
    else:
        ids = [f"tile_{i}" for i in range(len(metadata))]
    return ids


def _year_bin_labels(years: np.ndarray, bins: list[float]) -> np.ndarray:
    if len(bins) < 2:
        return np.array(["all"] * len(years), dtype=object)
    # right=False keeps deterministic lower-inclusive bins.
    out = pd.cut(years, bins=bins, right=False, include_lowest=True)
    return np.asarray(out.astype(str), dtype=object)


def _region_labels(metric_df: pd.DataFrame) -> np.ndarray:
    cx = metric_df["_center_x_m"].to_numpy(dtype=float)
    cy = metric_df["_center_y_m"].to_numpy(dtype=float)
    mx = float(np.median(cx))
    my = float(np.median(cy))
    labels = []
    for x, y in zip(cx, cy):
        ns = "N" if y >= my else "S"
        ew = "E" if x >= mx else "W"
        labels.append(f"{ns}{ew}")
    return np.asarray(labels, dtype=object)


def _choose_split_for_component(
    *,
    component_size: int,
    component_year: str,
    component_region: str,
    split_totals: dict[str, int],
    target_totals: dict[str, float],
    split_year_counts: dict[str, dict[str, int]],
    split_region_counts: dict[str, dict[str, int]],
) -> str:
    candidates = ["train", "val", "test"]
    best = None
    best_score = math.inf
    for split in candidates:
        projected_total = split_totals[split] + component_size
        count_penalty = abs(projected_total - target_totals[split])
        year_penalty = split_year_counts[split].get(component_year, 0)
        region_penalty = split_region_counts[split].get(component_region, 0)
        # Main objective: keep ratio; secondary: balance year/region tags.
        score = (10.0 * count_penalty) + (1.0 * year_penalty) + (1.0 * region_penalty)
        if score < best_score:
            best_score = score
            best = split
    assert best is not None
    return best


def build_spatial_splits(
    *,
    metadata: pd.DataFrame,
    output_dir: Path,
    split_policy: dict[str, Any],
    tile_exclusion_policy_sha256: str | None,
    d_leak_km: float,
    split_seed: int = 42,
) -> SpatialSplitBuildResult:
    """Build deterministic leakage-safe train/val/test split manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    metric_df = tile_bounds_to_metric(metadata, target_epsg=25832, strict=True)
    dist_mat = pairwise_edge_distance_matrix(metric_df)
    labels = _component_labels(dist_mat, threshold_km=float(d_leak_km))

    tile_ids = _resolve_tile_ids(metric_df)
    years = pd.to_numeric(metric_df.get("year"), errors="coerce").fillna(0).to_numpy(dtype=float)

    split_cfg = split_policy.get("split", {}) if isinstance(split_policy, dict) else {}
    ratio_cfg = split_cfg.get("ratios", {}) if isinstance(split_cfg, dict) else {}
    ratios = {
        "train": float(ratio_cfg.get("train", 0.70)),
        "val": float(ratio_cfg.get("val", 0.15)),
        "test": float(ratio_cfg.get("test", 0.15)),
    }
    total_ratio = sum(ratios.values()) or 1.0
    ratios = {k: v / total_ratio for k, v in ratios.items()}

    year_bins = split_cfg.get("year_bins", [1800, 1875, 1900, 1918, 1933, 1950])
    year_labels = _year_bin_labels(years, list(year_bins))
    region_labels = _region_labels(metric_df)

    comp_ids = np.unique(labels)
    components: list[dict[str, Any]] = []
    for cid in comp_ids:
        idx = np.where(labels == cid)[0]
        size = int(len(idx))
        comp_year = pd.Series(year_labels[idx]).mode().iat[0]
        comp_region = pd.Series(region_labels[idx]).mode().iat[0]
        components.append(
            {
                "component_id": int(cid),
                "indices": idx.tolist(),
                "size": size,
                "year_tag": str(comp_year),
                "region_tag": str(comp_region),
            }
        )
    # Deterministic assignment by size desc then component id.
    components = sorted(components, key=lambda c: (-c["size"], c["component_id"]))

    n_total = len(metric_df)
    target_totals = {k: ratios[k] * n_total for k in ratios}
    split_totals = {"train": 0, "val": 0, "test": 0}
    split_year_counts = {"train": {}, "val": {}, "test": {}}
    split_region_counts = {"train": {}, "val": {}, "test": {}}
    split_indices = {"train": [], "val": [], "test": []}
    component_assignments: dict[int, str] = {}

    for comp in components:
        chosen = _choose_split_for_component(
            component_size=comp["size"],
            component_year=comp["year_tag"],
            component_region=comp["region_tag"],
            split_totals=split_totals,
            target_totals=target_totals,
            split_year_counts=split_year_counts,
            split_region_counts=split_region_counts,
        )
        component_assignments[int(comp["component_id"])] = chosen
        split_indices[chosen].extend(comp["indices"])
        split_totals[chosen] += int(comp["size"])
        split_year_counts[chosen][comp["year_tag"]] = (
            split_year_counts[chosen].get(comp["year_tag"], 0) + int(comp["size"])
        )
        split_region_counts[chosen][comp["region_tag"]] = (
            split_region_counts[chosen].get(comp["region_tag"], 0) + int(comp["size"])
        )

    manifest = {
        "version": 1,
        "seed": int(split_seed),
        "d_leak_km": float(d_leak_km),
        "distance_metric": "edge_to_edge_km",
        "ratios": ratios,
        "source_crs": metric_df.attrs.get("source_crs"),
        "metric_crs": metric_df.attrs.get("metric_crs"),
        "tile_exclusion_policy_sha256": tile_exclusion_policy_sha256,
        "component_count": int(len(components)),
        "split_counts": {k: int(len(v)) for k, v in split_indices.items()},
        "split_tile_ids": {
            split: [str(tile_ids[i]) for i in sorted(indices)]
            for split, indices in split_indices.items()
        },
        "split_indices": {
            split: [int(i) for i in sorted(indices)]
            for split, indices in split_indices.items()
        },
        "component_assignments": {
            str(cid): split for cid, split in sorted(component_assignments.items())
        },
    }

    manifest_path = split_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    manifest_sha = compute_file_sha256(manifest_path)
    manifest["split_manifest_sha256"] = manifest_sha
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return SpatialSplitBuildResult(
        split_manifest_path=manifest_path,
        split_manifest_sha256=manifest_sha,
        component_count=int(len(components)),
        split_sizes={k: int(len(v)) for k, v in split_indices.items()},
    )

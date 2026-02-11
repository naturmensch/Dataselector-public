"""Leakage audit for spatial split manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataselector.data.spatial_distance import (
    edge_distance_km,
    tile_bounds_to_metric,
)


@dataclass(frozen=True)
class LeakageAuditResult:
    audit_csv_path: Path
    violations_count: int
    min_train_val_km: float | None
    min_train_test_km: float | None
    min_val_test_km: float | None


def _resolve_id_col(metadata: pd.DataFrame) -> str:
    if "shortName" in metadata.columns:
        return "shortName"
    if "longName" in metadata.columns:
        return "longName"
    return "__index__"


def audit_split_leakage(
    *,
    metadata: pd.DataFrame,
    split_manifest: dict[str, Any],
    d_leak_km: float,
    output_dir: Path,
) -> LeakageAuditResult:
    """Check all inter-split pairs for edge-distance leakage violations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    metric_df = tile_bounds_to_metric(metadata, target_epsg=25832, strict=True)
    id_col = _resolve_id_col(metric_df)
    if id_col == "__index__":
        metric_df = metric_df.copy()
        metric_df[id_col] = [f"tile_{i}" for i in range(len(metric_df))]

    split_indices = split_manifest.get("split_indices", {})
    train_idx = [int(i) for i in split_indices.get("train", [])]
    val_idx = [int(i) for i in split_indices.get("val", [])]
    test_idx = [int(i) for i in split_indices.get("test", [])]

    rows: list[dict[str, Any]] = []
    minima: dict[str, float | None] = {
        "train_val": None,
        "train_test": None,
        "val_test": None,
    }

    def _pair_distance(i: int, j: int) -> float:
        a = metric_df.loc[i]
        b = metric_df.loc[j]
        return edge_distance_km(
            float(a["_minx_m"]),
            float(a["_maxx_m"]),
            float(a["_miny_m"]),
            float(a["_maxy_m"]),
            float(b["_minx_m"]),
            float(b["_maxx_m"]),
            float(b["_miny_m"]),
            float(b["_maxy_m"]),
        )

    def _scan(group_a: list[int], group_b: list[int], label: str) -> None:
        local_min = None
        for i in group_a:
            for j in group_b:
                d = _pair_distance(i, j)
                local_min = d if local_min is None else min(local_min, d)
                if d < float(d_leak_km):
                    rows.append(
                        {
                            "pair_type": label,
                            "tile_a_idx": int(i),
                            "tile_b_idx": int(j),
                            "tile_a_id": str(metric_df.loc[i, id_col]),
                            "tile_b_id": str(metric_df.loc[j, id_col]),
                            "edge_distance_km": float(d),
                            "d_leak_km": float(d_leak_km),
                            "violation": True,
                        }
                    )
        minima[label] = local_min

    _scan(train_idx, val_idx, "train_val")
    _scan(train_idx, test_idx, "train_test")
    _scan(val_idx, test_idx, "val_test")

    if not rows:
        rows.append(
            {
                "pair_type": "summary",
                "tile_a_idx": -1,
                "tile_b_idx": -1,
                "tile_a_id": "",
                "tile_b_id": "",
                "edge_distance_km": float("nan"),
                "d_leak_km": float(d_leak_km),
                "violation": False,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_csv = split_dir / "leakage_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    violations = int((audit_df.get("violation", pd.Series(dtype=bool)) == True).sum())  # noqa: E712

    return LeakageAuditResult(
        audit_csv_path=audit_csv,
        violations_count=violations,
        min_train_val_km=minima["train_val"],
        min_train_test_km=minima["train_test"],
        min_val_test_km=minima["val_test"],
    )

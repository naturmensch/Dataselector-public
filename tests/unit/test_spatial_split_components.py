from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataselector.workflows.spatial_split_builder import build_spatial_splits


def test_spatial_split_builder_keeps_components_intact(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "shortName": ["A1", "A2", "B1", "B2"],
            "ul_x": [0, 1000, 20000, 21000],
            "ul_y": [1000, 1000, 1000, 1000],
            "lr_x": [1000, 2000, 21000, 22000],
            "lr_y": [0, 0, 0, 0],
            "year": [1900, 1901, 1930, 1931],
        }
    )
    metadata.attrs["source_crs"] = "EPSG:25832"
    split_policy = {
        "split": {
            "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
            "year_bins": [1800, 1910, 1950],
            "region_mode": "quadrant",
        }
    }

    result = build_spatial_splits(
        metadata=metadata,
        output_dir=tmp_path,
        split_policy=split_policy,
        tile_exclusion_policy_sha256="abc",
        d_leak_km=0.1,
        split_seed=42,
    )
    payload = json.loads(result.split_manifest_path.read_text(encoding="utf-8"))
    assert payload["component_count"] == 2
    assigned_splits = set(payload["component_assignments"].values())
    assert len(assigned_splits) >= 1
    all_ids = (
        payload["split_tile_ids"]["train"]
        + payload["split_tile_ids"]["val"]
        + payload["split_tile_ids"]["test"]
    )
    assert sorted(all_ids) == ["A1", "A2", "B1", "B2"]

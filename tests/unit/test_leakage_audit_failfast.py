from __future__ import annotations

import pandas as pd

from dataselector.workflows.leakage_audit import audit_split_leakage


def test_leakage_audit_detects_inter_split_violations(tmp_path) -> None:
    metadata = pd.DataFrame(
        {
            "shortName": ["T1", "T2"],
            "ul_x": [0.0, 1000.0],
            "ul_y": [1000.0, 1000.0],
            "lr_x": [1000.0, 2000.0],
            "lr_y": [0.0, 0.0],
        }
    )
    metadata.attrs["source_crs"] = "EPSG:25832"
    split_manifest = {
        "split_indices": {"train": [0], "val": [], "test": [1]},
    }
    result = audit_split_leakage(
        metadata=metadata,
        split_manifest=split_manifest,
        d_leak_km=0.5,
        output_dir=tmp_path,
    )
    assert result.violations_count > 0
    assert result.audit_csv_path.exists()

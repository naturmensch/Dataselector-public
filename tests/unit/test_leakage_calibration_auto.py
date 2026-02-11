from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataselector.workflows.leakage_calibration import calibrate_leakage_buffer


def test_leakage_calibration_auto_writes_artifacts(tmp_path: Path) -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "ul_x": [0, 1000, 2000, 20000, 21000, 22000],
            "ul_y": [1000, 1000, 1000, 1000, 1000, 1000],
            "lr_x": [1000, 2000, 3000, 21000, 22000, 23000],
            "lr_y": [0, 0, 0, 0, 0, 0],
            "year": [1900, 1901, 1902, 1930, 1931, 1932],
        }
    )
    metadata.attrs["source_crs"] = "EPSG:25832"

    split_policy = {
        "leakage": {
            "calibration": {
                "bin_width_km": 2.0,
                "min_pairs_per_bin": 1,
                "stability_bins": 1,
                "far_percentile": 60,
                "similarity_epsilon": 0.01,
            }
        }
    }
    result = calibrate_leakage_buffer(
        features=features,
        metadata=metadata,
        output_dir=tmp_path,
        split_policy=split_policy,
        leakage_buffer_km="auto",
    )
    assert result.d_leak_km >= 0.0
    assert result.calibration_csv.exists()
    assert result.policy_json.exists()

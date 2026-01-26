import pytest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_pareto_candidates import validate


@pytest.mark.slow
def _make_pareto_csv(tmp_path):
    df = pd.DataFrame(
        [
            {"alpha": 0.7, "beta": 0.15, "gamma": 0.15},
            {"alpha": 0.5, "beta": 0.35, "gamma": 0.15},
        ]
    )
    p = tmp_path / "pareto.csv"
    df.to_csv(p, index=False)
    # create fake outputs for features and metadata in a temp outdir
    out = tmp_path / "outputs"
    out.mkdir()
    np.save(out / "features.npy", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    pd.DataFrame(
        {
            "longName": ["a.png", "b.png", "c.png"],
            "N": [50, 51, 52],
            "left": [10, 11, 12],
            "year": [1900, 1914, 1918],
            "image_path": ["a", "b", "c"],
        }
    ).to_csv(out / "metadata.csv", index=False)
    return str(p), str(out)


@pytest.mark.slow
def test_validate_small(tmp_path, monkeypatch):
    pareto, outdir = _make_pareto_csv(tmp_path)
    # Run validation with small params to be quick and point to temp outdir
    df = validate(
        pareto, min_distances=[10], seeds=[1, 2], n_samples=2, output_dir=outdir
    )
    assert "n_selected" in df.columns
    assert len(df) == 4

    # Check that plots were generated
    plots_dir = Path(outdir) / "plots" / "sel_a0.7_b0.15_g0.15_d10_s1"
    assert plots_dir.exists()
    assert (plots_dir / "spatial_distribution.png").exists()

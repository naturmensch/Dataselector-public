from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_pareto_candidates import validate


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
    # Increase sample size so that n_neighbors (15) < N and UMAP does not trigger the k>=N warning
    N = 20
    rng = np.random.RandomState(0)
    features = rng.rand(N, 2)
    np.save(out / "features.npy", features)
    md = pd.DataFrame(
        {
            "N": list(range(50, 50 + N)),
            "left": list(range(10, 10 + N)),
            "year": [1900 + (i % 30) for i in range(N)],
            "image_path": [f"img_{i}.png" for i in range(N)],
        }
    )
    md.to_csv(out / "metadata.csv", index=False)
    return str(p), str(out)


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

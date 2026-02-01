from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


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
    # Increase sample size so that n_neighbors (15) < N and UMAP does not trigger the k>=N warning
    N = 20
    rng = np.random.RandomState(0)
    features = rng.rand(N, 2)
    np.save(out / "features.npy", features)
    md = pd.DataFrame(
        {
        }
    )
    md.to_csv(out / "metadata.csv", index=False)
    return str(p), str(out)


@pytest.mark.slow
def test_validate_small(tmp_path, monkeypatch):
    pareto, outdir = _make_pareto_csv(tmp_path)

    # Dynamically load the validate function from script after skip checks
    ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "validate_pareto_candidates", ROOT / "scripts" / "validate_pareto_candidates.py"
    )
    validate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validate_mod)
    validate = validate_mod.validate

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

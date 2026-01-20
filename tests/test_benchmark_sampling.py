import pytest
import importlib.util
from pathlib import Path

pytest.importorskip("numba", exc_type=ImportError)
pytestmark = pytest.mark.integration


def test_benchmark_small(tmp_path):
    # Dynamically load the script module to avoid module-level imports after pytest.skip
    ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "benchmark_sampling_methods", ROOT / "scripts" / "benchmark_sampling_methods.py"
    )
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)

    sample_sizes = [10, 20]
    df = bm.benchmark_space_filling(sample_sizes, n_trials=3, dim=3, seed=123)
    assert "min_pairwise_dist" in df.columns
    assert set(df["method"].unique()) == {"lhs", "sobol"}

    # write outputs to tmp and check files
    out_prefix = tmp_path / "test_bench"
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    assert out_prefix.with_suffix(".csv").exists()

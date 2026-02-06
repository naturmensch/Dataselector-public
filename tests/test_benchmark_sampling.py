from pathlib import Path

import pandas as pd
import pytest

from dataselector.workflows import benchmark_sampling as bm

pytestmark = pytest.mark.integration


def test_benchmark_small(tmp_path: Path):
    csv_path, plot_path = bm._run_sampling_benchmark(
        out_dir_path=tmp_path,
        n_samples=[10, 20],
        n_dims=3,
        n_repeats=3,
    )

    assert csv_path.exists()
    assert plot_path.exists()

    df = pd.read_csv(csv_path)
    assert {"method", "n_samples", "discrepancy", "min_distance"}.issubset(df.columns)
    assert set(df["method"].unique()) == {"lhs", "sobol"}

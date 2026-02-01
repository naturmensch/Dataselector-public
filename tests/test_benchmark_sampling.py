from pathlib import Path
import pandas as pd
from scripts import benchmark_sampling_methods as bm


def test_benchmark_small(tmp_path):
    sample_sizes = [10, 20]
    df = bm.benchmark_space_filling(sample_sizes, n_trials=3, dim=3, seed=123)
    assert 'min_pairwise_dist' in df.columns
    assert set(df['method'].unique()) == {'lhs', 'sobol'}

    # write outputs to tmp and check files
    out_prefix = tmp_path / 'test_bench'
    df.to_csv(out_prefix.with_suffix('.csv'), index=False)
    assert out_prefix.with_suffix('.csv').exists()

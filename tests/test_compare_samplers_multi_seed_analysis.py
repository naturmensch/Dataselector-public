from pathlib import Path

import pandas as pd

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
mod = load_script(
    ROOT / "scripts" / "compare_samplers_multi_seed.py",
    module_name="scripts.compare_samplers_multi_seed_test",
)


def test_compare_and_analyze(tmp_path):
    # Synthetic per-run results with two samplers and 3 seeds each
    data = [
        {"sampler": "qmc", "seed": 1, "best_value": 70.0, "run_dir": "."},
        {"sampler": "qmc", "seed": 2, "best_value": 71.5, "run_dir": "."},
        {"sampler": "qmc", "seed": 3, "best_value": 69.8, "run_dir": "."},
        {"sampler": "tpe", "seed": 1, "best_value": 72.0, "run_dir": "."},
        {"sampler": "tpe", "seed": 2, "best_value": 73.1, "run_dir": "."},
        {"sampler": "tpe", "seed": 3, "best_value": 71.9, "run_dir": "."},
    ]
    df = pd.DataFrame(data)
    out = tmp_path / "out"
    out.mkdir()

    _ = mod.compare_and_analyze(df, out)

    assert (out / "per_run_results.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "pairwise_stats.csv").exists()
    assert (out / "best_value_boxplot.png").exists()
    # median convergence may not be produced because run_dir '.' doesn't have trials.csv
    # but function should still return summary and stats
    s = pd.read_csv(out / "summary.csv")
    assert "sampler" in s.columns

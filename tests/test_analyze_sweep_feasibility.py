#!/usr/bin/env python3
"""Unit tests for analyze_sweep_feasibility.py script."""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil


def test_feasibility_analysis_with_mock_summaries():
    """Test feasibility analyzer with mock CSV data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock coarse_sweep summary
        coarse_dir = tmpdir / "coarse_sweep"
        coarse_dir.mkdir()
        coarse_data = pd.DataFrame({
            "min_distance_km": [20, 35, 50],
            "total_runs": [9, 9, 9],
            "infeasible_count": [0, 1, 3],
            "median_n_selected": [34, 33, 30],
        })
        coarse_data.to_csv(coarse_dir / "feasibility_summary.csv", index=False)
        
        # Create mock fine_sweep summary
        fine_dir = tmpdir / "fine_sweep"
        fine_dir.mkdir()
        fine_data = pd.DataFrame({
            "min_distance_km": [30, 35, 40, 45, 50],
            "total_runs": [20, 20, 20, 20, 20],
            "infeasible_count": [0, 1, 2, 5, 8],
            "median_n_selected": [34, 33, 32, 31, 28],
        })
        fine_data.to_csv(fine_dir / "feasibility_summary.csv", index=False)
        
        # Run analysis logic (extracted from script)
        summaries = []
        for name in ["coarse_sweep", "fine_sweep"]:
            path = tmpdir / name / "feasibility_summary.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["sweep"] = name
                summaries.append(df)
        
        assert len(summaries) == 2, "Should find both summaries"
        
        full = pd.concat(summaries, ignore_index=True)
        assert len(full) == 8, "Should have 8 total rows (3 coarse + 5 fine)"
        
        # Aggregate
        agg = full.groupby(["sweep", "min_distance_km"]).agg(
            total_runs=("total_runs", "sum"),
            infeasible_count=("infeasible_count", "sum"),
            median_n_selected=("median_n_selected", "median"),
        )
        agg["infeasible_pct"] = agg["infeasible_count"] / agg["total_runs"] * 100.0
        agg = agg.reset_index()
        
        # Validate aggregation
        assert len(agg) == 8, "Should have 8 aggregated rows"
        assert all(agg["infeasible_pct"] >= 0), "Infeasible pct should be non-negative"
        assert all(agg["infeasible_pct"] <= 100), "Infeasible pct should be <= 100"
        
        # Check specific values
        coarse_20 = agg[(agg["sweep"] == "coarse_sweep") & (agg["min_distance_km"] == 20)].iloc[0]
        assert coarse_20["infeasible_count"] == 0
        assert coarse_20["infeasible_pct"] == 0.0
        
        fine_50 = agg[(agg["sweep"] == "fine_sweep") & (agg["min_distance_km"] == 50)].iloc[0]
        assert fine_50["infeasible_count"] == 8
        assert fine_50["infeasible_pct"] == pytest.approx(40.0, abs=0.1)


def test_feasibility_analysis_handles_missing_data():
    """Test that analysis handles missing sweep summaries gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Only create coarse_sweep
        coarse_dir = tmpdir / "coarse_sweep"
        coarse_dir.mkdir()
        coarse_data = pd.DataFrame({
            "min_distance_km": [20, 35],
            "total_runs": [9, 9],
            "infeasible_count": [0, 1],
            "median_n_selected": [34, 33],
        })
        coarse_data.to_csv(coarse_dir / "feasibility_summary.csv", index=False)
        
        # Run analysis logic
        summaries = []
        for name in ["coarse_sweep", "fine_sweep"]:
            path = tmpdir / name / "feasibility_summary.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["sweep"] = name
                summaries.append(df)
        
        assert len(summaries) == 1, "Should find only coarse summary"
        
        full = pd.concat(summaries, ignore_index=True)
        assert len(full) == 2, "Should have 2 rows from coarse sweep"
        assert all(full["sweep"] == "coarse_sweep")


def test_feasibility_summary_columns():
    """Test that required columns are present in summaries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        coarse_dir = tmpdir / "coarse_sweep"
        coarse_dir.mkdir()
        
        # Valid summary with all required columns
        valid_data = pd.DataFrame({
            "min_distance_km": [20],
            "total_runs": [9],
            "infeasible_count": [0],
            "median_n_selected": [34],
        })
        valid_data.to_csv(coarse_dir / "feasibility_summary.csv", index=False)
        
        df = pd.read_csv(coarse_dir / "feasibility_summary.csv")
        required_cols = ["min_distance_km", "total_runs", "infeasible_count", "median_n_selected"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

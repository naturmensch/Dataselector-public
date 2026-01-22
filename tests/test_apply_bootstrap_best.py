#!/usr/bin/env python3
"""Unit tests for apply_bootstrap_best.py script."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import functions from script using importlib to avoid altering module import order
spec = importlib.util.spec_from_file_location(
    "apply_bootstrap_best", ROOT / "scripts" / "apply_bootstrap_best.py"
)
apply_bootstrap_best = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_bootstrap_best)

find_best_bootstrap_candidate = apply_bootstrap_best.find_best_bootstrap_candidate
inject_into_config = apply_bootstrap_best.inject_into_config
write_new_config = apply_bootstrap_best.write_new_config


def test_find_best_bootstrap_candidate():
    """Test finding best bootstrap candidate based on composite score."""
    # Mock bootstrap summary data
    data = pd.DataFrame(
        {
            "alpha": [0.40, 0.50, 0.60],
            "beta": [0.30, 0.25, 0.20],
            "gamma": [0.30, 0.25, 0.20],
            "min_distance_km": [35, 40, 45],
            "temporal_std_mean": [6.5, 6.8, 7.2],
            "temporal_std_std": [0.5, 0.3, 0.8],  # Lower is better (more stable)
            "wwi_percent_mean": [25.0, 23.0, 28.0],  # Lower is better (less WWI bias)
            "wwi_percent_std": [2.0, 1.5, 3.0],
            "jaccard_mean": [0.85, 0.90, 0.75],  # Higher is better (more reproducible)
            "jaccard_std": [0.05, 0.03, 0.08],
        }
    )

    best = find_best_bootstrap_candidate(data)

    # Check that best candidate is returned
    assert "alpha" in best
    assert "beta" in best
    assert "gamma" in best
    assert "min_distance_km" in best
    assert "composite_score" in best

    # Best should be row 1 (index 1) based on scoring
    # - lowest temporal_std_std (0.3)
    # - lowest wwi_percent_mean (23.0)
    # - highest jaccard_mean (0.90)
    assert best["alpha"] == pytest.approx(0.50, abs=0.01)
    assert best["beta"] == pytest.approx(0.25, abs=0.01)
    assert best["min_distance_km"] == pytest.approx(40.0, abs=0.1)


def test_inject_into_config():
    """Test injecting bootstrap-best params into config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock config
        cfg_path = tmpdir / "pipeline_config.yaml"
        cfg = {
            "selection": {
                "n_samples": 34,
                "alpha_visual": 0.70,
                "beta_spatial": 0.05,
                "gamma_temporal": 0.25,
                "min_distance_km": 50.0,
            }
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        # Bootstrap-best params
        params = {
            "alpha": 0.40,
            "beta": 0.30,
            "gamma": 0.30,
            "min_distance_km": 40.0,
            "composite_score": 0.85,
            "temporal_std_mean": 6.5,
            "wwi_percent_mean": 23.0,
            "jaccard_mean": 0.90,
        }

        # Inject
        bak = inject_into_config(cfg_path, params, backup=True)

        # Check backup exists
        assert bak.exists()

        # Check updated config
        with open(cfg_path, "r") as f:
            updated_cfg = yaml.safe_load(f)

        assert updated_cfg["selection"]["alpha_visual"] == pytest.approx(0.40)
        assert updated_cfg["selection"]["beta_spatial"] == pytest.approx(0.30)
        assert updated_cfg["selection"]["gamma_temporal"] == pytest.approx(0.30)
        assert updated_cfg["selection"]["min_distance_km"] == pytest.approx(40.0)
        assert "_bootstrap_provenance" in updated_cfg["selection"]
        assert updated_cfg["selection"]["_bootstrap_provenance"][
            "composite_score"
        ] == pytest.approx(0.85)


def test_write_new_config():
    """Test writing new config with bootstrap-best params."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create base config
        base_cfg_path = tmpdir / "base_config.yaml"
        base_cfg = {
            "selection": {
                "n_samples": 34,
                "alpha_visual": 0.70,
                "beta_spatial": 0.05,
                "gamma_temporal": 0.25,
            }
        }
        with open(base_cfg_path, "w") as f:
            yaml.safe_dump(base_cfg, f)

        # Bootstrap-best params
        params = {
            "alpha": 0.40,
            "beta": 0.30,
            "gamma": 0.30,
            "min_distance_km": 40.0,
            "composite_score": 0.85,
            "temporal_std_mean": 6.5,
            "wwi_percent_mean": 23.0,
            "jaccard_mean": 0.90,
        }

        # Write new config
        out_path = tmpdir / "bootstrap_config.yaml"
        result = write_new_config(out_path, params, base_cfg_path=base_cfg_path)

        assert result == out_path
        assert out_path.exists()

        # Check new config
        with open(out_path, "r") as f:
            new_cfg = yaml.safe_load(f)

        assert new_cfg["selection"]["alpha_visual"] == pytest.approx(0.40)
        assert new_cfg["selection"]["min_distance_km"] == pytest.approx(40.0)
        assert "_bootstrap_provenance" in new_cfg["selection"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

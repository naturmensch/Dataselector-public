"""
Validation tests for min_distance_km computation and data integrity.

This test file validates:
1. No duplicate coordinates in dataset
2. NN distance calculation is correct
3. Median NN distance is statistically sound
4. Tile geometry is reasonable
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def metadata_csv():
    """Load the KDR100 metadata."""
    csv_path = Path(__file__).parents[1] / "data" / "new_all_tiles.csv"
    if not csv_path.exists():
        pytest.skip(f"Metadata CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


class TestDataIntegrity:
    """Validate basic data integrity."""

    def test_no_duplicate_coordinates(self, metadata_csv):
        """Check if any tiles have identical coordinates (duplicates)."""
        coord_cols = ["ul_x", "ul_y"]
        duplicates = metadata_csv[
            metadata_csv.duplicated(subset=coord_cols, keep=False)
        ]

        if len(duplicates) > 0:
            dup_names = duplicates["shortName"].tolist()
            pytest.fail(
                f"Found {len(duplicates)} rows with duplicate coordinates:\n"
                f"{dup_names}\n"
                f"This would cause 0.0 km NN-distances. "
                f"Data needs cleaning (see clean.py)"
            )

    def test_coordinates_are_not_null(self, metadata_csv):
        """Check if coordinates have NaN or infinity values."""
        coord_cols = ["ul_x", "ul_y", "lr_x", "lr_y"]

        for col in coord_cols:
            assert not metadata_csv[col].isna().any(), f"Found NaN in {col}"
            assert np.isfinite(metadata_csv[col]).all(), f"Found inf in {col}"

    def test_coordinate_ranges_are_realistic(self, metadata_csv):
        """Check if EPSG:3857 coordinates are in a realistic Germany range."""
        # Web Mercator (EPSG:3857) ranges in this dataset are expected roughly as:
        # X: 0.6M - 2.6M
        # Y: 5.9M - 7.6M
        # Keep broad sanity bounds to catch corrupt coordinates without overfitting.

        assert metadata_csv["ul_x"].min() > 500_000, "ul_x too far west"
        assert metadata_csv["ul_x"].max() < 3_000_000, "ul_x too far east"
        assert metadata_csv["ul_y"].min() > 5_500_000, "ul_y too far south"
        assert metadata_csv["ul_y"].max() < 8_000_000, "ul_y too far north"

    def test_tiles_have_non_zero_extent(self, metadata_csv):
        """Check if tiles have reasonable size (not points)."""
        width = (metadata_csv["lr_x"] - metadata_csv["ul_x"]).abs()
        height = (metadata_csv["ul_y"] - metadata_csv["lr_y"]).abs()

        assert (width > 0).all(), "Some tiles have zero width"
        assert (height > 0).all(), "Some tiles have zero height"

        # Median tile size should be reasonable (e.g., ~30-100 km)
        median_width_km = width.median() / 1000
        assert (
            20 < median_width_km < 200
        ), f"Tile width unreasonable: {median_width_km:.1f} km"


class TestNearestNeighborDistances:
    """Validate NN distance calculations."""

    def test_nn_distances_are_positive(self, metadata_csv):
        """Check that all NN distances are > 0 (no self-distances or duplicates)."""
        xs = metadata_csv["ul_x"].values
        ys = metadata_csv["ul_y"].values

        nn_distances = []
        zero_pairs = []

        for i in range(len(xs)):
            x1, y1 = xs[i], ys[i]
            distances = []
            for j in range(len(xs)):
                if i != j:
                    x2, y2 = xs[j], ys[j]
                    dist_km = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000.0
                    distances.append(dist_km)
            if distances:
                nn_distance = min(distances)
                nn_distances.append(nn_distance)

                if nn_distance == 0.0:
                    tile_i = metadata_csv.iloc[i]["shortName"]
                    zero_pairs.append((tile_i, nn_distance))

        nn_distances = np.array(nn_distances)

        # Check for 0.0 values
        zero_count = (nn_distances == 0.0).sum()
        if zero_count > 0:
            msg = (
                f"\n{'='*70}\n"
                f"⚠️  FOUND {zero_count} tiles with 0.0 km NN-distance!\n"
                f"{'='*70}\n"
                f"\nExamples:\n"
            )
            for tile, dist in zero_pairs[:5]:
                msg += f"  • {tile}: {dist} km NN-distance\n"

            msg += (
                f"\nThis indicates:\n"
                f"  1. Duplicate coordinates in dataset\n"
                f"  2. Numerical precision issue (u_x/ul_y too similar)\n"
                f"  3. Edge-to-edge instead of centroid measurement\n"
                f"\n⚠️  These 0.0 values will SKEW the median calculation!\n"
                f"    The reported 45.0 km median might be INCORRECT.\n"
                f"\nAction needed:\n"
                f"  • Check: Are these really duplicates or measurement artifacts?\n"
                f"  • If duplicates: Use clean.py to remove them\n"
                f"  • If measurement issue: Recalculate using centroids\n"
                f"{'='*70}\n"
            )
            pytest.fail(msg)

        assert (nn_distances > 0).all(), "All NN-distances must be positive"

    def test_nn_distance_distribution_is_reasonable(self, metadata_csv):
        """Check if NN distance distribution looks reasonable."""
        xs = metadata_csv["ul_x"].values
        ys = metadata_csv["ul_y"].values

        nn_distances = []
        for i in range(len(xs)):
            x1, y1 = xs[i], ys[i]
            distances = []
            for j in range(len(xs)):
                if i != j:
                    x2, y2 = xs[j], ys[j]
                    dist_km = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000.0
                    distances.append(dist_km)
            if distances:
                nn_distances.append(min(distances))

        nn_distances = np.array(nn_distances)
        median_nn = np.median(nn_distances)
        mean_nn = np.mean(nn_distances)

        print(f"\n{'='*70}")
        print("NN-Distance Statistics (centroid-to-centroid):")
        print(f"  Count:    {len(nn_distances)}")
        print(f"  Min:      {np.min(nn_distances):.4f} km")
        print(f"  Q1:       {np.percentile(nn_distances, 25):.4f} km")
        print(f"  Median:   {median_nn:.4f} km")
        print(f"  Q3:       {np.percentile(nn_distances, 75):.4f} km")
        print(f"  Max:      {np.max(nn_distances):.4f} km")
        print(f"  Mean:     {mean_nn:.4f} km")
        print(f"  Std:      {np.std(nn_distances):.4f} km")
        print(f"{'='*70}")

        # Sanity check: median should be >= mean (if there are 0.0 values, mean << median)
        if median_nn < mean_nn - 5:
            print(f"\n⚠️  Median ({median_nn:.1f}) < Mean ({mean_nn:.1f})")
            print("    This is unusual! Check for outliers or 0.0 values.")

        # For grid-like spacing, expect median in reasonable range
        # For KDR100: expect 25-60 km
        assert 10 < median_nn < 100, (
            f"Median NN-distance ({median_nn:.1f} km) outside expected range. "
            f"Expected ~25-60 km for KDR100. Check data!"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

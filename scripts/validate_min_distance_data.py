#!/usr/bin/env python3
"""
Standalone data validation script for min_distance_km calculation.

Directly analyzes the KDR100 dataset to identify:
1. Duplicate coordinates
2. Nearest-neighbor distance distribution
3. Potential data quality issues
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_coordinates(csv_path):
    """Load and analyze coordinate data."""
    print("\n" + "="*70)
    print("KDR100 DATASET VALIDATION")
    print("="*70)
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_file)
    print(f"\n✓ Loaded {len(df)} tiles from {csv_path}")
    
    # ========================================================================
    # 1. CHECK FOR DUPLICATES
    # ========================================================================
    print("\n" + "-"*70)
    print("1. CHECKING FOR DUPLICATE COORDINATES")
    print("-"*70)
    
    duplicates = df[df.duplicated(subset=['ul_x', 'ul_y'], keep=False)].sort_values('ul_x')
    if len(duplicates) > 0:
        print(f"⚠️  FOUND {len(duplicates)} rows with duplicate (ul_x, ul_y):")
        for idx, row in duplicates.iterrows():
            print(f"    {row['shortName']}: ({row['ul_x']:.0f}, {row['ul_y']:.0f})")
        return False
    else:
        print("✓ No duplicate coordinates found")
    
    # ========================================================================
    # 2. CHECK COORDINATE RANGES
    # ========================================================================
    print("\n" + "-"*70)
    print("2. CHECKING COORDINATE RANGES (UTM EPSG:3857)")
    print("-"*70)
    
    print(f"ul_x range: {df['ul_x'].min():.0f} to {df['ul_x'].max():.0f} m")
    print(f"ul_y range: {df['ul_y'].min():.0f} to {df['ul_y'].max():.0f} m")
    print(f"lr_x range: {df['lr_x'].min():.0f} to {df['lr_x'].max():.0f} m")
    print(f"lr_y range: {df['lr_y'].min():.0f} to {df['lr_y'].max():.0f} m")
    
    # Check for reasonable range (these might be Web Mercator or other projection)
    # KDR100 uses EPSG:3857, so we just check that they're in a reasonable range
    min_x = df['ul_x'].min()
    max_x = df['ul_x'].max()
    min_y = df['ul_y'].min()
    max_y = df['ul_y'].max()
    
    # Check that coordinates span a reasonable area (not all in one point)
    x_span = max_x - min_x
    y_span = max_y - min_y
    
    if x_span > 1_000_000 and y_span > 1_000_000:
        print(f"✓ Coordinates span {x_span/1e6:.1f} × {y_span/1e6:.1f} M meters (reasonable for Germany)")
    else:
        print(f"⚠️  Coordinate span too small: {x_span/1e6:.1f} × {y_span/1e6:.1f} M meters")
        return False
    
    # ========================================================================
    # 3. CHECK TILE SIZES
    # ========================================================================
    print("\n" + "-"*70)
    print("3. CHECKING TILE SIZES")
    print("-"*70)
    
    df['width_m'] = (df['lr_x'] - df['ul_x']).abs()
    df['height_m'] = (df['ul_y'] - df['lr_y']).abs()
    
    print(f"Width:  min={df['width_m'].min():.0f} m, median={df['width_m'].median():.0f} m, max={df['width_m'].max():.0f} m")
    print(f"Height: min={df['height_m'].min():.0f} m, median={df['height_m'].median():.0f} m, max={df['height_m'].max():.0f} m")
    print(f"Median tile size: {(df['width_m'].median()/1000):.1f} × {(df['height_m'].median()/1000):.1f} km")
    
    if (df['width_m'] > 0).all() and (df['height_m'] > 0).all():
        print("✓ All tiles have positive extent")
    else:
        print("❌ Some tiles have zero or negative extent")
        return False
    
    # ========================================================================
    # 4. COMPUTE NEAREST-NEIGHBOR DISTANCES
    # ========================================================================
    print("\n" + "-"*70)
    print("4. COMPUTING NEAREST-NEIGHBOR DISTANCES (centroid-to-centroid)")
    print("-"*70)
    
    xs = df["ul_x"].values
    ys = df["ul_y"].values
    n = len(xs)
    
    print(f"Computing {n}² = {n*n:,} pairwise distances...")
    
    nn_distances = []
    zero_distance_tiles = []
    
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n} tiles processed...")
        
        x1, y1 = xs[i], ys[i]
        min_dist = float('inf')
        
        for j in range(n):
            if i != j:
                x2, y2 = xs[j], ys[j]
                dist_meters = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dist_km = dist_meters / 1000.0
                min_dist = min(min_dist, dist_km)
        
        if min_dist < float('inf'):
            nn_distances.append(min_dist)
            if min_dist == 0.0:
                zero_distance_tiles.append((i, df.iloc[i]['shortName'], min_dist))
    
    nn_distances = np.array(nn_distances)
    
    print(f"\n✓ Computed {len(nn_distances)} NN distances")
    
    # ========================================================================
    # 5. ANALYZE NN DISTANCE DISTRIBUTION
    # ========================================================================
    print("\n" + "-"*70)
    print("5. NEAREST-NEIGHBOR DISTANCE STATISTICS")
    print("-"*70)
    
    min_nn = np.min(nn_distances)
    q1_nn = np.percentile(nn_distances, 25)
    median_nn = np.median(nn_distances)
    q3_nn = np.percentile(nn_distances, 75)
    max_nn = np.max(nn_distances)
    mean_nn = np.mean(nn_distances)
    std_nn = np.std(nn_distances)
    
    print(f"Min:      {min_nn:.4f} km")
    print(f"Q1 (25%): {q1_nn:.4f} km")
    print(f"Median:   {median_nn:.4f} km")
    print(f"Q3 (75%): {q3_nn:.4f} km")
    print(f"Max:      {max_nn:.4f} km")
    print(f"Mean:     {mean_nn:.4f} km")
    print(f"Std Dev:  {std_nn:.4f} km")
    
    print(f"\nPercentiles:")
    for pct in [10, 25, 50, 75, 90]:
        val = np.percentile(nn_distances, pct)
        print(f"  {pct}th: {val:.4f} km")
    
    # ========================================================================
    # 6. CHECK FOR PROBLEMS
    # ========================================================================
    print("\n" + "-"*70)
    print("6. DATA QUALITY CHECKS")
    print("-"*70)
    
    issues = []
    
    # Check for 0.0 values
    zero_count = (nn_distances == 0.0).sum()
    if zero_count > 0:
        issues.append(f"⚠️  FOUND {zero_count} tiles with 0.0 km NN-distance")
        if len(zero_distance_tiles) <= 10:
            print(f"\nTiles with 0.0 km NN-distance:")
            for tile_idx, tile_name, dist in zero_distance_tiles:
                print(f"  • {tile_name} (index {tile_idx})")
    else:
        print("✓ No 0.0 km distances (good!)")
    
    # Check for small distances < 10 km
    small_count = (nn_distances < 10.0).sum()
    print(f"✓ Tiles with NN < 10 km: {small_count} ({100*small_count/len(nn_distances):.1f}%)")
    
    # Check median/mean relationship
    if median_nn >= mean_nn - 5:
        print(f"✓ Median ({median_nn:.1f}) ≈ Mean ({mean_nn:.1f}) - good distribution")
    else:
        issues.append(f"⚠️  Median ({median_nn:.1f}) << Mean ({mean_nn:.1f}) - suggests low outliers")
    
    # ========================================================================
    # 7. RECOMMENDATION
    # ========================================================================
    print("\n" + "="*70)
    print("RECOMMENDATION FOR min_distance_km")
    print("="*70)
    
    if not issues:
        min_dist_km = round(median_nn * 2) / 2
        print(f"\n✅ Data is CLEAN. Recommended min_distance_km:")
        print(f"\n   min_distance_km = {min_dist_km} km")
        print(f"   (rounded from median {median_nn:.1f} km)")
        print(f"\nRationale:")
        print(f"  • Median NN-distance is {median_nn:.1f} km")
        print(f"  • This represents 'typical' spacing in dataset")
        print(f"  • Using this as constraint prevents clustering bias")
        return True
    else:
        print(f"\n❌ Data has ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\nBefore using min_distance_km calculation:")
        print(f"  1. Investigate and resolve these issues")
        print(f"  2. Re-run this validation")
        print(f"  3. Then compute final min_distance_km")
        return False


if __name__ == "__main__":
    csv_path = "data/new_all_tiles.csv"
    success = analyze_coordinates(csv_path)
    sys.exit(0 if success else 1)

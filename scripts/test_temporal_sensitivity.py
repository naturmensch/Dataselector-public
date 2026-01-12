"""
Temporal Weight Sensitivity Test mit Constraint-integrierter Methode

Prüft ob temporal_weight jetzt messbare Effekte hat.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector

OUT = Path('outputs')
features = np.load(OUT / 'features.npy')
metadata = pd.read_csv(OUT / 'metadata.csv')

print("=" * 80)
print("TEMPORAL WEIGHT SENSITIVITY TEST (Constraint-Integrated)")
print("=" * 80)

# Test both subset and full
for name, n in [('Subset N=50', 50), ('Full N=673', None)]:
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print('=' * 80)
    
    if n:
        feat = features[:n]
        meta = metadata.iloc[:n].reset_index(drop=True)
    else:
        feat = features
        meta = metadata
    
    # Clustering
    cl = ClusteringPipeline(n_clusters=8)
    emb, labels = cl.fit_transform(feat)
    
    # Test various temporal weights
    temporal_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    results = []
    
    print(f"\nTesting temporal_weights: {temporal_weights}")
    print(f"Fixed: min_distance_km=40.0\n")
    
    for tw in temporal_weights:
        selector = DiversitySelector(
            n_samples=40,
            use_constraint_integration=True
        )
        
        selected = selector.select(
            feat, meta,
            temporal_weight=tw,
            spatial_constraint=True,
            min_distance_km=40.0
        )
        
        # Metrics
        years = meta.iloc[selected]['year'].dropna().values
        temporal_std = float(np.std(years)) if len(years) > 0 else np.nan
        temporal_range = float(np.max(years) - np.min(years)) if len(years) > 0 else np.nan
        
        # Spatial
        from src.spatial_facility_location import haversine_distance
        pairwise = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                r1 = meta.iloc[selected[i]]
                r2 = meta.iloc[selected[j]]
                pairwise.append(haversine_distance(r1['N'], r1['left'], r2['N'], r2['left']))
        mean_pairwise = float(np.mean(pairwise)) if pairwise else np.nan
        
        print(f"tw={tw:.1f}: n={len(selected):2d}, temporal_std={temporal_std:5.2f}, "
              f"temporal_range={temporal_range:5.1f}, mean_dist={mean_pairwise:6.1f}km")
        
        results.append({
            'dataset': name,
            'temporal_weight': tw,
            'n_selected': len(selected),
            'temporal_std': temporal_std,
            'temporal_range': temporal_range,
            'mean_pairwise_km': mean_pairwise
        })
    
    # Analysis
    df = pd.DataFrame(results)
    print(f"\nCorrelation temporal_weight ↔ temporal_std: r={df['temporal_weight'].corr(df['temporal_std']):.3f}")
    print(f"Temporal STD range: {df['temporal_std'].min():.2f} - {df['temporal_std'].max():.2f} (Δ={df['temporal_std'].max()-df['temporal_std'].min():.2f})")

print(f"\n{'=' * 80}")
print("CONCLUSION")
print('=' * 80)
print("Wenn temporal_weight jetzt Effekte zeigt (r ≠ 0, signifikante Δ),")
print("dann ist die Constraint-Integration erfolgreich!")

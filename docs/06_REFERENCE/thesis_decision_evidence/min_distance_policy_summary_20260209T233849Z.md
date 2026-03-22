# Min Distance Policy Comparison

- UTC timestamp: `20260209T233849Z`
- Metadata: `<dataselector-repo>/data/new_all_tiles.csv`
- Rows: `676`
- n_samples: `34`
- seeds: `[42, 43, 44, 45, 46]`
- distances: `[28.5, 40.0, 45.0]`
- weights: alpha=0.4, beta=0.3, gamma=0.3

## Summary

```text
 distance  runs  target_met_rate  mean_n_selected  std_n_selected  mean_shortfall  shortfall_rate  mean_clusters_covered  mean_temporal_std  mean_wwi_percent  mean_spatial_mean_km  mean_spatial_min_km  stability_jaccard_mean  stability_jaccard_min  decision_score  recommended_distance                                                                                                      recommendation_rationale
     28.5     5              1.0             34.0             0.0             0.0             0.0                    8.0          10.175159         29.411765            514.687130            42.689172                     1.0                    1.0           114.0                  28.5 feasibility/stability/coverage rule with near-tie preference for smaller distance (higher downstream combination flexibility)
     40.0     5              1.0             34.0             0.0             0.0             0.0                    8.0          10.175159         29.411765            514.687130            42.689172                     1.0                    1.0           114.0                  28.5 feasibility/stability/coverage rule with near-tie preference for smaller distance (higher downstream combination flexibility)
     45.0     5              1.0             34.0             0.0             0.0             0.0                    7.4          10.144588         26.470588            508.659294            59.039742                     1.0                    1.0           113.7                  28.5 feasibility/stability/coverage rule with near-tie preference for smaller distance (higher downstream combination flexibility)
```


## Recommendation

- Recommended `min_distance_km`: **28.5**
- Rationale: feasibility/stability/coverage rule with near-tie preference for smaller distance (higher downstream combination flexibility)
- Rule applied: prefer low-shortfall and stable candidates; in near-ties choose smaller distance.

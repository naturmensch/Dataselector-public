# Algorithm Overview

Short overview of the Multi-Criteria Facility Location algorithm used:

- Objective: Maximize coverage over visual, spatial and temporal distances
- Use submodular greedy facility location (apricot-select)
- Combine distances: d = α·d_visual + β·d_spatial + γ·d_temporal
- Enforce spatial hard-constraint via `min_distance_km` with fallback logic

(Expand with pseudocode and links to `src/lazy_facility_location.py`.)
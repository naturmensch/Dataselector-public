import numpy as np
import pandas as pd

from src.diversity_selector import DiversitySelector
from src.io import load_metadata, load_or_extract_features
from src.spatial_facility_location import haversine_distance


def test_preselection_includes_seed(tmp_path):
    meta = load_metadata("data/new_all_tiles.csv")
    features = load_or_extract_features(tmp_path, csv_meta=str("data/new_all_tiles.csv"), cache=True)

    # find Hamburg by longName
    mask = meta["longName"].str.contains("Hamburg", case=False)
    assert mask.any(), "Hamburg not found in metadata for test"
    seed_pos = int(mask[mask].index[0])

    ds = DiversitySelector(n_samples=5, use_multi_criteria=True, random_state=42)
    selected = ds.select(
        features=features,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.05,
        gamma_temporal=0.25,
        spatial_constraint=True,
        min_distance_km=25.0,
        pre_selected=[seed_pos],
    )

    assert seed_pos in selected, "Pre-selected seed not included in selection"
    assert len(selected) == 5


def test_preselection_respects_min_distance(tmp_path):
    meta = load_metadata("data/new_all_tiles.csv")
    features = load_or_extract_features(tmp_path, csv_meta=str("data/new_all_tiles.csv"), cache=True)

    mask = meta["longName"].str.contains("Hamburg", case=False)
    seed_pos = int(mask[mask].index[0])

    ds = DiversitySelector(n_samples=6, use_multi_criteria=True, random_state=42)
    selected = ds.select(
        features=features,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.05,
        gamma_temporal=0.25,
        spatial_constraint=True,
        min_distance_km=50.0,
        pre_selected=[seed_pos],
    )

    # Ensure no other selection is within 50km of seed
    seed_lat = meta.iloc[seed_pos]["N"]
    seed_lon = meta.iloc[seed_pos]["left"]
    for idx in selected:
        if idx == seed_pos:
            continue
        lat = meta.iloc[idx]["N"]
        lon = meta.iloc[idx]["left"]
        d = haversine_distance(seed_lat, seed_lon, lat, lon)
        assert d >= 50.0, f"Selected index {idx} is within min_distance of seed ({d:.1f} km)"
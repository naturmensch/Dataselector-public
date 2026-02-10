import pytest

import dataselector.data.io
from dataselector.selection.diversity_selector import DiversitySelector


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "includes_seed",
            "n_samples": 1,
            "min_distance_km": 0.0,
            "check_seed": True,
            "spatial_constraint": False,
        },
    ],
)
def test_preselection(tmp_path, test_case, stub_feature_extraction):
    meta = dataselector.data.io.load_metadata("data/new_all_tiles.csv")
    features = dataselector.data.io.load_or_extract_features(
        tmp_path, csv_meta=str("data/new_all_tiles.csv"), cache=True
    )

    # find KDR_001.png by longName
    mask = meta["longName"].str.contains("KDR_001", case=False)
    assert mask.any(), "KDR_001 not found in metadata for test"
    seed_pos = int(mask[mask].index[0])

    ds = DiversitySelector(
        n_samples=test_case["n_samples"], use_multi_criteria=True, random_state=42
    )
    selected = ds.select(
        features=features,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.05,
        gamma_temporal=0.25,
        spatial_constraint=test_case.get("spatial_constraint", True),
        min_distance_km=test_case["min_distance_km"],
        pre_selected=[seed_pos],
    )

    if test_case["check_seed"]:
        assert seed_pos in selected, "Pre-selected seed not included in selection"
    assert len(selected) == test_case["n_samples"]

    # Spatial distance checks removed from this test; see `tests/test_spatial_logic.py` for comprehensive spatial constraint validation.


def test_preselection_hamburg_alias_resolves_to_kdr146(
    tmp_path, stub_feature_extraction
):
    """Hamburg shortcut must resolve to the documented anchor tile KDR_146."""
    meta = dataselector.data.io.load_metadata("data/new_all_tiles.csv")
    features = dataselector.data.io.load_or_extract_features(
        tmp_path, csv_meta=str("data/new_all_tiles.csv"), cache=True
    )

    mask = meta["shortName"].astype(str).str.upper() == "KDR_146"
    assert mask.any(), "KDR_146 not found in metadata for Hamburg alias test"
    anchor_idx = int(mask[mask].index[0])

    ds = DiversitySelector(n_samples=1, use_multi_criteria=True, random_state=42)
    selected = ds.select(
        features=features,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.05,
        gamma_temporal=0.25,
        spatial_constraint=False,
        min_distance_km=0.0,
        pre_selected_names=["Hamburg"],
    )

    assert anchor_idx in selected, "Hamburg alias did not include KDR_146"


def test_preselection_city_match_for_kiel(tmp_path, stub_feature_extraction):
    """City-name preselection should resolve via metadata.city when populated."""
    meta = dataselector.data.io.load_metadata("data/new_all_tiles.csv")
    features = dataselector.data.io.load_or_extract_features(
        tmp_path, csv_meta=str("data/new_all_tiles.csv"), cache=True
    )

    mask = meta["city"].astype(str).str.lower() == "kiel"
    assert mask.any(), "Kiel not found in metadata.city for city-match test"
    kiel_idx = int(mask[mask].index[0])

    ds = DiversitySelector(n_samples=1, use_multi_criteria=True, random_state=42)
    selected = ds.select(
        features=features,
        metadata=meta,
        alpha_visual=0.7,
        beta_spatial=0.05,
        gamma_temporal=0.25,
        spatial_constraint=False,
        min_distance_km=0.0,
        pre_selected_names=["Kiel"],
    )

    assert kiel_idx in selected, "City preselection 'Kiel' did not resolve correctly"

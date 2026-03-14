from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataselector.data.tile_policy import apply_tile_exclusion_policy


def test_tile_exclusion_policy_excludes_kdr_155b(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "shortName": ["KDR_155", "KDR_155b", "KDR_200"],
            "longName": ["a", "b", "c"],
        }
    )
    policy = {
        "rules": [
            {
                "id": "exclude_kdr_155b",
                "action": "exclude_from_candidate_pool",
                "match": {"shortName": ["KDR_155b"]},
            }
        ]
    }
    policy_path = tmp_path / "tile_policy.yaml"
    policy_path.write_text("rules: []\n", encoding="utf-8")

    result = apply_tile_exclusion_policy(df, policy=policy, policy_path=policy_path)
    assert result.excluded_count == 1
    assert "KDR_155b" in result.excluded_shortnames
    assert list(result.dataframe["shortName"]) == ["KDR_155", "KDR_200"]


def test_tile_exclusion_policy_flags_temporal_outliers_by_year_without_excluding(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "shortName": ["KDR_039", "KDR_521", "KDR_777", "KDR_200"],
            "longName": ["a", "b", "c", "d"],
            "city": ["X", "Y", "Q", "Z"],
            "year": [1980, 1985, 1860, 1910],
        }
    )
    policy = {
        "constants": {
            "kdr_core_publication_frame": {
                "year_min": 1878,
                "year_max": 1945,
            }
        },
        "rules": [
            {
                "id": "flag_temporal_outliers",
                "class": "temporal_scope_outlier",
                "action": "flag_for_reporting",
                "rationale": "Retain, but report critically.",
                "match": {"year_gt_ref": "kdr_core_publication_frame.year_max"},
            },
            {
                "id": "flag_pre_series_temporal_outliers",
                "class": "temporal_scope_outlier",
                "action": "flag_for_reporting",
                "rationale": "Retain, but report critically.",
                "match": {"year_lt_ref": "kdr_core_publication_frame.year_min"},
            },
        ],
    }
    policy_path = tmp_path / "tile_policy.yaml"
    policy_path.write_text("rules: []\n", encoding="utf-8")

    result = apply_tile_exclusion_policy(df, policy=policy, policy_path=policy_path)
    assert result.excluded_count == 0
    assert list(result.dataframe["shortName"]) == [
        "KDR_039",
        "KDR_521",
        "KDR_777",
        "KDR_200",
    ]
    assert result.flagged_count == 3
    assert result.flagged_shortnames == ["KDR_039", "KDR_521", "KDR_777"]
    assert result.flagged_classes == ["temporal_scope_outlier"]
    assert {entry["shortName"] for entry in result.flagged_caveats} == {
        "KDR_039",
        "KDR_521",
        "KDR_777",
    }

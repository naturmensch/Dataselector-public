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

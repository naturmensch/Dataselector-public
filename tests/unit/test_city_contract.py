from pathlib import Path

import pandas as pd


def test_canonical_city_contract_complete():
    """Canonical metadata must have fully resolved city + city_source."""
    csv_path = Path("data/new_all_tiles.csv")
    assert csv_path.exists(), "Canonical metadata CSV missing"

    df = pd.read_csv(csv_path)
    assert len(df) > 0, "Canonical metadata must not be empty"
    assert "city" in df.columns, "Canonical metadata missing 'city' column"
    assert (
        "city_source" in df.columns
    ), "Canonical metadata missing 'city_source' column"

    city_missing = int((df["city"].fillna("").astype(str).str.strip() == "").sum())
    source_missing = int(
        (df["city_source"].fillna("").astype(str).str.strip() == "").sum()
    )

    assert (
        city_missing == 0
    ), f"Canonical city contract violated: {city_missing} rows unresolved"
    assert (
        source_missing == 0
    ), f"Canonical city_source contract violated: {source_missing} rows unresolved"


def test_canonical_city_contract_excludes_backup_fill():
    """Canonical metadata must not depend on historical backup-fill provenance."""
    csv_path = Path("data/new_all_tiles.csv")
    assert csv_path.exists(), "Canonical metadata CSV missing"

    df = pd.read_csv(csv_path)
    sources = set(df["city_source"].fillna("").astype(str).str.strip())
    assert (
        "backup_fill" not in sources
    ), "Canonical city_source contract still exposes deprecated backup_fill"

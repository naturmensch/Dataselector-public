from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from dataselector.data.tiles import TileSet


def load_tiles(
    csv: str | Path,
    image_dir: str | Path = "data/images",
    *,
    prefer_shortname: bool = True,
    fill_missing_image_paths: bool = True,
    missing_placeholder: str = "missing_placeholder.png",
) -> TileSet:
    """Load and normalize tile metadata and resolve image paths.

    This is the new canonical entrypoint for metadata loading.

    It intentionally reuses the existing, battle-tested implementation in
    `src.metadata_processor.MetadataProcessor`.
    """

    from dataselector.data.metadata_processor import MetadataProcessor

    csv_path = Path(csv)
    img_dir = Path(image_dir)

    mp = MetadataProcessor(str(csv_path))
    df = mp.load_csv()
    df = mp.add_temporal_metadata()
    df = mp.resolve_image_paths(img_dir, prefer_shortname=prefer_shortname)

    if fill_missing_image_paths and "image_path" in df.columns:
        df["image_path"] = df["image_path"].fillna(missing_placeholder)

    prov = {
        "metadata_csv": str(csv_path),
        "image_dir": str(img_dir),
        "prefer_shortname": bool(prefer_shortname),
    }

    return TileSet(
        df=df,
        metadata_csv=csv_path,
        image_dir=img_dir,
        missing_placeholder=missing_placeholder,
        provenance=prov,
    )


def load_metadata_df(
    csv: str | Path,
    image_dir: str | Path = "data/images",
    *,
    prefer_shortname: bool = True,
) -> pd.DataFrame:
    """Backwards-compatible convenience wrapper returning only a DataFrame."""

    return load_tiles(csv, image_dir=image_dir, prefer_shortname=prefer_shortname).df

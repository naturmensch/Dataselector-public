from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from dataselector.data.tiles import TileSet


def load_or_compute_features(
    tiles: TileSet,
    *,
    out_dir: str | Path = "outputs",
    batch_size: int = 16,
    cache: bool = True,
) -> np.ndarray:
    """Return features aligned with `tiles.df`.

    Today this is a thin wrapper over the existing cache implementation in
    `src.io.load_or_extract_features`, but the signature is intentionally
    library-friendly: it accepts a `TileSet` instead of ad-hoc paths.
    """

    from src.io import load_or_extract_features

    out_dir = Path(out_dir)
    csv_meta: Optional[str]
    if tiles.metadata_csv is not None:
        csv_meta = str(tiles.metadata_csv)
    else:
        csv_meta = None

    return load_or_extract_features(
        out_dir=out_dir,
        csv_meta=csv_meta,
        batch_size=int(batch_size),
        cache=bool(cache),
    )

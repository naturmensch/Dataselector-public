from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class TileSet:
    """Container for tile metadata and associated paths.

    This is intentionally lightweight: most code in this repo historically
    passed around a pandas DataFrame. `TileSet` keeps that flexibility while
    providing a single place for:
    - provenance (where did metadata come from?)
    - image directory context
    - consistent downstream expectations (required columns)
    """

    df: pd.DataFrame
    metadata_csv: Optional[Path] = None
    image_dir: Optional[Path] = None
    missing_placeholder: str = "missing_placeholder.png"
    provenance: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:  # pragma: no cover (trivial)
        return int(len(self.df))

    @property
    def has_resolved_images(self) -> bool:
        return "image_path" in self.df.columns

    def require_columns(self, cols: list[str]) -> None:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"TileSet missing required columns: {missing}")

"""Shared script utilities.
Provides a canonical DATA_DIR that respects the DATA_DIR environment variable.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# DATA_DIR can be overridden with the DATA_DIR environment variable for tests
DATA_DIR = Path(os.environ.get("DATA_DIR", str(ROOT / "data")))


def data_path(*parts):
    """Return a Path under DATA_DIR."""
    return DATA_DIR.joinpath(*parts)

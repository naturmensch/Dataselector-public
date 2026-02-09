from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataselector.data import io as io_mod
from dataselector.workflows import autoscale as autoscale_mod

pytestmark = pytest.mark.canonical_source_contract


def test_io_rejects_implicit_outputs_metadata_fallback(tmp_path, monkeypatch):
    """`csv_meta=None` must not silently fall back to outputs/metadata.csv."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.csv").write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"data/new_all_tiles\.csv"):
        io_mod.load_or_extract_features(out_dir=out, csv_meta=None, cache=False)


def test_io_uses_canonical_metadata_when_csv_meta_is_none(tmp_path, monkeypatch):
    """When canonical file exists, default resolution should use it."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical = data_dir / "new_all_tiles.csv"
    canonical.write_text("id\n1\n2\n", encoding="utf-8")

    seen = {}

    def _fake_load_metadata(path):
        seen["path"] = Path(path).resolve()
        return pd.DataFrame({"image_path": ["a.png", "b.png"]})

    monkeypatch.setattr(io_mod, "load_metadata", _fake_load_metadata)
    monkeypatch.setattr(
        io_mod,
        "extract_features",
        lambda metadata, batch_size=16: np.zeros((len(metadata), 4), dtype=np.float32),
    )

    feats = io_mod.load_or_extract_features(
        out_dir=tmp_path / "outputs",
        csv_meta=None,
        batch_size=4,
        cache=False,
    )
    assert feats.shape == (2, 4)
    assert seen["path"] == canonical.resolve()


def test_io_enforce_canonical_rejects_noncanonical_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Only 'data/new_all_tiles\.csv' is allowed"):
        io_mod.load_or_extract_features(
            out_dir=tmp_path / "outputs",
            csv_meta=str(custom_csv),
            cache=False,
            enforce_canonical=True,
        )


def test_autoscale_require_metadata_fails_without_canonical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"data/new_all_tiles\.csv"):
        autoscale_mod.load_or_create_data(
            out_dir=tmp_path / "outputs",
            n=10,
            dim=4,
            seed=42,
            require_metadata=True,
        )


def test_autoscale_require_metadata_uses_canonical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    canonical = data_dir / "new_all_tiles.csv"
    canonical.write_text("id\n1\n2\n3\n", encoding="utf-8")

    seen = {}

    def _fake_loader(out_dir, csv_meta, batch_size=16, cache=True, **kwargs):
        seen["csv_meta"] = Path(csv_meta).resolve()
        return np.zeros((3, 2), dtype=np.float32)

    def _fake_meta(csv_meta):
        seen["meta"] = Path(csv_meta).resolve()
        return pd.DataFrame(
            {
                "ul_x": [0.0, 1.0, 2.0],
                "ul_y": [3.0, 4.0, 5.0],
                "lr_x": [0.5, 1.5, 2.5],
                "lr_y": [2.5, 3.5, 4.5],
                "year": [1900, 1901, 1902],
            }
        )

    monkeypatch.setattr(io_mod, "load_or_extract_features", _fake_loader)
    monkeypatch.setattr(io_mod, "load_metadata", _fake_meta)

    feats, meta = autoscale_mod.load_or_create_data(
        out_dir=tmp_path / "outputs",
        n=10,
        dim=4,
        seed=42,
        require_metadata=True,
    )
    assert feats.shape == (3, 2)
    assert len(meta) == 3
    assert seen["csv_meta"] == canonical.resolve()
    assert seen["meta"] == canonical.resolve()

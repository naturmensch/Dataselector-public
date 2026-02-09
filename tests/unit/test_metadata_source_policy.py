from __future__ import annotations

import pytest

from dataselector.data.metadata_source import (
    CANONICAL_METADATA_RELATIVE_PATH,
    assert_canonical_metadata,
    canonical_metadata_path,
)

pytestmark = pytest.mark.canonical_source_contract


def test_canonical_metadata_path_uses_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert canonical_metadata_path() == (tmp_path / CANONICAL_METADATA_RELATIVE_PATH)


def test_assert_canonical_metadata_accepts_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert assert_canonical_metadata(None, context="test") == (
        tmp_path / CANONICAL_METADATA_RELATIVE_PATH
    )


def test_assert_canonical_metadata_accepts_equivalent_relative_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    assert assert_canonical_metadata("data/new_all_tiles.csv", context="test") == (
        tmp_path / CANONICAL_METADATA_RELATIVE_PATH
    )


def test_assert_canonical_metadata_rejects_noncanonical_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Only 'data/new_all_tiles.csv' is allowed"):
        assert_canonical_metadata("outputs/metadata.csv", context="test")

from __future__ import annotations

import pytest

from dataselector.workflows.model_compare_contract import (
    assert_same_split_manifest_sha256,
)


def test_same_manifest_contract_accepts_identical_hashes() -> None:
    payloads = [
        {"extra": {"split_manifest_sha256": "abc"}},
        {"extra": {"split_manifest_sha256": "abc"}},
        {"extra": {"split_manifest_sha256": "abc"}},
    ]
    assert assert_same_split_manifest_sha256(payloads) == "abc"


def test_same_manifest_contract_rejects_mismatch() -> None:
    payloads = [
        {"extra": {"split_manifest_sha256": "abc"}},
        {"extra": {"split_manifest_sha256": "def"}},
    ]
    with pytest.raises(ValueError, match="identical split_manifest_sha256"):
        assert_same_split_manifest_sha256(payloads)

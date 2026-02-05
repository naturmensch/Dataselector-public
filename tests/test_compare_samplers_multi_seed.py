import pytest


def test_compare_samplers_multi_seed_deprecated():
    """Deprecated: scripts/compare_samplers_multi_seed.py has been migrated.

    See tests/test_compare_samplers.py for workflow-based tests.
    """
    pytest.skip("compare_samplers_multi_seed script deprecated; workflow tests cover functionality")

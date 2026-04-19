# tests/archive

This directory holds **historical or archived tests**.

These files are retained for traceability or narrow compatibility/debugging
needs. They are **not** part of the active showcase-release confidence set.

For the active test strategy, start with:

- `docs/08_GOVERNANCE/TEST_SUITE_CURATION.md`
- `tests/test_thesis_pipeline.py`
- `tests/unit/`

Guideline:

- new active contract or governance tests should not be added here
- if a historical test becomes relevant again, move or rewrite it into the
  active suite instead of silently promoting this archive directory

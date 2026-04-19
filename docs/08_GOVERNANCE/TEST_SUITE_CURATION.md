# Test Suite Curation (Active)

This document describes the **current active test strategy** for the
Dataselector showcase-release state. It supersedes older one-off consolidation
planning notes and focuses on the thesis-v2 architecture that is currently
authoritative.

## 1. Purpose

The test suite is curated around the active thesis path:

1. protect the scientific freeze contract,
2. protect runtime/package architecture and thin-wrapper boundaries,
3. protect policy/documentation consistency,
4. keep long-running checks explicit instead of hiding them inside noisy
   default test runs.

## 2. Authoritative coverage map

| Active subsystem | Authoritative tests |
|---|---|
| Thesis pipeline contract | `tests/test_thesis_pipeline.py` |
| Thesis orchestrator | `tests/unit/test_thesis_orchestrate.py` |
| CRS strict thesis mode | `tests/unit/test_crs_strict_thesis_mode.py` |
| Cache identity / provenance | `tests/unit/test_feature_cache_identity.py`, `tests/test_cache_hash.py` |
| Handoff bundle and wrapper contract | `tests/unit/test_handoff_bundle.py`, `tests/unit/test_handoff_check_script.py` |
| Reporting and diagnostics | `tests/test_generate_reports_diagnostics.py` |
| Policy/documentation governance | `tests/unit/test_config_policy_docs.py`, `tests/unit/test_authoritative_docs_consistency.py`, `tests/unit/test_no_legacy_script_references.py` |

These files define the minimum active confidence set for the canonical thesis
workflow.

## 3. Directory and tier map

### `tests/unit/`

Primary home for active governance, contract, runtime, and wrapper-boundary
tests. New thesis-v2 contract tests should prefer `tests/unit/` unless they are
true end-to-end or pipeline gates.

### Root-level `tests/test_*.py`

Mixed zone for:

- the canonical long pipeline gate (`tests/test_thesis_pipeline.py`)
- retained workflow-level smoke and integration coverage
- selected **secondary active** tests that still protect live boundaries

This area should become gradually clearer over time, but not through churn that
throws away useful coverage.

### `tests/integration/`

Opt-in integration-heavy checks. Keep for dependency-sensitive or multi-step
paths that are still live, but do not treat them as the default fast local
story.

### `tests/e2e/`

Long-running and explicit end-to-end validation. These remain useful for real
runtime confidence, but they are not the first-line active suite for everyday
iteration.

### `tests/archive/`

Historical or compatibility-focused tests kept for traceability. They are not
part of the active showcase contract and must not define the default repo
narrative.

## 4. Test tiers

### Fast governance / docs

Use for quick local iteration:

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m pytest -q tests/unit/test_config_policy_docs.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_authoritative_docs_consistency.py
micromamba run -n dataselector python -m pytest -q tests/unit/test_no_legacy_script_references.py
```

### Fast unit / contract checks

Focused contract checks for current thesis-v2 behavior:

- orchestrator plumbing
- CRS strict mode
- cache identity and provenance
- handoff bundle and wrapper resolution
- parameter, source-of-truth, and policy invariants

### Targeted integration

Integration-heavy checks stay opt-in and are documented in
[`tests/INTEGRATION_TESTS.md`](../../tests/INTEGRATION_TESTS.md).

### Long manual release checks

These are intentionally treated as manual release checks because they are
slower and more token-expensive:

```bash
micromamba run -n dataselector python -m pytest -q tests/test_thesis_pipeline.py
make format-check
make test
```

### Historical / compatibility

Historical or compatibility-oriented tests may remain when they still protect a
live boundary, but they must no longer dominate the default story. Pure
skip-only merge artifacts are not part of the active suite.

Examples of retained **secondary active** tests include optional workflow or
tooling coverage such as:

- Optuna-related smoke checks
- selected compare/audit helpers
- archive-tool coverage that still protects live tool behavior
- thin compatibility tests that still guard a real boundary

### Expected release skips

The active release run may still contain **explicit opt-in skips** for:

- full E2E profiles gated by `RUN_FULL_INTEGRATION=1`
- real-image profiles gated by `DATASELECTOR_IMAGE_DIR`

These are acceptable because they reflect intentional execution profiles, not
dead coverage.

By contrast, unconditional placeholder skips such as `skipif(True)`,
module-level `pytestmark = skip(...)`, or empty `pass` tests marked with
`@pytest.mark.skip(...)` do **not** belong in the active suite.

## 5. Current curation decisions

1. The active thesis-v2 pipeline and packaging path are the primary coverage
   target.
2. Pure `Phase 2 merge artifact` skip-only files were removed because they no
   longer exercised any live contract.
3. Older one-off consolidation planning docs remain useful as historical
   planning context, but are not authoritative for the current suite.
4. Root-level tests may remain mixed where they still protect live workflows,
   but new active contract tests should not be added there by default.
5. Historical tests belong in archive zones, not in the active confidence set.
6. The active suite should not carry unconditional skip stubs; either convert
   them into real active checks or remove/demote them.

## 6. Reader guidance

If you want to understand whether the current thesis architecture is protected,
start with:

1. `tests/test_thesis_pipeline.py`
2. `tests/unit/test_thesis_orchestrate.py`
3. `tests/unit/test_crs_strict_thesis_mode.py`
4. `tests/unit/test_handoff_bundle.py`
5. `tests/test_generate_reports_diagnostics.py`
6. the doc-governance tests

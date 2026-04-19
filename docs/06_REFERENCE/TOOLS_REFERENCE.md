# Administrative Tools Reference (Active)

This page documents the active utility and governance commands that are exposed
via `python -m dataselector`.

Canonical invocation:

```bash
micromamba run -n dataselector python -m dataselector <command> [args]
```

For the complete command list, see
[CLI_COMMAND_CATALOG.md](CLI_COMMAND_CATALOG.md).

## Environment and Runtime Validation

### `check-geo`
Purpose: validate critical geospatial dependencies and report versions.

### `check-env`
Purpose: enforce environment usage policy for canonical runtime execution.

### `check-runtime-readiness`
Purpose: verify that runtime prerequisites for thesis commands are satisfied.

## Script and Repository Governance

### `check-script-wrappers`
Purpose: ensure scripts act as wrappers/orchestrators and do not duplicate
scientific core logic from `dataselector/`.

### `check-protected`
Purpose: detect modifications under protected repository paths.

### `verify-archive`
Purpose: validate archive integrity and reference hygiene.

## Documentation Governance

### `docs-link-check`
Purpose: validate relative links in active documentation.

Recommended:

```bash
micromamba run -n dataselector python -m dataselector docs-link-check
```

Optional (broader scan including historical docs):

```bash
micromamba run -n dataselector python -m dataselector docs-link-check --include-historical
```

### `docs-link-autofix`
Purpose: attempt automatic fixes for broken relative links.

Dry-run by default:

```bash
micromamba run -n dataselector python -m dataselector docs-link-autofix
```

Apply changes:

```bash
micromamba run -n dataselector python -m dataselector docs-link-autofix --yes
```

## Data and Artifact Utilities

### `align-audit`
Purpose: audit alignment between metadata and raster surfaces.

### `archive-outputs`
Purpose: archive selected output artifacts.

### `list-archives`
Purpose: enumerate available archives.

### `clean-workspace`
Purpose: remove regenerable workspace artifacts.

## Practical Validation Sequence

Run a concise governance check sequence:

```bash
micromamba run -n dataselector python -m dataselector check-runtime-readiness
micromamba run -n dataselector python -m dataselector check-script-wrappers --strict
micromamba run -n dataselector python -m dataselector docs-link-check
```

## Notes

1. Command names and registration are authoritative in code under
   `dataselector/tools/` and related workflow modules.
2. Prefer this page plus [CLI_COMMAND_CATALOG.md](CLI_COMMAND_CATALOG.md)
   over historical or archived references.

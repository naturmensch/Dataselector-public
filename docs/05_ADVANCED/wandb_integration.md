# wandb Integration (Advanced)

## Scope

This guide describes the current wandb integration points in the
`dataselector` package.

## Runtime

Use micromamba-canonical invocation:

```bash
micromamba run -n dataselector \
  python -m dataselector thesis-pipeline
```

## Code Location

wandb logging utilities are provided by:

1. `dataselector/analysis/wandb_logger.py`

Example import:

```python
from dataselector.analysis.wandb_logger import WandBLogger
```

## Enable / Disable

1. Enable by providing wandb credentials in the environment.
2. Disable for local/offline runs by using the logger's disabled mode or
   `WANDB_DISABLED=true`.

## Notes

1. This guide intentionally avoids legacy `src/*` paths.
2. Scripts should call CLI/workflow paths rather than duplicating logging logic.

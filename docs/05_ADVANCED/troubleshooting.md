# Troubleshooting & FAQ

Common errors & fixes:

- "wandb not installed": `pip install wandb`
- Missing features: run `python -c "from src.io import load_or_extract_features; ..."`
- Monitor resume failures: check `outputs/` and Optuna DB integrity (`PRAGMA integrity_check`)

If you cannot fix an issue, open an issue describing steps to reproduce and include `pipeline.log` and relevant `outputs/` files.
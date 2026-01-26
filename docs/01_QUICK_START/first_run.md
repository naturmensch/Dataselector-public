# First Run — Schritt-für-Schritt

1. Activate env:
```bash
conda activate dataselector
```
2. Run a small LHS exploration:
```bash
./scripts/exec_in_env.sh --env dataselector -- python scripts/tune_weights_and_run.py --n-samples 20
```
3. Check `outputs/tuning_weights/pareto/pareto_solutions.csv` and plots.

If a step fails, consult `docs/05_ADVANCED/troubleshooting.md` and open an issue with `outputs/` and `pipeline.log`.
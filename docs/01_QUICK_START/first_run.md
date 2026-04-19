# First Run — Schritt-für-Schritt

1. Activate env:
```bash
micromamba create -f environment.yml -n dataselector
micromamba run -n dataselector python -V
```
2. Run a small exploration with the current CLI:
```bash
micromamba run -n dataselector python -m dataselector adaptive-auto \
  --output-dir outputs/runs/first_run \
  --n-samples 20 \
  --n-trials 20
```
3. Check `outputs/runs/first_run/` for the generated exploration and follow-up artifacts.

If a step fails, consult `docs/05_ADVANCED/troubleshooting.md` and open an issue with the run directory and logs.
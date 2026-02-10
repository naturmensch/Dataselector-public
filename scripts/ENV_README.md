# Environment Usage Guidelines

To ensure reproducible and robust runs across developer machines and CI, follow these rules:

- For interactive development (manually running tests or scripts):
  - Activate the `dataselector` env:
    ```bash
    micromamba activate dataselector
    pytest -q
    ```

- Canonical runtime invocation for scripts, Makefile targets, and CI:
  ```bash
  micromamba run -n dataselector -- <your command>
  ```
  Compatibility wrapper (optional):
  ```bash
  ./scripts/exec_in_env.sh --env dataselector -- <your command>
  ```

- Avoid `conda activate` or `source activate` inside executable scripts — they are fragile in non-interactive shells. Use `mamba run -n <env> -- <cmd>` or the wrapper instead.

- Recommended Makefile pattern for commands that need the project env:
  ```makefile
  test:
	@micromamba run -n dataselector -- pytest -q
  ```

- To bypass env checks for debugging only, set:
  ```bash
  export DATASELECTOR_IGNORE_ENV_CHECK=1
  ```
  Use sparingly and never in CI.

- CI/Workflows: prefer running with `uses: conda-incubator/setup-miniconda` and `conda run -n dataselector -- pytest` or use the wrapper via a shell step.

If you want, I can add a pre-commit hook or CI lint job that runs `scripts/check_env_usage.py` to catch regressions automatically.

Pre-commit:
- Install pre-commit: `pip install pre-commit`
- Enable hooks: `pre-commit install`

CI:
- A GitHub Actions workflow runs `scripts/check_env_usage.py` on PRs and fails when suspicious/bad patterns are found.

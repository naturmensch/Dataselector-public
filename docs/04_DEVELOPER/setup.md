# Developer Setup

Quick developer setup steps:

1. Create env: `micromamba create -f environment.yml -n dataselector`
2. Install pip extras: `./scripts/exec_in_env.sh --env dataselector -- pip install -r requirements.txt`
3. Optional: Create local editable install: `./scripts/exec_in_env.sh --env dataselector -- pip install -e .`

See `docs/01_QUICK_START/installation.md` for reproducible installation using lockfiles.
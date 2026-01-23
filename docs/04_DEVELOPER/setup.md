# Developer Setup

Quick developer setup steps:

1. Create conda env: `mamba env create -f environment.yml -n dataselector`
2. Activate and install pip extras: `conda activate dataselector && pip install -r requirements.txt`
3. Optional: Create local editable install: `pip install -e .`

See `docs/01_QUICK_START/installation.md` for reproducible installation using lockfiles.
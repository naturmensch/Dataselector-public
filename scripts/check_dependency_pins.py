#!/usr/bin/env python3
"""Check that environment.yml and requirements.txt contain the exact, validated pins.

Exits non-zero if a mismatch is found.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_YML = ROOT / "environment.yml"
REQ_TXT = ROOT / "requirements.txt"

REQUIRED = {
    "numpy": "1.26.4",
    "numba": "0.63.1",
    "umap-learn": "0.5.11",
    "apricot-select": "0.6.1",
}


def load_env_deps(path: Path):
    data = yaml.safe_load(path.read_text())
    deps = {}
    for d in data.get("dependencies", []):
        if isinstance(d, str) and "=" in d:
            name, ver = d.split("=", 1)
            deps[name.strip()] = ver.strip()
    return deps


def check_pins() -> int:
    if not ENV_YML.exists():
        print(f"ERROR: {ENV_YML} not found")
        return 2
    if not REQ_TXT.exists():
        print(f"ERROR: {REQ_TXT} not found")
        return 2

    env_deps = load_env_deps(ENV_YML)
    req_text = REQ_TXT.read_text()

    errors = []
    for pkg, ver in REQUIRED.items():
        env_ver = env_deps.get(pkg)
        if env_ver != ver:
            errors.append(f"environment.yml: {pkg}=={env_ver} != {ver}")
        if f"{pkg}=={ver}" not in req_text:
            errors.append(f"requirements.txt: missing {pkg}=={ver}")

    if errors:
        print("Dependency pin check failed:")
        for e in errors:
            print(" - ", e)
        return 1

    print("Dependency pins OK")
    return 0


if __name__ == "__main__":
    sys.exit(check_pins())

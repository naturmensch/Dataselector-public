"""Check tools for protected files and environment usage."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set

from dataselector.cli_decorators import cli_command

# Protected paths configuration
DEFAULT_PROTECTED = {
    "data/images",
    "data/archive",
    "data/raw",
    "models",
    "outputs/final_selection",
    "outputs/kdr100_selection",
}
ENV_VAR = "PROTECTED_PATHS"  # comma-separated


def get_protected_paths(extra: Iterable[str] | None = None) -> Set[str]:
    """Get the set of protected paths from defaults, environment, and extras."""
    s = set(DEFAULT_PROTECTED)
    env = os.environ.get(ENV_VAR)
    if env:
        for p in env.split(","):
            p = p.strip()
            if p:
                s.add(p)
    if extra:
        for p in extra:
            s.add(p)
    return s


def offending_files(
    staged_files: Iterable[str], protected_paths: Iterable[str]
) -> List[str]:
    """Return list of files that are inside protected paths."""
    prot = [Path(p) for p in protected_paths]
    offenders: List[str] = []
    for f in staged_files:
        pf = Path(f)
        for p in prot:
            try:
                if p == pf or p in pf.parents:
                    offenders.append(f)
                    break
            except Exception:
                # defensive
                if str(f).startswith(str(p)):
                    offenders.append(f)
                    break
    return offenders


def git_staged_files() -> List[str]:
    """Return list of staged files (git diff --name-only --cached)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "--cached"], stderr=subprocess.DEVNULL
        )
        files = out.decode().splitlines()
        return [f for f in files if f]
    except Exception:
        # Not a git repo or git not available — return empty list
        return []


def git_tracked_files() -> List[str]:
    """Return a list of tracked files in the repository (git ls-files).

    Useful for CI checks that need to ensure protected files aren't tracked.
    """
    try:
        out = subprocess.check_output(["git", "ls-files"], stderr=subprocess.DEVNULL)
        files = out.decode().splitlines()
        return [f for f in files if f]
    except Exception:
        return []


@cli_command(
    "check-protected",
    help="Check for modifications inside protected paths",
    args={
        "list": {
            "type": bool,
            "action": "store_true",
            "help": "List protected paths and exit",
        },
        "all": {
            "type": bool,
            "action": "store_true",
            "help": "Check all tracked files (git ls-files)",
        },
        "protect": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Add protected path (repeatable)",
        },
    },
)
def check_protected(
    list: bool = False,
    all: bool = False,
    protect: List[str] | None = None,
    staged_override: List[str] | None = None,
) -> int:
    """Check staged files for modifications inside protected paths.

    Args:
        list: If True, just list protected paths and return 0
        all: If True, check all tracked files (git ls-files)
        protect: Additional paths to protect
        staged_override: Explicit list of files to check (for testing)

    Returns:
        0 if OK, 2 if offending files found
    """
    protected = get_protected_paths(protect)

    if list:
        for x in sorted(protected):
            print(x)
        return 0

    staged = None
    if staged_override is not None:
        staged = staged_override
    elif all:
        staged = git_tracked_files()
    else:
        staged = git_staged_files()

    offenders = offending_files(staged, protected)
    if offenders:
        print("ERROR: The following staged files are inside protected paths:")
        for f in offenders:
            print(f"  {f}")
        print(
            "\nPlease remove them from the commit or add an exception. See PROTECTED_PATHS env var to configure."
        )
        return 2

    # All good
    return 0


@cli_command(
    "check-geo",
    help="Check geo dependencies (geopandas, pyproj, shapely, fiona, rtree)",
    args={},
)
def check_geo() -> int:
    """Quick Geo dependency checker.

    Checks if all required geo packages (geopandas, pyproj, shapely, fiona, rtree)
    are importable. Returns 0 if all OK, 2 if any missing.

    Respects pipeline_config.yaml features.geo setting - skips check if disabled.
    """
    from importlib import import_module

    import yaml

    REQS = ["geopandas", "pyproj", "shapely", "fiona", "rtree"]

    # Check pipeline config to see whether geo features are enabled
    try:
        cfg = yaml.safe_load(open("config/pipeline_config.yaml"))
        geo_enabled = bool(cfg.get("features", {}).get("geo", True))
    except Exception:
        geo_enabled = True

    if not geo_enabled:
        print(
            "Geo feature disabled in config/pipeline_config.yaml — skipping geo dependency check."
        )
        return 0

    failures = []
    for pkg in REQS:
        try:
            m = import_module(pkg)
            ver = getattr(m, "__version__", None)
            print(f"{pkg}: OK (version={ver})")
        except Exception as e:
            print(f"{pkg}: MISSING ({e})")
            failures.append(pkg)

    if failures:
        print("\nMissing geo dependencies: ", ", ".join(failures))
        print(
            "Install with conda: conda install -n dataselector -c conda-forge geopandas pyproj shapely fiona rtree rasterio"
        )
        return 2
    else:
        print("All geo dependencies available.")
        return 0


# ===== Environment Usage Checker =====


def _get_repo_root() -> Path:
    """Get repository root directory."""
    # From dataselector/tools/check.py -> repo root
    return Path(__file__).resolve().parents[2]


ROOT = _get_repo_root()
DEFAULT_SCAN_PATHS = [ROOT / "scripts", ROOT / "Makefile", ROOT / ".github/workflows"]

ENV_PATTERNS = [
    (re.compile(r"\.\/scripts\/exec_in_env\.sh"), "uses exec_in_env wrapper", "good"),
    (
        re.compile(r"(conda|mamba)\s+run\s+-n\s+['\"]?(?P<env>\w+)"),
        "uses conda/mamba run -n <env>",
        "good_if_env_matches",
    ),
    (
        re.compile(r"conda\s+activate\b"),
        "uses 'conda activate' inside scripts (fragile)",
        "bad",
    ),
    (re.compile(r"source\s+activate\b"), "uses 'source activate' (fragile)", "bad"),
    (
        re.compile(r"\bpython\b"),
        "calls python directly (may rely on PATH)",
        "suspicious",
    ),
    (
        re.compile(r"#!/usr/bin/env\s+python"),
        "shebang uses /usr/bin/env python (portable but depends on PATH)",
        "neutral",
    ),
    (
        re.compile(r"#!/usr/bin/python"),
        "shebang uses absolute python path (not portable)",
        "suspicious",
    ),
    (re.compile(r"pytest\b"), "invokes pytest directly", "suspicious"),
    (
        re.compile(r"exec_in_env\.sh\s+--env\s+dataselector"),
        "explicitly runs with dataselector env",
        "good",
    ),
]


def scan_file_env(path: Path):
    """Scan a file for environment usage patterns."""
    if not path.is_file():
        return []
    text = path.read_text(errors="ignore")
    findings = []
    for pat, desc, level in ENV_PATTERNS:
        for m in pat.finditer(text):
            snippet = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
            findings.append((pat.pattern, desc, level, m.group(0), snippet.strip()))
    # special handling for Makefile lines invoking python without wrapper
    if path.name.lower() == "makefile" or path.match("*Makefile"):
        for ln in text.splitlines():
            if ln.strip().startswith("#") or not ln.strip():
                continue
            if re.search(r"\b(py)?test\b", ln) or re.search(r"\bpython\b", ln):
                uses_wrapper = (
                    "exec_in_env.sh" in ln or "conda run" in ln or "mamba run" in ln
                )
                findings.append(
                    (
                        "Makefile-line",
                        "Makefile command",
                        "good" if uses_wrapper else "suspicious",
                        ln.strip(),
                        ln.strip(),
                    )
                )
    return findings


def scan_paths_env(paths):
    """Scan multiple paths for environment usage patterns."""
    report = {}
    for p in paths:
        p = Path(p)
        # Convert relative paths to absolute based on ROOT
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix in {".sh", ".py", ".yml", ".yaml", ".md"}
                    or f.name.lower() == "makefile"
                    or f.suffix == ""
                ):
                    findings = scan_file_env(f)
                    if findings:
                        rel_path = str(f.relative_to(ROOT))
                        report[rel_path] = findings
        else:
            findings = scan_file_env(p)
            if findings:
                rel_path = str(p.relative_to(ROOT))
                report[rel_path] = findings
    return report


@cli_command(
    "check-env",
    help="Check environment usage in scripts/CI",
    args={
        "paths": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Paths to scan (defaults to scripts, Makefile, .github/workflows)",
        },
    },
)
def check_env_usage(paths=None) -> int:
    """Check how scripts and CI targets handle environment activation/usage.

    Scans the repository (scripts/, Makefile, .github/workflows/, top-level shell scripts)
    and reports:
    - uses of exec_in_env.sh
    - uses of conda/mamba run/activate
    - bare python invocations in Makefile or scripts
    - shebangs pointing to python
    - suggestions to wrap calls with exec_in_env.sh

    Args:
        paths: List of paths to scan (defaults to scripts, Makefile, .github/workflows)

    Returns:
        0 if no hard issues found, 2 if suspicious direct calls are found
    """
    paths = paths or DEFAULT_SCAN_PATHS
    paths = [Path(p) for p in paths]
    report = scan_paths_env(paths)

    print("Environment usage audit report:\n")
    bad = 0
    for path, findings in report.items():
        print(f"- {path}:")
        for pat, desc, level, match, snippet in findings:
            print(f"    * [{level}] {desc}: '{match}'")
            if level in ("bad", "suspicious"):
                bad += 1
        print("")

    if not report:
        print("No issues found. Great!")
    else:
        print("Summary:")
        print(f"  files with findings: {len(report)}")
        print(f"  suspicious/bad occurrences: {bad}")
        print("\nRecommendations:")
        print(
            "- Prefer using './scripts/exec_in_env.sh --env dataselector -- <cmd>' or 'conda/mamba run -n dataselector -- <cmd>' for CI/Makefile targets."
        )
        print(
            "- Avoid 'conda activate' or 'source activate' inside scripts; prefer explicit runner invocation."
        )
        print(
            "- Replace direct 'python' or 'pytest' calls in Makefile with the wrapper or annotate them in the Makefile."
        )

    return 2 if bad else 0

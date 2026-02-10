"""Check tools for protected files and environment usage."""

from __future__ import annotations

import os
import re
import subprocess
import shutil
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
            "Install with micromamba: micromamba install -n dataselector -c conda-forge geopandas pyproj shapely fiona rtree rasterio"
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
DEFAULT_SCAN_PATHS = [
    ROOT / "dataselector",
    ROOT / "tests",
    ROOT / "Makefile",
    ROOT / ".github/workflows",
]

ENV_PATTERNS = [
    (
        re.compile(r"\.\/scripts\/exec_in_env\.sh"),
        "uses compatibility exec_in_env wrapper",
        "good",
    ),
    (
        re.compile(r"(micromamba|conda|mamba)\s+run\s+-n\s+['\"]?(?P<env>\w+)"),
        "uses micromamba/conda/mamba run -n <env>",
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
        "explicitly runs with dataselector env via compatibility wrapper",
        "good",
    ),
]

# ===== Script wrapper governance =====

TRANSITIONAL_WRAPPER_ALLOWLIST_VERSION = "2026-02-10"
TRANSITIONAL_WRAPPER_ALLOWLIST: set[str] = set()

INTERNAL_DOMAIN_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+dataselector\.(pipeline|selection|features|workflows)\b"),
    re.compile(
        r"^\s*import\s+dataselector\.(pipeline|selection|features|workflows)\b"
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
                uses_wrapper = "exec_in_env.sh" in ln
                uses_explicit_env = (
                    "micromamba run" in ln
                    or "conda run" in ln
                    or "mamba run" in ln
                )
                findings.append(
                    (
                        "Makefile-line",
                        "Makefile command",
                        (
                            "good"
                            if uses_wrapper
                            else ("good" if uses_explicit_env else "suspicious")
                        ),
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
    help="Check environment usage in package, tests, Makefile, and CI workflows",
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
    """Check how code and CI targets handle environment activation/usage.

    Scans the repository (scripts/, Makefile, .github/workflows/, top-level shell scripts)
    and reports:
    - uses of exec_in_env.sh compatibility wrapper
    - uses of micromamba/conda/mamba run/activate
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
            "- Prefer CLI-first commands ('python -m dataselector ...') executed via 'micromamba run -n dataselector <cmd>' (or exec_in_env compatibility wrapper)."
        )
        print(
            "- Avoid 'conda activate' or 'source activate' inside scripts; prefer explicit runner invocation."
        )
        print(
            "- Replace direct Makefile shell wrappers with canonical package entrypoints."
        )

    return 2 if bad else 0


def _scan_file_lines_for_patterns(path: Path, patterns: list[re.Pattern]) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in patterns:
            if pat.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
                break
    return hits


@cli_command(
    "check-script-wrappers",
    help="Fail if non-allowlisted top-level scripts import scientific core modules",
    args={
        "strict": {
            "type": bool,
            "action": "store_true",
            "help": "Also fail on transitional allowlist entries (for migration closeout)",
        },
    },
)
def check_script_wrappers(strict: bool = False) -> int:
    scripts_dir = ROOT / "scripts"
    if not scripts_dir.exists():
        print("scripts/ directory not found; nothing to check.")
        return 0

    offenders: list[str] = []
    allowlisted_hits: list[str] = []

    for script in sorted(scripts_dir.glob("*.py")):
        rel = str(script.relative_to(ROOT))
        hits = _scan_file_lines_for_patterns(script, INTERNAL_DOMAIN_IMPORT_PATTERNS)
        if not hits:
            continue
        if rel in TRANSITIONAL_WRAPPER_ALLOWLIST:
            allowlisted_hits.extend(hits)
            continue
        offenders.extend(hits)

    if strict and allowlisted_hits:
        offenders.extend(allowlisted_hits)

    if offenders:
        print("ERROR: Script wrapper governance violation(s) detected:")
        for item in offenders:
            print(f"  {item}")
        print(
            "\nPolicy: scripts/* may orchestrate/delegate but scientific core logic must live in dataselector/*."
        )
        if not strict:
            print(
                f"Transitional allowlist version {TRANSITIONAL_WRAPPER_ALLOWLIST_VERSION}: "
                + ", ".join(sorted(TRANSITIONAL_WRAPPER_ALLOWLIST))
            )
        return 2

    print("Script wrapper governance check passed.")
    if allowlisted_hits:
        print(
            "Note: transitional allowlist still active ({} entries).".format(
                len(TRANSITIONAL_WRAPPER_ALLOWLIST)
            )
        )
    return 0


@cli_command(
    "check-runtime-readiness",
    help="Validate canonical runtime readiness (micromamba-first policy)",
    args={
        "env": {
            "type": str,
            "default": "dataselector",
            "help": "Expected micromamba environment name",
        },
        "allow_compat": {
            "type": bool,
            "action": "store_true",
            "help": "Allow compatibility fallback (exec_in_env/conda) when micromamba is missing",
        },
    },
)
def check_runtime_readiness(env: str = "dataselector", allow_compat: bool = False) -> int:
    micromamba = shutil.which("micromamba")
    if micromamba:
        probes = [
            ["micromamba", "run", "-n", env, "python", "-c", "import sys"],
            ["micromamba", "run", "-n", env, "--", "python", "-c", "import sys"],
        ]
        for cmd in probes:
            probe = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                print(f"OK: micromamba runtime ready at {micromamba} (env={env})")
                return 0

        if allow_compat:
            print(
                "WARN: micromamba found but env '{}' is not runnable; compatibility mode allowed.".format(
                    env
                )
            )
            return 0

        print(
            "ERROR: micromamba found at {}, but env '{}' is not runnable.\n"
            "Create/update it with one of:\n"
            "  micromamba create -n {} -f environment.yml\n"
            "  micromamba env update -n {} -f environment.yml --prune".format(
                micromamba, env, env, env
            )
        )
        return 2

    if allow_compat:
        print(
            "WARN: micromamba not found, but compatibility mode allowed "
            "(exec_in_env.sh / conda fallbacks)."
        )
        return 0

    print(
        "ERROR: micromamba not found. Canonical runtime policy requires:\n"
        "  micromamba run -n dataselector <command>\n"
        "Install micromamba or run with --allow-compat for transitional checks."
    )
    return 2

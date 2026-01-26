#!/usr/bin/env python3
"""Auto-fix helper for env-usage issues detected by `scripts/check_env_usage.py`.

Usage:
  # Dry-run (default): report proposed changes
  ./scripts/auto_fix_env_usage.py --paths scripts .github Makefile

  # Apply changes: create a branch, apply edits with backups and commit
  ./scripts/auto_fix_env_usage.py --apply --branch chore/fix-env-auto-<ts> --paths scripts Makefile .github

Behaviour:
- Finds lines invoking `python` or `pytest` (excluding shebangs and lines containing 'exec_in_env.sh' or 'mamba run' or 'conda run').
- Proposes prefixing the command with `./scripts/exec_in_env.sh --env dataselector -- ` where safe.
- In `Makefile` suggests using a variable `EXEC_ENV ?= ./scripts/exec_in_env.sh --env dataselector --` and prefixing commands with `$(EXEC_ENV)`.
- Dry-run prints a concise report; --apply modifies files (backups created with `.bak`), creates a git branch and commits changes.

This tool is conservative and intended to speed up bulk remediation while keeping changes reviewable.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXEC_WRAPPER = "./scripts/exec_in_env.sh --env dataselector --"

# Patterns to match; skip shebang lines and lines already using wrapper/conda/mamba
PYTHON_PAT = re.compile(r"\bpython\b")
PYTEST_PAT = re.compile(r"\bpytest\b")
SKIP_PAT = re.compile(r"exec_in_env\.sh|mamba run|conda run|conda activate|source activate|\$\(EXEC_ENV\)")
SHEBANG = re.compile(r"^#!")

TARGET_EXTS = {".sh", ".py", ""}  # files without ext (Makefile) included via path

# Files and dirs to skip from auto-fix to avoid self-modification or touching third-party code
EXCLUDE_FILES = {"scripts/check_env_usage.py", "scripts/auto_fix_env_usage.py"}
SKIP_DIRS = {"venv", ".venv", "site-packages", "lib", "bin", "env", "venvs"}

# Restrict default allowed roots to repository-owned code only
ALLOWED_ROOTS = {"scripts", "src", ".github", "Makefile"}


def find_candidates(paths: List[Path]) -> List[Tuple[Path, int, str, str]]:
    """Return list of (file, lineno, orig_line, suggested_line)"""
    findings = []
    for p in paths:
        # If scanning '.' or an absolute root, be conservative and scan only allowed roots
        if str(p) == '.' or str(p) == str(ROOT):
            p = ROOT
            roots = [ROOT / r for r in ALLOWED_ROOTS if (ROOT / r).exists()]
            for root_p in roots:
                for fp in sorted(root_p.rglob("*")):
                    if not fp.is_file():
                        continue
                    rel = os.path.relpath(fp, ROOT)
                    # skip files explicitly excluded
                    if rel in EXCLUDE_FILES:
                        continue
                    # skip third-party / venv-ish directories
                    if any(part in SKIP_DIRS for part in Path(rel).parts):
                        continue
                    if fp.suffix not in TARGET_EXTS and fp.name not in {"Makefile"}:
                        continue
                    _scan_file(fp, findings)
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if not fp.is_file():
                    continue
                rel = os.path.relpath(fp, ROOT)
                if rel in EXCLUDE_FILES:
                    continue
                if any(part in SKIP_DIRS for part in Path(rel).parts):
                    continue
                if fp.suffix not in TARGET_EXTS and fp.name not in {"Makefile"}:
                    continue
                _scan_file(fp, findings)
        elif p.is_file():
            rel = os.path.relpath(p, ROOT)
            if rel not in EXCLUDE_FILES and not any(part in SKIP_DIRS for part in Path(rel).parts):
                _scan_file(p, findings)
    return findings


def _scan_file(fp: Path, findings: List[Tuple[Path, int, str, str]]):
    try:
        text = fp.read_text(encoding="utf-8")
    except Exception:
        return

    # Special handling for YAML workflow files: edit 'run:' blocks
    if fp.suffix in {".yml", ".yaml"} and "github/workflows" in str(fp):
        lines = text.splitlines()
        in_run_block = False
        run_indent = None
        for i, line in enumerate(lines, start=1):
            if re.match(r"^\s*run:\s*\|", line):
                in_run_block = True
                run_indent = len(line) - len(line.lstrip())
                continue
            if re.match(r"^\s*run:\s*.*", line) and not '|' in line:
                # inline run: CLI
                if SKIP_PAT.search(line):
                    continue
                if PYTHON_PAT.search(line) or PYTEST_PAT.search(line):
                    sug = line
                    sug = re.sub(r"\bpython\b", f"{EXEC_WRAPPER} python", sug)
                    findings.append((fp, i, line, sug))
                continue
            if in_run_block:
                # End of block when indentation less than run_indent and non-empty
                if line.strip() and (len(line) - len(line.lstrip())) <= run_indent:
                    in_run_block = False
                    run_indent = None
                    continue
                # Inside block
                if SKIP_PAT.search(line):
                    continue
                if PYTHON_PAT.search(line) or PYTEST_PAT.search(line):
                    sug = line
                    sug_stripped = sug.lstrip()
                    pref = ' ' * (len(sug) - len(sug_stripped)) + EXEC_WRAPPER + ' '
                    sug = pref + sug_stripped
                    findings.append((fp, i, line, sug))
        return

    for i, line in enumerate(text.splitlines(), start=1):
        if SHEBANG.match(line):
            continue
        if SKIP_PAT.search(line):
            continue
        if PYTHON_PAT.search(line) or PYTEST_PAT.search(line):
            # Heuristic: avoid touching comment lines and simple pip insns in Makefile
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Skip explicit shebang or env python paths
            if re.search(r"/python[0-9.]*\b", line):
                continue
            suggestion = _suggest_replacement(fp, line)
            if suggestion and suggestion != line:
                findings.append((fp, i, line, suggestion))


def _suggest_replacement(fp: Path, line: str) -> str | None:
    # For Makefile or files named Makefile: use $(EXEC_ENV)
    if fp.name == "Makefile":
        # If the line contains 'python -m pip' or 'pytest' or 'python -m' suggest prefixing with $(EXEC_ENV)
        if PYTEST_PAT.search(line) or PYTHON_PAT.search(line):
            if "$(EXEC_ENV)" in line:
                return None
            return line.replace("python", "$(EXEC_ENV) python")
        return None

    # For shell scripts / other files: prefix the entire command with wrapper if it appears to run python/pytest
    if PYTHON_PAT.search(line) or PYTEST_PAT.search(line):
        # If the line already contains wrapper or conda/mamba, skip
        if SKIP_PAT.search(line):
            return None
        # If the line contains variable substitutions like ${ROOT}/scripts/exec_in_env.sh, treat as already fixed
        if "exec_in_env.sh" in line:
            return None
        # Build suggestion: attempt to preserve leading indentation and shell assignment parts
        m = re.match(r"(\s*)(.*)$", line)
        indent, rest = m.group(1), m.group(2)
        # Avoid double prefix if line begins with environment assignment like 'PYTHONPATH=. python ...'
        suggestion = f"{indent}{EXEC_WRAPPER} {rest}"
        return suggestion
    return None


def print_report(findings: List[Tuple[Path, int, str, str]]):
    if not findings:
        print("No findings.")
        return
    print(f"Found {len(findings)} proposed edits:\n")
    last_file = None
    for fp, ln, orig, sug in findings:
        if fp != last_file:
            print(f"File: {fp}")
            last_file = fp
        print(f"  Line {ln}:")
        print(f"    - orig: {orig.strip()}")
        print(f"    - sug : {sug.strip()}\n")


def apply_changes(findings: List[Tuple[Path, int, str, str]], branch: str):
    if not findings:
        print("Nothing to apply.")
        return
    # Create git branch
    subprocess.check_call(["git", "checkout", "-b", branch])
    files_changed = set()
    # Group findings per file and apply replacements safely
    by_file = {}
    for fp, ln, orig, sug in findings:
        by_file.setdefault(fp, []).append((ln, orig, sug))
    for fp, changes in by_file.items():
        rel = os.path.relpath(fp, ROOT)
        # Only apply changes to allowed roots (safety guard)
        if not any(str(rel).startswith(ar.rstrip('/')) for ar in ALLOWED_ROOTS):
            print(f"Skipping applying changes to {fp} (outside allowed roots)")
            continue
        text = fp.read_text(encoding="utf-8")
        lines = text.splitlines()
        # Backup
        bak = fp.with_suffix(fp.suffix + ".bak")
        shutil.copy(fp, bak)
        for ln, orig, sug in sorted(changes, key=lambda x: x[0], reverse=False):
            idx = ln - 1
            if lines[idx].rstrip() == orig.rstrip():
                lines[idx] = sug
            else:
                # best-effort: search for the orig fragment near the line
                found = False
                for d in range(max(0, idx - 3), min(len(lines), idx + 3)):
                    if orig.strip() == lines[d].strip():
                        lines[d] = sug
                        found = True
                        break
                if not found:
                    print(f"Warning: could not apply change in {fp} at {ln}; skipping")
        new_text = "\n".join(lines) + "\n"
        fp.write_text(new_text, encoding="utf-8")
        files_changed.add(str(fp))
    # Commit
    if files_changed:
        subprocess.check_call(["git", "add"] + sorted(files_changed))
        subprocess.check_call(["git", "commit", "-m", f"chore: auto-fix env usage (script) ({time.strftime('%Y%m%dT%H%M%SZ')})"])
        subprocess.check_call(["git", "push", "--set-upstream", "origin", branch])
        print(f"Applied changes and pushed branch {branch}.")
    else:
        print("No allowed files to change; nothing was applied.")


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=["scripts", "Makefile", ".github"], help="Paths or globs to scan (use '.' carefully; default scans only project code)")
    parser.add_argument("--yes", action="store_true", help="When --apply, proceed without an interactive confirmation")
    parser.add_argument("--apply", action="store_true", help="Apply changes (create branch, modify files, commit & push)")
    parser.add_argument("--branch", default=f"chore/fix-env-auto-{time.strftime('%Y%m%dT%H%M%SZ')}", help="Branch name to create when applying")
    parser.add_argument("--env-wrapper", default=None, help="Path to wrapper to use in suggestions")
    args = parser.parse_args(argv)

    # Validate paths
    paths = [Path(p) for p in args.paths]
    inc = [p for p in paths if p.exists()]
    if not inc:
        print("No paths found to scan.")
        sys.exit(1)

    global EXEC_WRAPPER
    if args.env_wrapper:
        EXEC_WRAPPER = args.env_wrapper

    findings = find_candidates(inc)
    print_report(findings)

    if args.apply:
        if not args.yes:
            resp = input(f"About to apply {len(findings)} suggested edits and create branch '{args.branch}'. Proceed? [y/N] ")
            if resp.strip().lower() != 'y':
                print("Aborting apply.")
                return
        apply_changes(findings, args.branch)


if __name__ == "__main__":
    main()

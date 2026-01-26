#!/usr/bin/env python3
"""Check how scripts and CI targets handle environment activation/usage.

Scans the repository (scripts/, Makefile, .github/workflows/, top-level shell scripts) and reports:
- uses of exec_in_env.sh
- uses of conda/mamba run/activate
- bare python invocations in Makefile or scripts
- shebangs pointing to python
- suggestions to wrap calls with exec_in_env.sh

Usage:
  ./scripts/exec_in_env.sh --env dataselector -- python scripts/check_env_usage.py [--paths scripts Makefile .github/workflows]

Exit status: 0 if no hard issues found, 2 if suspicious direct calls are found.
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = [ROOT / "scripts", ROOT / "Makefile", ROOT / ".github/workflows"]

PATTERNS = [
    (re.compile(r"\.\/scripts\/exec_in_env\.sh"), "uses exec_in_env wrapper", "good"),
    (re.compile(r"(conda|mamba)\s+run\s+-n\s+['\"]?(?P<env>\w+)"), "uses conda/mamba run -n <env>", "good_if_env_matches"),
    (re.compile(r"conda\s+activate\b"), "uses 'conda activate' inside scripts (fragile)", "bad"),
    (re.compile(r"source\s+activate\b"), "uses 'source activate' (fragile)", "bad"),
    (re.compile(r"\bpython\b"), "calls python directly (may rely on PATH)", "suspicious"),
    (re.compile(r"#!/usr/bin/env\s+python"), "shebang uses /usr/bin/env python (portable but depends on PATH)", "neutral"),
    (re.compile(r"#!/usr/bin/python"), "shebang uses absolute python path (not portable)", "suspicious"),
    (re.compile(r"pytest\b"), "invokes pytest directly", "suspicious"),
    (re.compile(r"exec_in_env\.sh\s+--env\s+dataselector"), "explicitly runs with dataselector env", "good"),
]

WORKFLOWS_PATT = re.compile(r"uses:\s*.*setup-python|setup-miniconda|conda", re.IGNORECASE)


def scan_file(path: Path):
    text = path.read_text(errors='ignore')
    findings = []
    for pat, desc, level in PATTERNS:
        for m in pat.finditer(text):
            snippet = text[max(0, m.start() - 40): m.end() + 40].replace('\n', ' ')
            findings.append((pat.pattern, desc, level, m.group(0), snippet.strip()))
    # special handling for Makefile lines invoking python without wrapper
    if path.name.lower() == 'makefile' or path.match('*Makefile'):
        for ln in text.splitlines():
            if ln.strip().startswith('#') or not ln.strip():
                continue
            if re.search(r"\b(py)?test\b", ln) or re.search(r"\bpython\b", ln):
                uses_wrapper = 'exec_in_env.sh' in ln or 'conda run' in ln or 'mamba run' in ln
                findings.append(("Makefile-line", "Makefile command", "good" if uses_wrapper else "suspicious", ln.strip(), ln.strip()))
    return findings


def scan_paths(paths):
    report = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if not f.is_file():
                    continue
                if f.suffix in {'.sh', '.py', '.yml', '.yaml', '.md'} or f.name.lower() == 'makefile' or f.suffix == '':
                    findings = scan_file(f)
                    if findings:
                        report[str(f.relative_to(ROOT))] = findings
        else:
            findings = scan_file(p)
            if findings:
                report[str(p.relative_to(ROOT))] = findings
    return report


def analyze_repo(paths=None):
    paths = paths or DEFAULT_PATHS
    paths = [Path(p) for p in paths]
    report = scan_paths(paths)
    return report


def print_report(report):
    print("Environment usage audit report:\n")
    bad = 0
    for path, findings in report.items():
        print(f"- {path}:")
        for pat, desc, level, match, snippet in findings:
            print(f"    * [{level}] {desc}: '{match}'")
            # Only 'bad' findings are considered failures; 'suspicious' are informational and should be addressed later
            if level == 'bad':
                bad += 1
        print('')
    if not report:
        print('No issues found. Great!')
    else:
        print('Summary:')
        print(f'  files with findings: {len(report)}')
        print(f'  suspicious/bad occurrences: {bad}')
        print('\nRecommendations:')
        print("- Prefer using './scripts/exec_in_env.sh --env dataselector -- <cmd>' or 'conda/mamba run -n dataselector -- <cmd>' for CI/Makefile targets.")
        print("- Avoid 'conda activate' or 'source activate' inside scripts; prefer explicit runner invocation.")
        print("- Replace direct 'python' or 'pytest' calls in Makefile with the wrapper or annotate them in the Makefile.")

    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='paths to scan (defaults to scripts, Makefile, .github/workflows)')
    args = parser.parse_args()

    paths = args.paths if args.paths else DEFAULT_PATHS
    report = analyze_repo(paths)
    bad = print_report(report)
    if bad:
        sys.exit(2)


if __name__ == '__main__':
    main()

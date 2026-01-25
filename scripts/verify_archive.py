#!/usr/bin/env python3
"""Lightweight verify script to ensure codebase does not reference archived tests.

Checks for textual references to `archive/tests` or `_OLD` test paths and exits with code 1 when found.
Usage:
    ./scripts/exec_in_env.sh --env dataselector -- python scripts/verify_archive.py --fail-on-reference

It is intentionally conservative (text search). If you have false positives, review matches and adjust.
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"\barchive/tests\b"),
    re.compile(r"\btests/.+_OLD\b"),
    re.compile(r"from\s+archive\.tests"),
    re.compile(r"import\s+archive\.tests"),
]

matches = []
for p in ROOT.rglob("**/*.*"):
    if not p.is_file():
        continue
    if any(part.startswith('.git') for part in p.parts):
        continue
    try:
        text = p.read_text(errors='ignore')
    except Exception:
        continue
    for pat in PATTERNS:
        for m in pat.finditer(text):
            # record snippet
            start = max(m.start() - 40, 0)
            end = min(m.end() + 40, len(text))
            snippet = text[start:end].replace('\n', ' ')
            matches.append((str(p.relative_to(ROOT)), m.group(0), snippet))

if matches:
    print("Found references to archived tests or _OLD files:")
    for fname, token, snippet in matches:
        print(f" - {fname}: '{token}' -> ...{snippet}...")
    print('\nPlease review and remove references before moving tests to archive or update the reference target.')
    if "--fail-on-reference" in sys.argv:
        sys.exit(1)
else:
    print("No references to archived tests found.")

sys.exit(0)
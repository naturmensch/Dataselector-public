#!/usr/bin/env python3
"""Apply regex pattern fixes to common malformed relative links.
"""
import re
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'docs' / 'Pipeline_260115.md'
archive = ROOT / 'archive_local' / 'docs_migration_backup' / 'patches'
archive.mkdir(parents=True, exist_ok=True)

text = TARGET.read_text(encoding='utf-8')
backup = archive / (TARGET.name + '.pattern.orig')
if not backup.exists():
    backup.write_bytes(text.encode('utf-8'))

replacements = [
    (r"\.\./src([A-Za-z0-9_]+\.py)", r"../src/\1"),
    (r"\.\./configpipeline_config\.yaml", r"../config/pipeline_config.yaml"),
    (r"scripts/(\w+\.py) \"scripts/\1\"", r"../scripts/\1"),
    (r"scripts/(\w+\.py) \"scripts/\1\"", r"../scripts/\1"),
    (r"\]\(LICENSE\)", r"](../LICENSE)"),
]

new_text = text
for pat, rep in replacements:
    new_text = re.sub(pat, rep, new_text)

if new_text != text:
    TARGET.write_text(new_text, encoding='utf-8')
    print('Applied pattern fixes to', TARGET)
else:
    print('No pattern matches found in', TARGET)

# Also apply similar replacements in docs/reorganize_readmes.md for LICENSE
ORGS = ROOT / 'docs' / 'reorganize_readmes.md'
text2 = ORGS.read_text(encoding='utf-8')
backup2 = archive / (ORGS.name + '.pattern.orig')
if not backup2.exists():
    backup2.write_bytes(text2.encode('utf-8'))
new_text2 = re.sub(r"\]\(LICENSE\)", r"](../LICENSE)", text2)
if new_text2 != text2:
    ORGS.write_text(new_text2, encoding='utf-8')
    print('Patched LICENSE link in', ORGS)
else:
    print('No LICENSE link to patch in', ORGS)

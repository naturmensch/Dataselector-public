#!/usr/bin/env python3
"""Patch frequently-occurring broken link patterns with explicit mapping rules.

Rules are of form: old_substring -> new_target (path relative to repo root).
The script replaces occurrences in all .md files and writes backups under archive_local/docs_migration_backup/patches/.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MD = list(ROOT.rglob("*.md"))
archive = ROOT / 'archive_local' / 'docs_migration_backup' / 'patches'
archive.mkdir(parents=True, exist_ok=True)

# Mapping from old link targets (substring match) to new target (repo-relative path)
MAPPINGS = {
    'THESIS_PIPELINE_QUICKSTART.md': 'docs/03_USER_GUIDES/thesis_pipeline.md',
    '../docs/REPRODUCIBILITY.md': 'docs/03_USER_GUIDES/reproducibility.md',
    'docs/REPRODUCIBILITY.md': 'docs/03_USER_GUIDES/reproducibility.md',
    '../docs/02_THEORY/methodology.md': 'docs/02_THEORY/methodology.md',
    'docs/METHODOLOGY.md': 'docs/02_THEORY/methodology.md',
    'docs/02_THEORY/methodology.md': 'docs/02_THEORY/methodology.md',
    'file://<dataselector-repo>/src/': 'src/',
    'file://<dataselector-repo>/config/': 'config/',
    'file://<dataselector-repo>/scripts/': 'scripts/',
    'docs/07_ARCHIVE/CHANGELOG.md': 'docs/07_ARCHIVE/CHANGELOG.md',
}

replacements_made = []
for md in MD:
    text = md.read_text(encoding='utf-8')
    new_text = text
    patched = False
    for old, new in MAPPINGS.items():
        if old in new_text:
            # compute relative path from md.parent to new target
            new_abs = (ROOT / new).resolve()
            rel = os.path.relpath(str(new_abs), start=str(md.parent))
            new_text = new_text.replace(old, rel)
            patched = True
    if patched and new_text != text:
        # backup
        backup = archive / (md.name + '.patch.orig')
        if not backup.exists():
            backup.write_bytes(text.encode('utf-8'))
        md.write_text(new_text, encoding='utf-8')
        replacements_made.append(str(md))

print(f'Patched {len(replacements_made)} files.')
if replacements_made:
    for p in replacements_made:
        print('Patched:', p)

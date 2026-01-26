#!/usr/bin/env python3
"""Attempt to auto-fix broken relative links reported by docs_link_check.py.

Workflow:
- Read outputs/docs_link_check_report.txt for lines: "BROKEN: source.md -> target"
- For each target, attempt to find unique candidate files in repo by basename match (case-insensitive)
- If unique candidate found, replace target in source markdown with relative path to candidate
- Backup original source file to archive_local/docs_migration_backup/
- Produce a report outputs/docs_link_autofix_report.txt with actions taken and remaining manual fixes
"""
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[1]
report_in = ROOT / "outputs" / "docs_link_check_report.txt"
if not report_in.exists():
    print("No link check report found. Run scripts/docs_link_check.py first.")
    raise SystemExit(1)

lines = report_in.read_text(encoding='utf-8').splitlines()
broken = []
for line in lines:
    line = line.strip()
    if not line or not line.startswith('BROKEN:'):
        continue
    # format: BROKEN: path/to/file.md -> target
    m = re.match(r'BROKEN:\s+(.*?)\s+->\s+(.*)$', line)
    if m:
        source = m.group(1)
        target = m.group(2)
        broken.append((source, target))

archive_dir = ROOT / 'archive_local' / 'docs_migration_backup' / 'autofix'
archive_dir.mkdir(parents=True, exist_ok=True)

actions = []
manual = []
for src, tgt in broken:
    src_path = (ROOT / src).resolve()
    if not src_path.exists():
        manual.append((src, tgt, 'source_missing'))
        continue
    # Normalize target basename
    tgt_basename = Path(tgt).name
    # Search repo for matching basename (case-insensitive)
    candidates = [p for p in ROOT.rglob('*') if p.is_file() and p.name.lower() == tgt_basename.lower()]
    if len(candidates) == 1:
        cand = candidates[0]
        # compute relative path from src parent (robust across directories)
        import os
        rel_path = os.path.relpath(str(cand), start=str(src_path.parent))
        rel = Path(rel_path)
        # backup source
        backup = archive_dir / (src_path.name + '.orig')
        if not backup.exists():
            backup.write_bytes(src_path.read_bytes())
        text = src_path.read_text(encoding='utf-8')
        # Replace all occurrences of the broken target (naive but sufficient)
        new_text = text.replace(f"({tgt})", f"({rel.as_posix()})")
        if new_text == text:
            # Maybe the link was wrapped or had ./ prefix; try basename match only
            new_text = text.replace(f"({tgt_basename})", f"({rel.as_posix()})")
        if new_text != text:
            src_path.write_text(new_text, encoding='utf-8')
            actions.append((src, tgt, rel.as_posix()))
        else:
            manual.append((src, tgt, 'no_replacement_made'))
    else:
        manual.append((src, tgt, f'{len(candidates)}_candidates'))

out = ROOT / 'outputs' / 'docs_link_autofix_report.txt'
with out.open('w', encoding='utf-8') as fh:
    if actions:
        fh.write('Auto-replacements made:\n')
        for a in actions:
            fh.write(f'{a[0]}: {a[1]} -> {a[2]}\n')
        fh.write('\n')
    if manual:
        fh.write('Manual review required for:\n')
        for m in manual:
            fh.write(f'{m[0]}: {m[1]} -> {m[2]}\n')

print(f'Auto-fixed {len(actions)} links; {len(manual)} need manual review.')
print(f'Report: {out}')
if manual:
    raise SystemExit(2)
else:
    raise SystemExit(0)

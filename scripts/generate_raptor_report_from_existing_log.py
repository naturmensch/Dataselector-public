#!/usr/bin/env python3
"""Generate raptor_bericht.md from existing XXL_FULL_RUN.log without starting a run."""
from pathlib import Path
import json
from datetime import datetime, timezone
import glob

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / 'outputs'
# Prefer the most recent timestamped log, fall back to latest symlink
logs = sorted(LOG_DIR.glob('XXL_FULL_RUN_*.log'))
if logs:
    LOG_FILE = logs[-1]
else:
    LOG_FILE = LOG_DIR / 'XXL_FULL_RUN.log'

# Find latest XXL run dir
xxl_dirs = sorted(glob.glob(str(ROOT / 'outputs' / 'runs' / '*hamburg_xxl_final*')))
latest_xxl = Path(xxl_dirs[-1]) if xxl_dirs else None

# Try load final selection
final_selection_file = ROOT / 'outputs' / 'THESIS_FINAL_SELECTION_XXL.json'
final_selection = None
if final_selection_file.exists():
    final_selection = json.load(open(final_selection_file))

# Scan log for phase events
phase_events = []
log_text = ''
if LOG_FILE.exists():
    log_text = LOG_FILE.read_text()
    if 'Phase 1 ABGESCHLOSSEN' in log_text or 'PHASE 1 COMPLETE' in log_text:
        phase_events.append('PHASE 1 COMPLETE')
    if 'Phase 2 COMPLETE' in log_text:
        phase_events.append('PHASE 2 COMPLETE')
    if 'Phase 3 COMPLETE' in log_text:
        phase_events.append('PHASE 3 COMPLETE')
    if 'Phase 4 COMPLETE' in log_text:
        phase_events.append('PHASE 4 COMPLETE')

report_lines = []
report_lines.append(f"# Raptor Bericht — XXL Full Run\n")
report_lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}Z")
report_lines.append('\n## Observed phase events')
for e in phase_events:
    report_lines.append(f"- {e}")

report_lines.append('\n## Artifacts')
if latest_xxl:
    report_lines.append(f"- XXL run dir: {latest_xxl}")
    if (latest_xxl / 'results' / 'trials.csv').exists():
        report_lines.append(f"  - trials.csv: {(latest_xxl / 'results' / 'trials.csv')} (size: { (latest_xxl / 'results' / 'trials.csv').stat().st_size } bytes)")
else:
    report_lines.append('- XXL run dir: Not found')

if final_selection:
    report_lines.append(f"- Final selection JSON: {final_selection_file}")
    report_lines.append(f"  - Best value: {final_selection.get('best_value')} @ trial #{final_selection.get('best_trial')}")
    report_lines.append(f"  - n_trials recorded: {final_selection.get('n_trials')}")
else:
    report_lines.append('- Final selection JSON: Not found')

# Add log excerpt
report_lines.append('\n## Log excerpt (last 500 lines)')
if LOG_FILE.exists():
    lines = LOG_FILE.read_text().splitlines()
    excerpt = '\n'.join(lines[-500:])
    report_lines.append('```\n' + excerpt + '\n```')
else:
    report_lines.append('Log file not found')

# Write report (timestamped)
report_ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
if latest_xxl:
    report_path = latest_xxl / f'raptor_bericht_{report_ts}.md'
    report_latest = latest_xxl / 'raptor_bericht.md'
else:
    report_path = ROOT / 'outputs' / f'raptor_bericht_{report_ts}.md'
    report_latest = ROOT / 'outputs' / 'raptor_bericht.md'
report_path.write_text('\n'.join(report_lines))
report_latest.write_text('\n'.join(report_lines))
print(f'Wrote report to: {report_path} (latest copy: {report_latest})')

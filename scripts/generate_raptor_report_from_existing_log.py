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
runs_root = ROOT / 'outputs' / 'runs'
xxl_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and 'hamburg' in p.name.lower() and 'xxl' in p.name.lower()]) if runs_root.exists() else []
latest_xxl = xxl_dirs[-1] if xxl_dirs else None

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

# Write report (timestamped) into monitor_reports dir
report_ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
if latest_xxl:
    reports_dir = latest_xxl / 'monitor_reports'
else:
    reports_dir = ROOT / 'outputs' / 'monitor_reports'
reports_dir.mkdir(parents=True, exist_ok=True)

report_md = reports_dir / f'monitor_report_{report_ts}.md'
report_meta = reports_dir / f'monitor_meta_{report_ts}.json'
report_latest_md = reports_dir / 'monitor_report.md'
report_latest_meta = reports_dir / 'monitor_meta.json'

report_md.write_text('\n'.join(report_lines))
try:
    report_meta.write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'observed_phase_events': phase_events,
        'xxl_run_dir': str(latest_xxl) if latest_xxl else None,
    }, indent=2))
except Exception:
    pass

report_latest_md.write_text('\n'.join(report_lines))
try:
    report_latest_meta.write_text(json.dumps({
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'observed_phase_events': phase_events,
        'xxl_run_dir': str(latest_xxl) if latest_xxl else None,
    }, indent=2))
except Exception:
    pass

print(f'Wrote report to: {report_md} (latest copies: {report_latest_md}, {report_latest_meta})')

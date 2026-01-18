#!/usr/bin/env python3
"""Start full XXL pipeline run and monitor progress.

This script launches `scripts/xxl_KDR146_run_thesis_complete.py` as a subprocess,
streams stdout/stderr to `outputs/XXL_FULL_RUN.log`, monitors progress by
inspecting the log and `outputs/runs/` for the final XXL run directory, and
writes a scientific report `raptor_bericht.md` inside the discovered run dir
when the run finishes (success or failure).

Usage:
    PYTHONPATH=. python scripts/xxl_full_run_monitor.py

Be aware: this performs a full production execution and can take a long time.
"""
from pathlib import Path
import subprocess
import sys
import time
import os
import json
from datetime import datetime, timezone
import glob
import re
import shutil
import signal

ROOT = Path(__file__).resolve().parents[1]
# Ensure ROOT is in sys.path so we can import scripts modules directly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_FILE = ROOT / 'outputs' / 'XXL_FULL_RUN.log'
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MAIN_SCRIPT = ROOT / 'scripts' / 'xxl_KDR146_run_thesis_complete.py'

# Ensure PYTHONPATH in env
env = os.environ.copy()
env['PYTHONPATH'] = str(ROOT)
env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output for real-time monitoring

# Create per-run timestamped log file and make `XXL_FULL_RUN.log` point to it (symlink)
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
LOG_FILE_TS = LOG_FILE.parent / f'XXL_FULL_RUN_{ts}.log'
# Ensure we do not overwrite an existing timestamped file
if not LOG_FILE_TS.exists():
    LOG_FILE_TS.touch()
# Replace or create a symlink `XXL_FULL_RUN.log` -> timestamped file
try:
    if LOG_FILE.exists() or LOG_FILE.is_symlink():
        try:
            LOG_FILE.unlink()
        except Exception:
            pass
    LOG_FILE.symlink_to(LOG_FILE_TS.name)
    print(f"Created symlink: {LOG_FILE} -> {LOG_FILE_TS.name}")
except Exception:
    # fallback: keep a copy mechanism if symlink not allowed
    pass

# ACTIVE_LOG is the actual file we will write to and read from
ACTIVE_LOG = LOG_FILE_TS

# Start the full run subprocess
cmd = [sys.executable, str(MAIN_SCRIPT)]
print(f"Starting full run: {' '.join(cmd)} (log: {ACTIVE_LOG})")
with open(ACTIVE_LOG, 'ab') as logf:
    # start_new_session=True creates a new process group, allowing us to kill the whole tree
    process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)

start_time = time.time()
print(f"PID: {process.pid}, writing logs to: {LOG_FILE}")

# Monitoring loop
last_size = 0
phase_events = []
detected_xxl = None
seen_lines = set()

# Helper to print and append monitor messages to active log
def _monitor_log(msg: str) -> None:
    tsmsg = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(tsmsg)
    try:
        with open(ACTIVE_LOG, 'a') as _lf:
            _lf.write(tsmsg + "\n")
    except Exception:
        pass

try:
    while True:
        time.sleep(30)  # poll interval
        # Check if process has exited
        ret = process.poll()
        # Read newly appended log content only from the ACTIVE log file
        try:
            with open(ACTIVE_LOG, 'r') as f:
                f.seek(last_size)
                new_content = f.read()
                last_size = f.tell()
        except Exception:
            new_content = ''

        # Process only new lines to avoid repeated messages
        if new_content:
            for line in new_content.splitlines():
                if not line or line in seen_lines:
                    continue
                seen_lines.add(line)

                # Detect running tasks (e.g., "Running 1/20: alpha=...")
                m = re.search(r'Running\s+\d+/\d+:\s*(.+)', line)
                if m:
                    _monitor_log(f"Starte: {m.group(1)}")
                    continue

                # Detect export completions
                m = re.search(r'Auswahl exportiert nach:\s*(.+)$', line)
                if m:
                    _monitor_log(f"Prozess beendet: Auswahl exportiert nach: {m.group(1)}")
                    continue

                m = re.search(r'Saved results:\s*(.+)\s*\(csv', line)
                if m:
                    _monitor_log(f"Prozess beendet: Saved results: {m.group(1)}")
                    continue

                # Phase completions (localized messages)
                if 'Phase 1 ABGESCHLOSSEN' in line or 'PHASE 1 COMPLETE' in line:
                    if 'PHASE 1 COMPLETE' not in phase_events:
                        phase_events.append('PHASE 1 COMPLETE')
                        _monitor_log('OBSERVED: PHASE 1 COMPLETE')
                    continue
                if 'Phase 2 COMPLETE' in line:
                    if 'PHASE 2 COMPLETE' not in phase_events:
                        phase_events.append('PHASE 2 COMPLETE')
                        _monitor_log('OBSERVED: PHASE 2 COMPLETE')
                    continue
                if 'Phase 3 COMPLETE' in line:
                    if 'PHASE 3 COMPLETE' not in phase_events:
                        phase_events.append('PHASE 3 COMPLETE')
                        _monitor_log('OBSERVED: PHASE 3 COMPLETE')
                    continue
                if 'Phase 4 COMPLETE' in line:
                    if 'PHASE 4 COMPLETE' not in phase_events:
                        phase_events.append('PHASE 4 COMPLETE')
                        _monitor_log('OBSERVED: PHASE 4 COMPLETE')
                    continue

        # Detect creation of XXXL run folder
        xxl_dirs = sorted(glob.glob(str(ROOT / 'outputs' / 'runs' / '*hamburg_xxl_final*')))
        if xxl_dirs:
            latest_xxl = Path(xxl_dirs[-1])
            if (latest_xxl / 'results' / 'trials.csv').exists():
                if latest_xxl != detected_xxl:
                    detected_xxl = latest_xxl
                    _monitor_log(f"Detected XXL run directory: {latest_xxl}")

        if ret is not None:
            print(f"Process exited with code: {ret}")
            break

except KeyboardInterrupt:
    print('KeyboardInterrupt received: terminating child process group')
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)
    ret = process.returncode

end_time = time.time()

# After process end: assemble report
exit_code = process.returncode
elapsed = end_time - start_time

# Find latest XXL run dir
xxl_dirs = sorted(glob.glob(str(ROOT / 'outputs' / 'runs' / '*hamburg_xxl_final*')))
if xxl_dirs:
    latest_xxl = Path(xxl_dirs[-1])
else:
    latest_xxl = None

# Load final selection if exists
final_selection = None
final_selection_file = ROOT / 'outputs' / 'THESIS_FINAL_SELECTION_XXL.json'
if final_selection_file.exists():
    with open(final_selection_file) as f:
        final_selection = json.load(f)

# Collect some metrics
report_lines = []
report_lines.append(f"# Raptor Bericht — XXL Full Run\n")
report_lines.append(f"**Run started**: {datetime.fromtimestamp(start_time, timezone.utc).isoformat()}Z")
report_lines.append(f"**Run finished**: {datetime.fromtimestamp(end_time, timezone.utc).isoformat()}Z")
report_lines.append(f"**Elapsed (s)**: {elapsed:.1f}")
report_lines.append(f"**Exit code**: {exit_code}\n")

report_lines.append("## Observed phase events")
for e in phase_events:
    report_lines.append(f"- {e}")

report_lines.append('\n## Artifacts')
if latest_xxl:
    report_lines.append(f"- XXL run dir: {latest_xxl}")
    if (latest_xxl / 'results' / 'trials.csv').exists():
        report_lines.append(f"  - trials.csv: {(latest_xxl / 'results' / 'trials.csv')} (size: { (latest_xxl / 'results' / 'trials.csv').stat().st_size } bytes)")
else:
    report_lines.append("- XXL run dir: Not found")

if final_selection:
    report_lines.append(f"- Final selection JSON: {final_selection_file}")
    report_lines.append(f"  - Best value: {final_selection.get('best_value')} @ trial #{final_selection.get('best_trial')}")
    report_lines.append(f"  - n_trials recorded: {final_selection.get('n_trials')}")
else:
    report_lines.append("- Final selection JSON: Not found")

# Convergence baseline analysis (attempt)
try:
    from scripts.xxl_KDR146_run_thesis_complete import _validate_convergence_from_validation_data
    conv = _validate_convergence_from_validation_data(ROOT)
    if conv:
        report_lines.append('\n## Convergence baseline analysis')
        report_lines.append(f"- n_seeds_analyzed: {conv['n_seeds_analyzed']}")
        report_lines.append(f"- convergence_99_trials_median: {conv['convergence_99_trials_median']}")
        report_lines.append(f"- convergence_99_trials_min: {conv['convergence_99_trials_min']}")
        report_lines.append(f"- convergence_99_trials_max: {conv['convergence_99_trials_max']}")
    else:
        report_lines.append('\n## Convergence baseline analysis')
        report_lines.append("- Could not compute convergence baseline from existing validation data (insufficient/missing runs)")
except Exception as e:
    report_lines.append('\n## Convergence baseline analysis')
    report_lines.append(f"- Error while computing baseline: {e}")

# Add section with log excerpts
report_lines.append('\n## Log excerpts (last 500 lines)')
try:
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    excerpt = ''.join(lines[-500:])
    report_lines.append('```\n' + excerpt + '\n```')
except Exception as e:
    report_lines.append(f"Could not read log file: {e}")

# Write report into the XXL run folder if available, else into outputs/ (timestamped)
report_text = '\n'.join(report_lines)
report_ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
if latest_xxl:
    report_path = latest_xxl / f'raptor_bericht_{report_ts}.md'
    # also update a latest copy
    report_latest = latest_xxl / 'raptor_bericht.md'
else:
    report_path = ROOT / 'outputs' / f'raptor_bericht_{report_ts}.md'
    report_latest = ROOT / 'outputs' / 'raptor_bericht.md'
report_path.write_text(report_text)
# update latest copy convenience file
report_latest.write_text(report_text)
print(f"Wrote report to: {report_path} (latest copy: {report_latest})")

# Also copy the ACTIVE log into the run folder for completeness (versioned)
try:
    if latest_xxl and ACTIVE_LOG.exists():
        copy_ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dest_log = latest_xxl / f'XXL_FULL_RUN_{copy_ts}.log'
        shutil.copy(ACTIVE_LOG, dest_log)
        print(f'Copied full log into run folder: {dest_log}')
except Exception:
    pass

print('Monitor finished')

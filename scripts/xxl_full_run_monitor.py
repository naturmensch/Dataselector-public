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
import argparse
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
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
# Ensure ROOT is in sys.path so we can import scripts modules directly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_FILE = ROOT / 'outputs' / 'XXL_FULL_RUN.log'
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

MAIN_SCRIPT = ROOT / 'scripts' / 'xxl_KDR146_run_thesis_complete.py'

# Helper to print and append monitor messages to active log
def _monitor_log(msg: str, active_log: Path) -> None:
    tsmsg = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(tsmsg)
    try:
        with open(active_log, 'a') as _lf:
            _lf.write(tsmsg + "\n")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Monitor XXL pipeline run")
    parser.add_argument('--no-new-session', action='store_true', help="Do not start child in new session (better for tests)")
    parser.add_argument('--poll-interval', type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

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
        # start_new_session=True creates a new process group, allowing us to kill the whole tree.
        # We disable this for tests via --no-new-session to allow standard signal propagation.
        start_new_session = not args.no_new_session
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=start_new_session)

    start_time = time.time()
    print(f"PID: {process.pid}, writing logs to: {LOG_FILE}")

    # PID Bookkeeping
    pid_file = LOG_FILE_TS.with_suffix('.pid')
    pgid = None
    try:
        # Use getpgid if available, otherwise just PID
        pgid = os.getpgid(process.pid) if hasattr(os, 'getpgid') else process.pid
        pid_file.write_text(f"PID={process.pid}\nPGID={pgid}\n")
    except Exception as e:
        print(f"Warning: could not write PID file: {e}")

    # Monitoring loop
    last_size = 0
    phase_events = []
    detected_xxl = None
    seen_lines = deque(maxlen=5000)  # Keep recent history to avoid memory leak

    # State for stability check of trials.csv
    trials_csv_candidate = None
    trials_csv_last_size = -1

    try:
        while True:
            time.sleep(args.poll_interval)  # poll interval
            # Check if process has exited
            ret = process.poll()
            # Read newly appended log content only from the ACTIVE log file
            try:
                with open(ACTIVE_LOG, 'r') as f:
                    # Handle log rotation/truncation
                    f.seek(0, os.SEEK_END)
                    if f.tell() < last_size:
                        last_size = 0
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
                    seen_lines.append(line)

                    # Detect running tasks (e.g., "Running 1/20: alpha=...")
                    m = re.search(r'Running\s+\d+/\d+:\s*(.+)', line)
                    if m:
                        _monitor_log(f"Starte: {m.group(1)}", ACTIVE_LOG)
                        continue

                    # Detect export completions
                    m = re.search(r'Auswahl exportiert nach:\s*(.+)$', line)
                    if m:
                        _monitor_log(f"Prozess beendet: Auswahl exportiert nach: {m.group(1)}", ACTIVE_LOG)
                        continue

                    m = re.search(r'Saved results:\s*(.+)\s*\(csv', line)
                    if m:
                        _monitor_log(f"Prozess beendet: Saved results: {m.group(1)}", ACTIVE_LOG)
                        continue

                    # Phase completions (localized messages)
                    if 'Phase 1 ABGESCHLOSSEN' in line or 'PHASE 1 COMPLETE' in line:
                        if 'PHASE 1 COMPLETE' not in phase_events:
                            phase_events.append('PHASE 1 COMPLETE')
                            _monitor_log('OBSERVED: PHASE 1 COMPLETE', ACTIVE_LOG)
                        continue
                    if 'Phase 2 COMPLETE' in line:
                        if 'PHASE 2 COMPLETE' not in phase_events:
                            phase_events.append('PHASE 2 COMPLETE')
                            _monitor_log('OBSERVED: PHASE 2 COMPLETE', ACTIVE_LOG)
                        continue
                    if 'Phase 3 COMPLETE' in line:
                        if 'PHASE 3 COMPLETE' not in phase_events:
                            phase_events.append('PHASE 3 COMPLETE')
                            _monitor_log('OBSERVED: PHASE 3 COMPLETE', ACTIVE_LOG)
                        continue
                    if 'Phase 4 COMPLETE' in line:
                        if 'PHASE 4 COMPLETE' not in phase_events:
                            phase_events.append('PHASE 4 COMPLETE')
                            _monitor_log('OBSERVED: PHASE 4 COMPLETE', ACTIVE_LOG)
                        continue

            # Detect creation of XXXL run folder
            xxl_dirs = sorted(glob.glob(str(ROOT / 'outputs' / 'runs' / '*hamburg_xxl_final*')))
            if xxl_dirs:
                latest_xxl = Path(xxl_dirs[-1])
                trials_csv = latest_xxl / 'results' / 'trials.csv'
                
                if trials_csv.exists() and trials_csv.stat().st_size > 0:
                    current_size = trials_csv.stat().st_size
                    
                    # Check stability: size must be stable across one poll interval
                    if trials_csv == trials_csv_candidate:
                        if current_size == trials_csv_last_size:
                            # Stable!
                            if latest_xxl != detected_xxl:
                                detected_xxl = latest_xxl
                                _monitor_log(f"Detected stable XXL run directory: {latest_xxl}", ACTIVE_LOG)
                        else:
                            trials_csv_last_size = current_size
                    else:
                        trials_csv_candidate = trials_csv
                        trials_csv_last_size = current_size

            if ret is not None:
                print(f"Process exited with code: {ret}")
                break

    except KeyboardInterrupt:
        print('KeyboardInterrupt received: terminating child process group')
        # Robust shutdown sequence: SIGTERM -> wait -> SIGKILL
        try:
            if start_new_session:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Timeout expired, forcing kill (SIGKILL)...")
            try:
                if start_new_session:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
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
    report_lines.append(f"**Process Info**:")
    report_lines.append(f"- PID: {process.pid}")
    report_lines.append(f"- PGID: {pgid if pgid is not None else 'N/A'}")
    report_lines.append(f"- Log: `{ACTIVE_LOG}`\n")

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

    # Write report into the monitor_reports folder under the XXL run dir if available, else into outputs/monitor_reports (timestamped)
    report_text = '\n'.join(report_lines)
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

    # Write markdown report
    report_md.write_text(report_text)
    # Write machine-readable metadata
    # Coerce meta values to basic types for safe JSON serialization
    meta = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'pid': int(process.pid) if hasattr(process, '__dict__') and isinstance(getattr(process, 'pid', None), int) else (int(process.pid) if hasattr(process.pid, '__int__') else str(process.pid)),
        'pgid': (int(pgid) if isinstance(pgid, int) else (int(pgid) if hasattr(pgid, '__int__') else (str(pgid) if pgid is not None else None))),
        'start_time': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        'end_time': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        'elapsed_sec': float(elapsed) if elapsed is not None else None,
        'exit_code': int(exit_code) if exit_code is not None else None,
        'observed_phase_events': [str(e) for e in phase_events],
        'xxl_run_dir': str(latest_xxl) if latest_xxl else None,
    }
    try:
        report_meta.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        _monitor_log(f"Warning: could not write report meta: {e}", ACTIVE_LOG)

    # update latest copies convenience files
    report_latest_md.write_text(report_text)
    try:
        report_latest_meta.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        _monitor_log(f"Warning: could not write latest meta: {e}", ACTIVE_LOG)

    print(f"Wrote report to: {report_md} (latest copies: {report_latest_md}, {report_latest_meta})")

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

if __name__ == "__main__":
    main()

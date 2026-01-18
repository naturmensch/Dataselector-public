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

def run_hook(name: str, cmd_str: str, base_log_dir: Path, active_log: Path, timeout: int, retries: int, env: dict, start_new_session: bool, pass_dry_run: bool) -> dict:
    """Run a hook command with retries, timeout and logging."""
    if pass_dry_run:
        cmd_str += " --dry-run"
    
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_file = base_log_dir / f'{name}_{ts}.log'
    
    meta = {
        'name': name,
        'command': cmd_str,
        'log_file': str(log_file),
        'attempts': [],
        'success': False
    }
    
    # If the command starts with a plain `python` executable name, replace it with
    # the exact interpreter used to run the monitor (sys.executable). This ensures
    # hooks run in the same environment as the monitor.
    import re, shlex
    rewritten = re.sub(r"^\s*(python3?|py)\b", shlex.quote(sys.executable), cmd_str, count=1)
    if rewritten != cmd_str:
        _monitor_log(f"[{name}] Rewrote hook command to use sys.executable: {rewritten}", active_log)
        cmd_str = rewritten

    _monitor_log(f"[{name}] Running hook: {cmd_str}", active_log)
    _monitor_log(f"[{name}] Log: {log_file}", active_log)

    for i in range(retries + 1):
        attempt_info = {
            'attempt': i + 1,
            'start': datetime.now(timezone.utc).isoformat(),
            'exit_code': None
        }
        
        try:
            with open(log_file, 'a') as f:
                # Use shell=True for flexibility with complex commands passed as string
                proc = subprocess.Popen(cmd_str, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      env=env, start_new_session=start_new_session)
                
                try:
                    # Treat timeout <= 0 as None (infinite wait)
                    wait_arg = timeout if timeout > 0 else None
                    ret = proc.wait(timeout=wait_arg)
                    attempt_info['exit_code'] = ret
                    if ret == 0:
                        meta['success'] = True
                except subprocess.TimeoutExpired:
                    _monitor_log(f"[{name}] Timeout ({timeout}s) expired, killing...", active_log)
                    attempt_info['error'] = 'timeout'
                    # Kill process group
                    if start_new_session and hasattr(os, 'killpg'):
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
        except Exception as e:
            attempt_info['error'] = str(e)
            _monitor_log(f"[{name}] Exception: {e}", active_log)
        
        attempt_info['end'] = datetime.now(timezone.utc).isoformat()
        meta['attempts'].append(attempt_info)
        
        if meta['success']:
            break
        
        if i < retries:
            _monitor_log(f"[{name}] Retrying ({i+1}/{retries})...", active_log)
            time.sleep(2)

    # Write hook meta
    meta_file = base_log_dir / f'{name}_meta_{ts}.json'
    try:
        meta_file.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass
        
    return meta

def main():
    parser = argparse.ArgumentParser(description="Monitor XXL pipeline run")
    parser.add_argument('--no-new-session', action='store_true', help="Do not start child in new session (better for tests)")
    parser.add_argument('--poll-interval', type=int, default=30, help="Polling interval in seconds")
    parser.add_argument('--child-dry-run', action='store_true', help='Pass --dry-run to the child orchestrator for fast smoke tests')
    
    # Hook arguments
    parser.add_argument('--pre-run-cmd', type=str, help="Command to run before the orchestrator")
    parser.add_argument('--post-run-cmd', type=str, help="Command to run after the orchestrator")
    parser.add_argument('--pre-run-timeout', type=int, default=600, help="Timeout for pre-run command in seconds (0 = no timeout)")
    parser.add_argument('--pre-run-delay', type=int, default=0, help="Delay in seconds after pre-run hook before starting main run")
    parser.add_argument('--pre-run-retries', type=int, default=0, help="Retries for pre-run command")
    parser.add_argument('--pre-run-fail-mode', choices=['abort', 'warn', 'continue'], default='abort', help="Action on pre-run failure")
    parser.add_argument('--pre-run-dry-run', action='store_true', help="Pass --dry-run to pre-run command if present")
    
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

    # 1. Run Pre-Run Hook
    pre_run_meta = None
    if args.pre_run_cmd:
        _monitor_log("Starting PRE-RUN hook...", ACTIVE_LOG)
        pre_run_meta = run_hook(
            name="pre_run",
            cmd_str=args.pre_run_cmd,
            base_log_dir=LOG_FILE.parent,
            active_log=ACTIVE_LOG,
            timeout=args.pre_run_timeout,
            retries=args.pre_run_retries,
            env=env,
            start_new_session=not args.no_new_session,
            pass_dry_run=args.pre_run_dry_run
        )
        if not pre_run_meta['success']:
            msg = f"PRE-RUN hook failed (exit code: {pre_run_meta['attempts'][-1].get('exit_code')})"
            _monitor_log(msg, ACTIVE_LOG)
            
            # Print log excerpt for debugging
            try:
                log_path = Path(pre_run_meta['log_file'])
                if log_path.exists():
                    _monitor_log(f"--- Log excerpt from {log_path.name} ---", ACTIVE_LOG)
                    _monitor_log(log_path.read_text(errors='replace')[-2000:], ACTIVE_LOG)
                    _monitor_log("-------------------------------------------", ACTIVE_LOG)
            except Exception as e:
                _monitor_log(f"Could not read hook log: {e}", ACTIVE_LOG)

            if args.pre_run_fail_mode == 'abort':
                _monitor_log("Aborting run due to pre-run failure.", ACTIVE_LOG)
                sys.exit(1)
            else:
                _monitor_log(f"Continuing despite pre-run failure (mode: {args.pre_run_fail_mode})", ACTIVE_LOG)

        if args.pre_run_delay > 0:
            _monitor_log(f"Waiting {args.pre_run_delay}s before starting main run...", ACTIVE_LOG)
            time.sleep(args.pre_run_delay)

    # Start the full run subprocess
    cmd = [sys.executable, str(MAIN_SCRIPT)]
    # Allow monitor to instruct child to run in dry-run mode for smoke tests
    if args.child_dry_run:
        cmd.append('--dry-run')
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
        # Use getpgid if available, otherwise just use PID as best-effort
        try:
            pgid = os.getpgid(process.pid) if hasattr(os, 'getpgid') else process.pid
        except Exception:
            pgid = None
        # write file with available information
        pid_text = f"PID={process.pid}\n"
        pid_text += f"PGID={pgid}\n" if pgid is not None else "PGID=N/A\n"
        pid_file.write_text(pid_text)
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
                # Prefer using the previously-determined PGID, but fall back to querying
                target_pgid = pgid
                if target_pgid is None:
                    try:
                        target_pgid = os.getpgid(process.pid)
                    except Exception:
                        target_pgid = None
                if target_pgid is not None:
                    try:
                        os.killpg(int(target_pgid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    # Fallback to terminating the single process
                    try:
                        process.terminate()
                    except Exception:
                        pass
            else:
                process.terminate()
        except Exception:
            pass
        
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Timeout expired, forcing kill (SIGKILL)...")
            try:
                if start_new_session:
                    target_pgid = pgid
                    if target_pgid is None:
                        try:
                            target_pgid = os.getpgid(process.pid)
                        except Exception:
                            target_pgid = None
                    if target_pgid is not None:
                        try:
                            os.killpg(int(target_pgid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        try:
                            process.kill()
                        except Exception:
                            pass
                else:
                    process.kill()
            except Exception:
                pass
        ret = process.returncode

    end_time = time.time()

    # 2. Run Post-Run Hook
    post_run_meta = None
    if args.post_run_cmd:
        _monitor_log("Starting POST-RUN hook...", ACTIVE_LOG)
        post_run_meta = run_hook(
            name="post_run",
            cmd_str=args.post_run_cmd,
            base_log_dir=LOG_FILE.parent,
            active_log=ACTIVE_LOG,
            timeout=600, # Default timeout for post-run
            retries=0,
            env=env,
            start_new_session=not args.no_new_session,
            pass_dry_run=args.pre_run_dry_run # Reuse dry-run flag logic
        )

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

    # Basic config validation for the discovered XXL run (if any)
    config_issues = []
    if latest_xxl:
        cfg_path = latest_xxl / 'config' / 'config_optuna.yaml'
        if cfg_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                sampler = cfg.get('sampler')
                n_trials_cfg = cfg.get('n_trials')
                n_candidates_cfg = cfg.get('n_candidates')

                if sampler and str(sampler).lower() != 'cmaes':
                    config_issues.append(f"unexpected sampler: {sampler}")

                try:
                    if n_trials_cfg is not None and int(n_trials_cfg) < 400:
                        config_issues.append(f"n_trials too small: {n_trials_cfg}")
                except Exception:
                    config_issues.append(f"n_trials not parseable: {n_trials_cfg}")

                try:
                    if n_candidates_cfg is not None and int(n_candidates_cfg) != 673:
                        config_issues.append(f"n_candidates mismatch: {n_candidates_cfg}")
                except Exception:
                    config_issues.append(f"n_candidates not parseable: {n_candidates_cfg}")
            except Exception as e:
                config_issues.append(f"failed to parse config: {e}")

    if config_issues:
        report_lines.append('\n## Configuration issues detected')
        for issue in config_issues:
            report_lines.append(f"- {issue}")

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
        'pid': int(process.pid) if getattr(process, 'pid', None) is not None else None,
        'pgid': int(pgid) if isinstance(pgid, int) else None,
        'start_time': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        'end_time': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        'elapsed_sec': float(elapsed),
        'exit_code': int(exit_code) if exit_code is not None else None,
        'observed_phase_events': [str(e) for e in phase_events],
        'xxl_run_dir': str(latest_xxl) if latest_xxl else None,
        'config_issues': config_issues if config_issues else [],
        'pre_run': pre_run_meta,
        'post_run': post_run_meta,
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

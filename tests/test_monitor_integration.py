import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
import scripts.xxl_full_run_monitor as monitor


def test_monitor_loop_detects_phases_and_writes_report(monkeypatch, tmp_path):
    """Integration-style test: simulate child run and log lines for phases; expect a report."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--poll-interval', '0', '--no-new-session'])
    monkeypatch.setattr(monitor, 'LOG_FILE', tmp_path / 'XXL_FULL_RUN.log')
    monkeypatch.setattr(monitor, 'ROOT', tmp_path)

    # Prepare a XXL run dir that should be discovered
    run_dir = tmp_path / 'outputs' / 'runs' / '20260119_T124819_thesis_xxl_hamburg_final'
    (run_dir / 'results').mkdir(parents=True)
    (run_dir / 'results' / 'trials.csv').write_text('trial_number,state,value\n0,TrialState.COMPLETE,1.0')

    mock_popen = MagicMock()
    mock_popen.pid = 12345
    # Simulate child running for two poll intervals and then exiting
    mock_popen.poll.side_effect = [None, None, 0]
    mock_popen.returncode = 0

    sleep_calls = {'n': 0}

    def sleep_side_effect(seconds):
        # On first sleep, write Phase 1 line; on second, write Phase 2; then nothing
        n = sleep_calls['n']
        if n == 0:
            monitor.LOG_FILE.write_text('Phase 1 COMPLETE\n')
        elif n == 1:
            # append
            monitor.LOG_FILE.write_text(monitor.LOG_FILE.read_text() + '\nPhase 2 COMPLETE\n')
        sleep_calls['n'] += 1

    with patch('subprocess.Popen', return_value=mock_popen), \
         patch('os.getpgid', return_value=12345), \
         patch('time.sleep', side_effect=sleep_side_effect):
        # Run monitor (should complete quickly due to mocked poll returning 0)
        monitor.main()

    # After run, report should be written inside the run dir monitor_reports
    reports_dir = run_dir / 'monitor_reports'
    assert reports_dir.exists(), f"reports dir not found: {reports_dir}"

    # There should be a latest meta file
    latest_meta = reports_dir / 'monitor_meta.json'
    assert latest_meta.exists(), "latest meta file not found"

    meta = json.loads(latest_meta.read_text())
    assert 'observed_phase_events' in meta
    assert 'PHASE 1 COMPLETE' in meta['observed_phase_events'] or 'PHASE 2 COMPLETE' in meta['observed_phase_events']

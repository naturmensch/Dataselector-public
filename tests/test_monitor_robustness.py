import sys
import os
import time
import pytest
import subprocess
from unittest.mock import MagicMock, patch
from pathlib import Path
import json

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.xxl_full_run_monitor as monitor

def test_monitor_args_no_new_session(monkeypatch, tmp_path):
    """Test that --no-new-session flag sets start_new_session=False."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--no-new-session', '--poll-interval', '1'])
    monkeypatch.setattr(monitor, 'LOG_FILE', tmp_path / 'XXL_FULL_RUN.log')
    
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    mock_popen.poll.return_value = 0
    
    with patch('subprocess.Popen', return_value=mock_popen) as mock_subprocess, \
         patch('os.getpgid', return_value=12345):
        
        monitor.main()
        
        args, kwargs = mock_subprocess.call_args
        assert kwargs.get('start_new_session') is False

def test_monitor_pid_file_creation(monkeypatch, tmp_path):
    """Test that PID file is created."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--poll-interval', '1'])
    
    mock_popen = MagicMock()
    mock_popen.pid = 9999
    mock_popen.poll.side_effect = [None, 0]
    
    # Mock datetime for predictable filename
    with patch('scripts.xxl_full_run_monitor.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "TEST_TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"
        
        # Point LOG_FILE and ROOT to tmp_path so monitor writes reports locally
        monkeypatch.setattr(monitor, 'LOG_FILE', tmp_path / 'XXL_FULL_RUN.log')
        monkeypatch.setattr(monitor, 'ROOT', tmp_path)
        
        with patch('subprocess.Popen', return_value=mock_popen), \
             patch('time.sleep'), \
             patch('os.getpgid', return_value=12345):
            
            monitor.main()
            
            pid_file = tmp_path / "XXL_FULL_RUN_TEST_TS.pid"
            assert pid_file.exists()
            content = pid_file.read_text()
            assert "PID=9999" in content
            assert "PGID=12345" in content

            # Also check that monitor wrote a machine-readable meta file
            reports_dir = tmp_path / 'outputs' / 'monitor_reports'
            meta_file = reports_dir / 'monitor_meta_TEST_TS.json'
            assert meta_file.exists()
            meta = json.loads(meta_file.read_text())
            assert meta.get('pid') == 9999
            assert meta.get('pgid') == 12345

def test_monitor_trials_stability_logic(monkeypatch, tmp_path):
    """Test that monitor waits for stable trials.csv size."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--poll-interval', '0'])
    
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    # Run loop: unstable -> unstable -> stable -> exit
    mock_popen.poll.side_effect = [None, None, None, 0] 
    
    run_dir = tmp_path / "outputs" / "runs" / "hamburg_xxl_final"
    run_dir.mkdir(parents=True)
    trials_csv = run_dir / "results" / "trials.csv"
    trials_csv.parent.mkdir()
    trials_csv.touch()
    
    def sleep_side_effect(seconds):
        # Simulate file growth
        current_size = trials_csv.stat().st_size
        if current_size == 0:
            trials_csv.write_text("x"*100) # Size 100
        elif current_size == 100:
            trials_csv.write_text("x"*200) # Size 200
        # Next time it stays 200 (stable)
    
    with patch('subprocess.Popen', return_value=mock_popen), \
         patch('time.sleep', side_effect=sleep_side_effect), \
         patch('glob.glob', return_value=[str(run_dir)]), \
         patch('scripts.xxl_full_run_monitor._monitor_log') as mock_log, \
         patch('scripts.xxl_full_run_monitor.datetime') as mock_dt, \
         patch('os.getpgid', return_value=12345):
            
            mock_dt.now.return_value.strftime.return_value = "TS"
            mock_dt.now.return_value.isoformat.return_value = "ISO"
            mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"
            
            monkeypatch.setattr(monitor, 'LOG_FILE', tmp_path / 'XXL_FULL_RUN.log')
            
            monitor.main()
            
            # Check logs for stability message
            log_calls = [args[0] for args, _ in mock_log.call_args_list]
            stable_msgs = [m for m in log_calls if "Detected stable XXL run directory" in m]
            assert len(stable_msgs) == 1

def test_monitor_sigkill_fallback(monkeypatch, tmp_path):
    """Test that monitor escalates to SIGKILL if SIGTERM times out."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--poll-interval', '1'])
    monkeypatch.setattr(monitor, 'LOG_FILE', tmp_path / 'XXL_FULL_RUN.log')
    
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    # Simulate process running forever
    mock_popen.poll.return_value = None
    # Simulate wait timeout
    mock_popen.wait.side_effect = [subprocess.TimeoutExpired(cmd='test', timeout=30), 0]
    
    with patch('subprocess.Popen', return_value=mock_popen), \
         patch('os.killpg') as mock_killpg, \
         patch('os.getpgid', return_value=12345), \
         patch('time.sleep', side_effect=KeyboardInterrupt): # Trigger interrupt immediately
        
        monitor.main()
        
        # Check that both signals were sent
        import signal
        calls = mock_killpg.call_args_list
        assert len(calls) >= 2
        assert calls[0][0][1] == signal.SIGTERM
        assert calls[1][0][1] == signal.SIGKILL

def test_monitor_log_truncation_handling(monkeypatch, tmp_path):
    """Test that monitor handles log file truncation (rotation) gracefully."""
    monkeypatch.setattr(sys, 'argv', ['monitor.py', '--poll-interval', '0'])
    log_file = tmp_path / 'XXL_FULL_RUN.log'
    monkeypatch.setattr(monitor, 'LOG_FILE', log_file)
    
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    # Run loop: normal -> truncated -> exit
    mock_popen.poll.side_effect = [None, None, 0]
    
    def sleep_side_effect(seconds):
        # Simulate log file changes
        if not log_file.exists():
            log_file.write_text("Line 1\n")
        elif log_file.read_text() == "Line 1\n":
            # Truncate file (simulate rotation)
            log_file.write_text("New Start\n")
    
    with patch('subprocess.Popen', return_value=mock_popen), \
         patch('time.sleep', side_effect=sleep_side_effect), \
         patch('scripts.xxl_full_run_monitor._monitor_log') as mock_log, \
         patch('scripts.xxl_full_run_monitor.datetime') as mock_dt, \
         patch('os.getpgid', return_value=12345):
            
            mock_dt.now.return_value.strftime.return_value = "TS"
            mock_dt.now.return_value.isoformat.return_value = "ISO"
            mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"
            
            # Should not raise exception
            monitor.main()
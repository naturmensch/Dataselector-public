import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

optuna = pytest.importorskip("optuna")

# Load monitor script without modifying sys.path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "xxl_full_run_monitor", ROOT / "scripts" / "xxl_full_run_monitor.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
# Also register module under package-style name expected by tests/patches
sys.modules.setdefault("scripts", types.ModuleType("scripts"))
sys.modules["scripts"].xxl_full_run_monitor = monitor
sys.modules["scripts.xxl_full_run_monitor"] = monitor


def test_monitor_args_no_new_session(monkeypatch, tmp_path):
    """Test that --no-new-session flag sets start_new_session=False."""
    monkeypatch.setattr(
        sys, "argv", ["monitor.py", "--no-new-session", "--poll-interval", "1"]
    )
    monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")

    mock_popen = MagicMock()
    mock_popen.pid = 12345
    mock_popen.poll.return_value = 0

    with (
        patch("subprocess.Popen", return_value=mock_popen) as mock_subprocess,
        patch("os.getpgid", return_value=12345),
    ):
        monitor.main()

        args, kwargs = mock_subprocess.call_args
        assert kwargs.get("start_new_session") is False


def test_monitor_pid_file_creation(monkeypatch, tmp_path):
    """Test that PID file is created."""
    """Test that PID file is created."""
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--poll-interval", "1"])

    mock_popen = MagicMock()
    mock_popen.pid = 9999
    mock_popen.poll.side_effect = [None, 0]

    # Mock datetime for predictable filename
    with patch("scripts.xxl_full_run_monitor.datetime") as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "TEST_TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        # Point LOG_FILE and ROOT to tmp_path so monitor writes reports locally
        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        monkeypatch.setattr(monitor, "ROOT", tmp_path)
        # Force deterministic timestamp for this test
        monkeypatch.setenv("MONITOR_FORCE_TS", "TEST_TS")

        with (
            patch("subprocess.Popen", return_value=mock_popen),
            patch("time.sleep"),
            patch("os.getpgid", return_value=12345),
        ):
            monitor.main()

            pid_file = tmp_path / "XXL_FULL_RUN_TEST_TS.pid"
            assert pid_file.exists()
            content = pid_file.read_text()
            assert "PID=9999" in content
            assert "PGID=12345" in content

            # Also check that monitor wrote a machine-readable meta file
            reports_dir = tmp_path / "outputs" / "monitor_reports"
            meta_file = reports_dir / "monitor_meta_TEST_TS.json"
            assert meta_file.exists()
            meta = json.loads(meta_file.read_text())
            assert meta.get("pid") == 9999
            assert meta.get("pgid") == 12345


def test_monitor_trials_stability_logic(monkeypatch, tmp_path):
    """Test that monitor waits for stable trials.csv size."""
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--poll-interval", "0"])

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
            trials_csv.write_text("x" * 100)  # Size 100
        elif current_size == 100:
            trials_csv.write_text("x" * 200)  # Size 200
        # Next time it stays 200 (stable)

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("time.sleep", side_effect=sleep_side_effect),
        patch("glob.glob", return_value=[str(run_dir)]),
        patch("scripts.xxl_full_run_monitor._monitor_log") as mock_log,
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=12345),
    ):
        mock_dt.now.return_value.strftime.return_value = "TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        # Force deterministic timestamp for this test
        monkeypatch.setenv("MONITOR_FORCE_TS", "TS")

        monitor.main()

        # Check logs for stability message: accept either a direct _monitor_log call or
        # presence in the active log file (some code paths write via logger)
        log_calls = [args[0] for args, _ in mock_log.call_args_list]
        stable_msgs = [m for m in log_calls if "Detected stable XXL run directory" in m]
        if not stable_msgs:
            # fallback: check in the saved active log file
            assert (tmp_path / "XXL_FULL_RUN.log").exists()
            txt = (tmp_path / "XXL_FULL_RUN.log").read_text()
            assert "Detected stable XXL run directory" in txt
        else:
            assert len(stable_msgs) == 1
    mock_popen = MagicMock()
    mock_popen.pid = 12345
    # Simulate process running forever
    mock_popen.poll.return_value = None
    # Simulate wait timeout
    mock_popen.wait.side_effect = [subprocess.TimeoutExpired(cmd="test", timeout=30), 0]

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("os.killpg") as mock_killpg,
        patch("os.getpgid", return_value=12345),
        patch("time.sleep", side_effect=KeyboardInterrupt),
    ):  # Trigger interrupt immediately
        monitor.main()

        # Check that both signals were sent
        import signal

        calls = mock_killpg.call_args_list
        assert len(calls) >= 2
        assert calls[0][0][1] == signal.SIGTERM
        assert calls[1][0][1] == signal.SIGKILL


def test_monitor_passes_child_dry_run(monkeypatch, tmp_path):
    """Test that monitor passes --dry-run to the child orchestrator when requested."""
    monkeypatch.setattr(
        sys, "argv", ["monitor.py", "--child-dry-run", "--poll-interval", "1"]
    )
    monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")

    mock_popen = MagicMock()
    mock_popen.pid = 1111
    mock_popen.poll.side_effect = [None, 0]

    with (
        patch("subprocess.Popen", return_value=mock_popen) as mock_subproc,
        patch("time.sleep"),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=1111),
    ):
        mock_dt.now.return_value.strftime.return_value = "TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monitor.main()
        # The first arg list passed to Popen should include the child script and the '--dry-run' flag
        args, kwargs = mock_subproc.call_args
        cmd_list = args[0]
        assert "--dry-run" in cmd_list


def test_run_hook_uses_monitor_python(monkeypatch, tmp_path):
    """If the hook command starts with `python`, it should be rewritten to use sys.executable."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor.py",
            "--pre-run-cmd",
            "python -c 'import sys; print(sys.executable)'",
            "--pre-run-fail-mode",
            "continue",
            "--poll-interval",
            "1",
        ],
    )
    monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")

    mock_popen = MagicMock()
    mock_popen.pid = 2222
    mock_popen.poll.side_effect = [None, 0]

    captured = {"calls": []}

    def fake_popen(*args, **kwargs):
        # record the raw cmd argument (first positional) or the cmd kw
        cmd = args[0] if args else kwargs.get("args")
        captured["calls"].append({"cmd": cmd, "args": args, "kwargs": kwargs})
        return mock_popen

    with (
        patch("subprocess.Popen", side_effect=fake_popen) as _mock_subproc,
        patch("time.sleep"),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=2222),
    ):
        mock_dt.now.return_value.strftime.return_value = "TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monitor.main()
        # Ensure at least one recorded call contains the absolute sys.executable path
        assert any(sys.executable in (c["cmd"] or "") for c in captured["calls"])


def test_monitor_log_truncation_handling(monkeypatch, tmp_path):
    """Test that monitor handles log file truncation (rotation) gracefully."""
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--poll-interval", "0"])
    log_file = tmp_path / "XXL_FULL_RUN.log"
    monkeypatch.setattr(monitor, "LOG_FILE", log_file)

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

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("time.sleep", side_effect=sleep_side_effect),
        patch("scripts.xxl_full_run_monitor._monitor_log") as _mock_log,
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=12345),
    ):
        mock_dt.now.return_value.strftime.return_value = "TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        # Should not raise exception
        monitor.main()


def test_monitor_config_validation(monkeypatch, tmp_path):
    """Test that monitor detects config issues in the discovered run."""
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--poll-interval", "0"])
    mock_popen = MagicMock()
    mock_popen.pid = 7777
    mock_popen.poll.side_effect = [None, 0]

    run_dir = tmp_path / "outputs" / "runs" / "hamburg_xxl_final"
    cfg_dir = run_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Write a config that intentionally mismatches expected values
    cfg = {"sampler": "tpe", "n_trials": 200, "n_candidates": 500}
    import yaml

    (cfg_dir / "config_optuna.yaml").write_text(yaml.safe_dump(cfg))

    # Create a minimal trials.csv so detection logic sees it
    trials_dir = run_dir / "results"
    trials_dir.mkdir(parents=True, exist_ok=True)
    (trials_dir / "trials.csv").write_text(
        "trial_number,value,state\n1,0.1,TrialState.COMPLETE"
    )

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("time.sleep"),
        patch("glob.glob", return_value=[str(run_dir)]),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=7777),
    ):
        mock_dt.now.return_value.strftime.return_value = "TEST_TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        monkeypatch.setattr(monitor, "ROOT", tmp_path)
        # Force deterministic timestamp for this test
        monkeypatch.setenv("MONITOR_FORCE_TS", "TEST_TS")

        monitor.main()

        run_reports_dir = run_dir / "monitor_reports"
        out_reports_dir = tmp_path / "outputs" / "monitor_reports"
        if (run_reports_dir / "monitor_meta_TEST_TS.json").exists():
            meta_file = run_reports_dir / "monitor_meta_TEST_TS.json"
        else:
            meta_file = out_reports_dir / "monitor_meta_TEST_TS.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text())
        assert any("n_trials too small" in s for s in meta["config_issues"])
        assert any("n_candidates mismatch" in s for s in meta["config_issues"])


def test_monitor_attempts_reconstruction(monkeypatch, tmp_path):
    """Test that monitor attempts to reconstruct missing trials.csv from optuna DB."""
    monkeypatch.setattr(sys, "argv", ["monitor.py", "--poll-interval", "0"])
    mock_popen = MagicMock()
    mock_popen.pid = 4444
    mock_popen.poll.side_effect = [None, 0]

    run_dir = tmp_path / "outputs" / "runs" / "hamburg_xxl_final"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal optuna DB with a few trials
    db_path = run_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize", storage=storage_url, study_name="recon_test"
    )

    def objective(trial):
        x = trial.suggest_float("a", 0.0, 1.0)
        return x

    for _ in range(4):
        study.optimize(objective, n_trials=1)

    # Ensure results dir exists but trials.csv is missing
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("time.sleep"),
        patch("glob.glob", return_value=[str(run_dir)]),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=4444),
    ):
        mock_dt.now.return_value.strftime.return_value = "TEST_TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        monkeypatch.setattr(monitor, "ROOT", tmp_path)

        monitor.main()

        # After running monitor, trials.csv should be reconstructed
        trials_csv = run_dir / "results" / "trials.csv"
        assert trials_csv.exists()
        content = trials_csv.read_text()
        assert "trial_number" in content
        assert "a" in content


def test_monitor_no_reconstruct_flag(monkeypatch, tmp_path):
    """Test that --no-reconstruct prevents reconstruction of missing trials.csv."""
    monkeypatch.setattr(
        sys, "argv", ["monitor.py", "--poll-interval", "0", "--no-reconstruct"]
    )
    mock_popen = MagicMock()
    mock_popen.pid = 5555
    mock_popen.poll.side_effect = [None, 0]

    run_dir = tmp_path / "outputs" / "runs" / "hamburg_xxl_final"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal optuna DB with a few trials
    db_path = run_dir / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize", storage=storage_url, study_name="recon_test_no"
    )

    def objective(trial):
        x = trial.suggest_float("a", 0.0, 1.0)
        return x

    for _ in range(2):
        study.optimize(objective, n_trials=1)

    # Ensure results dir exists but trials.csv is missing
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    with (
        patch("subprocess.Popen", return_value=mock_popen),
        patch("time.sleep"),
        patch("glob.glob", return_value=[str(run_dir)]),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
        patch("os.getpgid", return_value=5555),
    ):
        mock_dt.now.return_value.strftime.return_value = "TEST_TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        monkeypatch.setattr(monitor, "ROOT", tmp_path)

        monitor.main()

        # After running monitor with --no-reconstruct, trials.csv should still be missing
        trials_csv = run_dir / "results" / "trials.csv"
        assert not trials_csv.exists()


def test_monitor_hooks(monkeypatch, tmp_path):
    """Test that monitor executes pre-run and post-run hooks."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor.py",
            "--poll-interval",
            "0",
            "--pre-run-cmd",
            "echo pre",
            "--post-run-cmd",
            "echo post",
            "--pre-run-dry-run",
        ],
    )

    # Mock Popen to handle 3 calls: pre-run, main run, post-run
    mock_pre = MagicMock()
    mock_pre.wait.return_value = 0

    mock_main = MagicMock()
    mock_main.pid = 8888
    mock_main.poll.side_effect = [None, 0]

    mock_post = MagicMock()
    mock_post.wait.return_value = 0

    # We need side_effect to return different mocks for each call
    # 1. pre-run (shell=True)
    # 2. main run (list args)
    # 3. post-run (shell=True)

    with (
        patch(
            "subprocess.Popen", side_effect=[mock_pre, mock_main, mock_post]
        ) as mock_popen,
        patch("time.sleep"),
        patch("os.getpgid", return_value=8888),
        patch("scripts.xxl_full_run_monitor.datetime") as mock_dt,
    ):
        mock_dt.now.return_value.strftime.return_value = "TS"
        mock_dt.now.return_value.isoformat.return_value = "ISO"
        mock_dt.fromtimestamp.return_value.isoformat.return_value = "ISO"

        monkeypatch.setattr(monitor, "LOG_FILE", tmp_path / "XXL_FULL_RUN.log")
        monkeypatch.setattr(monitor, "ROOT", tmp_path)
        monkeypatch.setenv("MONITOR_FORCE_TS", "TS")

        monitor.main()

        # Verify calls
        assert mock_popen.call_count == 3
        # Check pre-run args (first call)
        args_pre, kwargs_pre = mock_popen.call_args_list[0]
        assert "echo pre" in args_pre[0]
        assert "--dry-run" in args_pre[0]  # passed via --pre-run-dry-run
        assert kwargs_pre["shell"] is True

        # Check main run args (second call)
        args_main, kwargs_main = mock_popen.call_args_list[1]
        assert isinstance(args_main[0], list)  # Main script is passed as list

        # Check post-run args (third call)
        args_post, kwargs_post = mock_popen.call_args_list[2]
        assert "echo post" in args_post[0]

        # Verify meta file contains hook info
        reports_dir = tmp_path / "outputs" / "monitor_reports"
        meta_file = reports_dir / "monitor_meta_TS.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text())
        assert meta["pre_run"]["success"] is True
        assert meta["post_run"]["success"] is True

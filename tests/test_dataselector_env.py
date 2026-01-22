import os
import shutil
from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
monitor = load_script(ROOT / "scripts" / "xxl_full_run_monitor.py", module_name="scripts.xxl_full_run_monitor_test")


def test_init_env_runner_prefers_mamba(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASELECTOR_ENV_NAME", "dataselector")
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/mamba" if name == "mamba" else None
    )
    monitor._init_env_runner(use_env=True, env_name="dataselector")
    assert monitor.ENV_RUNNER_CMD is not None
    assert "mamba run -n dataselector" in monitor.ENV_RUNNER_CMD
    assert monitor.ENV_RUNNER_LIST[0] == "mamba"


def test_run_hook_prefixes_with_env(monkeypatch, tmp_path):
    # pretend mamba is available
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/mamba" if name == "mamba" else None
    )
    monitor._init_env_runner(use_env=True, env_name="dataselector")

    # fake Popen to avoid actually running commands
    class FakeProc:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: FakeProc())

    meta = monitor.run_hook(
        name="test",
        cmd_str="echo hello",
        base_log_dir=tmp_path,
        active_log=tmp_path / "log.txt",
        timeout=1,
        retries=0,
        env=os.environ.copy(),
        start_new_session=False,
        pass_dry_run=False,
    )
    assert meta["command"].startswith("mamba run -n dataselector --")
    assert meta["success"] is True

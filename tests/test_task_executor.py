from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
recovery = load_script(ROOT / "scripts" / "recovery.py", module_name="scripts.recovery_test")
Task = recovery.Task
TaskExecutor = recovery.TaskExecutor


class FakeHookRunner:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        name,
        cmd_str,
        base_log_dir,
        active_log,
        timeout,
        retries,
        env,
        start_new_session,
        pass_dry_run,
    ):
        self.calls.append((name, cmd_str))
        return {"name": name, "command": cmd_str, "success": True}


class FakeCmdRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, retries=2, delay=5, cwd=None, fail_ok=False):
        self.calls.append(cmd)
        return 0


def test_executor_runs_tasks(tmp_path):
    hook = FakeHookRunner()
    cmd = FakeCmdRunner()

    exec = TaskExecutor(run_hook_func=hook, run_cmd_func=cmd)
    tasks = [
        Task(name="optuna", params={"n_trials": 5}),
        Task(name="repro", params={"seeds": [101]}),
        Task(name="finalize", params={"n_samples": 32}),
    ]
    _ = exec.execute(tasks, str(tmp_path))

    # optuna -> one cmd call
    assert any("optuna_optimize" in c for c in cmd.calls)
    # repro -> hook called for seed 101
    assert any(
        c[0] == "repro_s101" or "run_adaptive_pipeline" in c[1] for c in hook.calls
    )
    # finalize -> hook called
    assert any("finalize" in call[0] for call in hook.calls)

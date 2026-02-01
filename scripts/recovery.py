<<<<<<< HEAD
"""Simple recovery planner and task executor for XXL run monitor.

This module provides a lightweight, well-tested implementation of the
RecoveryPlanner and TaskExecutor used by `xxl_full_run_monitor.py`.

The goal is not to implement a feature-complete system but to provide a
minimal and predictable planner that can be extended later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import Any, Callable, Dict, List, Optional
=======
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
>>>>>>> chore/ci-lint-attrs-gdf


@dataclass
class Task:
<<<<<<< HEAD
    """Represents a planned recovery task.

    Attributes:
        name: short task name (e.g., 'reconstruct', 'optuna', 'repro', 'finalize')
        params: arbitrary parameters for the task
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


class RecoveryPlanner:
    """Plan recovery tasks based on observed run `state`.

    This planner is intentionally conservative and deterministic: it inspects
    the provided `state` dictionary (as assembled by the monitor) and
    returns a sequence of tasks required to resume/complete the run.

    The planner accepts an optional `configured_n` (target number of trials)
    and a list of `repro_seeds` to use for reproducibility phases.
    """

    def __init__(self, configured_n: Optional[int] = None, repro_seeds: Optional[List[int]] = None):
        self.configured_n = int(configured_n) if configured_n is not None else None
        self.repro_seeds = list(repro_seeds) if repro_seeds else []

    def plan(self, state: Dict[str, Any]) -> List[Task]:
        """Return an ordered list of `Task` objects to recover the run.

        Expected keys in `state` (not exhaustive):
            - csv_exists, csv_completed, db_exists, db_integrity_ok
            - repro_done, final_exists, n_samples
        """
        tasks: List[Task] = []

        # If DB exists but CSV does not, reconstruct from DB first
        if bool(state.get("db_exists", False)) and not bool(state.get("csv_exists", False)):
            tasks.append(Task("reconstruct", {}))

        # Schedule optuna to complete remaining trials when configured target is known
        csv_completed = int(state.get("csv_completed", 0) or 0)
        if self.configured_n is not None and csv_completed < self.configured_n:
            n_remaining = max(0, int(self.configured_n) - csv_completed)
            if n_remaining > 0:
                tasks.append(Task("optuna", {"n_trials": n_remaining}))

        # Reproducibility step if not done
        if not bool(state.get("repro_done", False)):
            tasks.append(Task("repro", {"seeds": list(self.repro_seeds)}))

        # Finalization step if the global final selection is missing
        if not bool(state.get("final_exists", False)):
            tasks.append(Task("finalize", {"n_samples": state.get("n_samples")}))
=======
    name: str
    params: Dict[str, Any]


class RecoveryPlanner:
    """Plan tasks to recover/complete a run based on observed state.

    The planner is intentionally simple and deterministic, designed to be
    easily unit tested. It does not perform filesystem or DB operations;
    it only decides which Tasks should be executed given a summarized state.
    """

    def __init__(
        self,
        configured_n: Optional[int] = None,
        repro_seeds: Optional[List[int]] = None,
    ):
        self.configured_n = configured_n
        self.repro_seeds = repro_seeds if repro_seeds is not None else [43, 44]

    def plan(self, state: Dict[str, Any]) -> List[Task]:
        tasks: List[Task] = []

        csv_exists = bool(state.get("csv_exists"))
        csv_completed = int(state.get("csv_completed") or 0)
        db_exists = bool(state.get("db_exists"))
        db_integrity_ok = bool(state.get("db_integrity_ok"))
        db_completed = int(state.get("db_completed") or 0)

        # 1) If DB exists and is valid and appears more complete than CSV -> reconstruct
        if (
            db_exists
            and db_integrity_ok
            and (not csv_exists or db_completed > csv_completed)
        ):
            tasks.append(
                Task(name="reconstruct", params={"source_db": state.get("db_path")})
            )
            # After reconstruct, we assume CSV will reflect DB; use db_completed as baseline
            completed_after = db_completed
        else:
            completed_after = csv_completed if csv_exists else db_completed

        # 2) If we know configured_n, compute remaining Optuna trials
        remaining = None
        if self.configured_n is not None:
            remaining = int(self.configured_n - completed_after)
            if remaining > 0:
                tasks.append(Task(name="optuna", params={"n_trials": remaining}))

        # 3) Reproducibility: if repro not done, schedule
        repro_done = bool(state.get("repro_done"))
        if not repro_done:
            tasks.append(Task(name="repro", params={"seeds": list(self.repro_seeds)}))

        # 4) Finalize: if final missing or belongs to different run, schedule
        final_exists = bool(state.get("final_exists"))
        final_belongs = bool(state.get("final_belongs"))
        if not final_exists or not final_belongs:
            tasks.append(
                Task(name="finalize", params={"n_samples": state.get("n_samples")})
            )
>>>>>>> chore/ci-lint-attrs-gdf

        return tasks


class TaskExecutor:
<<<<<<< HEAD
    """Execute planned tasks using provided hook/cmd functions.

    The monitor provides `run_hook` and `run_cmd_with_retry`. The executor
    implements a simple mapping from task -> command and runs it via
    the provided helpers. Execution results are returned in a list with one
    entry per task.
    """

    def __init__(self, run_hook_func: Optional[Callable] = None, run_cmd_func: Optional[Callable] = None):
        self.run_hook = run_hook_func
        self.run_cmd = run_cmd_func

    def execute(self, tasks: List[Task], run_dir: str) -> List[Any]:
        """Execute tasks sequentially and return per-task results.

        Results may be either integers (exit codes) or dicts with more
        detailed metadata, depending on the underlying command helpers.
        """
        results: List[Any] = []
        for t in tasks:
            try:
                if t.name == "reconstruct":
                    # Reconstruction is handled internally in the monitor; here we
                    # return a not-implemented indicator so the monitor can
                    # handle it deterministically after optuna runs when necessary.
                    results.append({"success": False, "reason": "reconstruct_not_implemented"})

                elif t.name == "optuna":
                    n = int(t.params.get("n_trials", 0))
                    cmd = f"{sys.executable} -m scripts.optuna_optimize --n-trials {n}"
                    if self.run_cmd:
                        rc = self.run_cmd(cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False)
                        results.append(rc)
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})

                elif t.name == "repro":
                    seeds = ",".join(str(s) for s in (t.params.get("seeds") or []))
                    cmd = f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase repro --seeds {seeds}"
                    if self.run_cmd:
                        rc = self.run_cmd(cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False)
                        results.append(rc)
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})

                elif t.name == "finalize":
                    cmd = f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase finalize --run-dir {run_dir}"
                    if t.params.get("n_samples"):
                        cmd += f" --n-samples {t.params.get('n_samples')}"
                    if self.run_cmd:
                        rc = self.run_cmd(cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False)
                        results.append(rc)
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})

                else:
                    results.append({"success": False, "reason": "unknown_task"})

            except Exception as e:  # pragma: no cover - defensive
                results.append({"success": False, "exception": str(e)})

        return results


__all__ = ["Task", "RecoveryPlanner", "TaskExecutor"]
=======
    """Execute planned Tasks using provided runner functions.

    The executor is dependency-injected with small callables to allow
    unit testing without spawning processes.
    """

    def __init__(self, run_hook_func, run_cmd_func):
        # run_hook(name, cmd_str, base_log_dir, active_log, timeout, retries, env, start_new_session, pass_dry_run)
        self.run_hook = run_hook_func
        self.run_cmd = run_cmd_func

    def execute(self, tasks: List[Task], run_dir: str) -> List[Dict[str, Any]]:
        results = []
        for t in tasks:
            if t.name == "reconstruct":
                # call monitor reconstruction hook: implemented as direct function/command externally
                res = self.run_hook(
                    name="resume_phase_reconstruct",
                    cmd_str="reconstruct",
                    base_log_dir=run_dir,
                    active_log=f"{run_dir}/logs/reconstruct.log",
                    timeout=600,
                    retries=0,
                    env=None,
                    start_new_session=False,
                    pass_dry_run=False,
                )
                results.append({"task": t.name, "meta": res})

            elif t.name == "optuna":
                n = int(t.params.get("n_trials", 0))
                cmd = f"python -m scripts.optuna_optimize --n-trials {n}"
                ret = self.run_cmd(cmd, retries=2, delay=5, cwd=None, fail_ok=False)
                results.append({"task": t.name, "exit_code": ret})

            elif t.name == "repro":
                seeds = t.params.get("seeds", [])
                # run repro seeds sequentially
                meta_list = []
                for s in seeds:
                    cmd_str = f"python -m scripts.run_adaptive_pipeline --seed {s} --hamburg --exp-name thesis_hamburg_reproducibility_s{s}"
                    meta = self.run_hook(
                        name=f"resume_phase_reproducibility_s{s}",
                        cmd_str=cmd_str,
                        base_log_dir=run_dir,
                        active_log=f"{run_dir}/logs/repro_s{s}.log",
                        timeout=3600,
                        retries=0,
                        env=None,
                        start_new_session=False,
                        pass_dry_run=False,
                    )
                    meta_list.append(meta)
                results.append({"task": t.name, "meta": meta_list})

            elif t.name == "finalize":
                # call finalization hook
                n_samples = t.params.get("n_samples")
                cmd = (
                    f"python -m scripts.final_selection --n-samples {n_samples}"
                    if n_samples
                    else "python -m scripts.final_selection"
                )
                meta = self.run_hook(
                    name="resume_phase_finalize",
                    cmd_str=cmd,
                    base_log_dir=run_dir,
                    active_log=f"{run_dir}/logs/finalize.log",
                    timeout=3600,
                    retries=0,
                    env=None,
                    start_new_session=False,
                    pass_dry_run=False,
                )
                results.append({"task": t.name, "meta": meta})

            else:
                results.append({"task": t.name, "error": "unknown task"})
        return results
>>>>>>> chore/ci-lint-attrs-gdf

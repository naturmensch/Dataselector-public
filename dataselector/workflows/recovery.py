"""Recovery planner and task executor for workflow-level resume logic."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Task:
    """Represents a planned recovery task."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


class RecoveryPlanner:
    """Plan recovery tasks based on observed run state."""

    def __init__(
        self,
        configured_n: Optional[int] = None,
        repro_seeds: Optional[List[int]] = None,
    ):
        self.configured_n = int(configured_n) if configured_n is not None else None
        self.repro_seeds = list(repro_seeds) if repro_seeds else []

    def plan(self, state: Dict[str, Any]) -> List[Task]:
        tasks: List[Task] = []

        if bool(state.get("db_exists", False)) and not bool(
            state.get("csv_exists", False)
        ):
            tasks.append(Task("reconstruct", {}))

        csv_completed = int(state.get("csv_completed", 0) or 0)
        if self.configured_n is not None and csv_completed < self.configured_n:
            n_remaining = max(0, int(self.configured_n) - csv_completed)
            if n_remaining > 0:
                tasks.append(Task("optuna", {"n_trials": n_remaining}))

        if not bool(state.get("repro_done", False)):
            tasks.append(Task("repro", {"seeds": list(self.repro_seeds)}))

        if not bool(state.get("final_exists", False)):
            tasks.append(Task("finalize", {"n_samples": state.get("n_samples")}))

        return tasks


class TaskExecutor:
    """Execute planned recovery tasks via injected hooks."""

    def __init__(
        self,
        run_hook_func: Optional[Callable] = None,
        run_cmd_func: Optional[Callable] = None,
    ):
        self.run_hook = run_hook_func
        self.run_cmd = run_cmd_func

    def execute(self, tasks: List[Task], run_dir: str) -> List[Any]:
        results: List[Any] = []
        for task in tasks:
            try:
                if task.name == "reconstruct":
                    results.append(
                        {"success": False, "reason": "reconstruct_not_implemented"}
                    )
                elif task.name == "optuna":
                    n_trials = int(task.params.get("n_trials", 0))
                    cmd = (
                        f"{sys.executable} -m dataselector "
                        f"optuna-optimize --n-trials {n_trials}"
                    )
                    if self.run_cmd:
                        results.append(
                            self.run_cmd(
                                cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False
                            )
                        )
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})
                elif task.name == "repro":
                    seeds = ",".join(str(s) for s in (task.params.get("seeds") or []))
                    cmd = (
                        f"{sys.executable} -m dataselector xxl "
                        f"--phase full --seed {seeds.split(',')[0] if seeds else 42}"
                    )
                    if self.run_cmd:
                        results.append(
                            self.run_cmd(
                                cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False
                            )
                        )
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})
                elif task.name == "finalize":
                    cmd = f"{sys.executable} -m dataselector xxl --phase finalize"
                    if self.run_cmd:
                        results.append(
                            self.run_cmd(
                                cmd, retries=0, delay=0, cwd=run_dir, fail_ok=False
                            )
                        )
                    else:
                        results.append({"success": False, "reason": "no_run_cmd"})
                else:
                    results.append({"success": False, "reason": "unknown_task"})
            except Exception as exc:  # pragma: no cover
                results.append({"success": False, "exception": str(exc)})

        return results


__all__ = ["Task", "RecoveryPlanner", "TaskExecutor"]

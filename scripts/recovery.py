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


@dataclass
class Task:
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

                return tasks


class TaskExecutor:
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

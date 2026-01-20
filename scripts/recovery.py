from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    name: str
    params: Dict[str, Any]


class RecoveryPlanner:
    """Plan tasks to recover/complete a run based on observed state.

    The planner is intentionally simple and deterministic, designed to be
    easily unit tested. It does not perform filesystem or DB operations;
    it only decides which Tasks should be executed given a summarized state.
    """

    def __init__(self, configured_n: Optional[int] = None, repro_seeds: Optional[List[int]] = None):
        self.configured_n = configured_n
        self.repro_seeds = repro_seeds if repro_seeds is not None else [43, 44]

    def plan(self, state: Dict[str, Any]) -> List[Task]:
        tasks: List[Task] = []

        csv_exists = bool(state.get('csv_exists'))
        csv_completed = int(state.get('csv_completed') or 0)
        db_exists = bool(state.get('db_exists'))
        db_integrity_ok = bool(state.get('db_integrity_ok'))
        db_completed = int(state.get('db_completed') or 0)

        # 1) If DB exists and is valid and appears more complete than CSV -> reconstruct
        if db_exists and db_integrity_ok and (not csv_exists or db_completed > csv_completed):
            tasks.append(Task(name='reconstruct', params={'source_db': state.get('db_path')}))
            # After reconstruct, we assume CSV will reflect DB; use db_completed as baseline
            completed_after = db_completed
        else:
            completed_after = csv_completed if csv_exists else db_completed

        # 2) If we know configured_n, compute remaining Optuna trials
        remaining = None
        if self.configured_n is not None:
            remaining = int(self.configured_n - completed_after)
            if remaining > 0:
                tasks.append(Task(name='optuna', params={'n_trials': remaining}))

        # 3) Reproducibility: if repro not done, schedule
        repro_done = bool(state.get('repro_done'))
        if not repro_done:
            tasks.append(Task(name='repro', params={'seeds': list(self.repro_seeds)}))

        # 4) Finalize: if final missing or belongs to different run, schedule
        final_exists = bool(state.get('final_exists'))
        final_belongs = bool(state.get('final_belongs'))
        if not final_exists or not final_belongs:
            tasks.append(Task(name='finalize', params={'n_samples': state.get('n_samples')}))

        return tasks


class TaskExecutor:
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
            if t.name == 'reconstruct':
                # call monitor reconstruction hook: implemented as direct function/command externally
                res = self.run_hook(name='resume_phase_reconstruct', cmd_str='reconstruct', base_log_dir=run_dir, active_log=f"{run_dir}/logs/reconstruct.log", timeout=600, retries=0, env=None, start_new_session=False, pass_dry_run=False)
                results.append({'task': t.name, 'meta': res})

            elif t.name == 'optuna':
                n = int(t.params.get('n_trials', 0))
                cmd = f"python -m scripts.optuna_optimize --n-trials {n}"
                ret = self.run_cmd(cmd, retries=2, delay=5, cwd=None, fail_ok=False)
                results.append({'task': t.name, 'exit_code': ret})

            elif t.name == 'repro':
                seeds = t.params.get('seeds', [])
                # run repro seeds sequentially
                meta_list = []
                for s in seeds:
                    cmd_str = f"python -m scripts.run_adaptive_pipeline --seed {s} --hamburg --exp-name thesis_hamburg_reproducibility_s{s}"
                    meta = self.run_hook(name=f'resume_phase_reproducibility_s{s}', cmd_str=cmd_str, base_log_dir=run_dir, active_log=f"{run_dir}/logs/repro_s{s}.log", timeout=3600, retries=0, env=None, start_new_session=False, pass_dry_run=False)
                    meta_list.append(meta)
                results.append({'task': t.name, 'meta': meta_list})

            elif t.name == 'finalize':
                # call finalization hook
                n_samples = t.params.get('n_samples')
                cmd = f"python -m scripts.final_selection --n-samples {n_samples}" if n_samples else f"python -m scripts.final_selection"
                meta = self.run_hook(name='resume_phase_finalize', cmd_str=cmd, base_log_dir=run_dir, active_log=f"{run_dir}/logs/finalize.log", timeout=3600, retries=0, env=None, start_new_session=False, pass_dry_run=False)
                results.append({'task': t.name, 'meta': meta})

            else:
                results.append({'task': t.name, 'error': 'unknown task'})
        return results

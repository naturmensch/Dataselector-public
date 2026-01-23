#!/usr/bin/env python3
"""Start full XXL pipeline run and monitor progress.

This script launches `scripts/xxl_KDR146_run_thesis_complete.py` as a subprocess,
streams stdout/stderr to `outputs/XXL_FULL_RUN.log`, monitors progress by
inspecting the log and `outputs/runs/` for the final XXL run directory, and
writes a monitor report `monitor_report.md` inside the discovered run dir
when the run finishes (success or failure).

Usage:
    PYTHONPATH=. python scripts/xxl_full_run_monitor.py

Be aware: this performs a full production execution and can take a long time.
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
# Do NOT modify sys.path at module import time; imports that require project root should
# be performed inside `main()` or by setting PYTHONPATH when running the script.
LOG_FILE = ROOT / "outputs" / "XXL_FULL_RUN.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Environment runner configuration (dataselector mamba/conda env integration)
USE_DATASELECTOR_ENV: bool = True  # default: use the dataselector environment
DATASELECTOR_ENV_NAME: str = os.environ.get("DATASELECTOR_ENV_NAME", "dataselector")
ENV_RUNNER_CMD: str | None = None  # e.g. 'mamba run -n dataselector --'
ENV_RUNNER_LIST: list[str] | None = (
    None  # e.g. ['mamba', 'run', '-n', 'dataselector', '--']
)
ENV_RUNNER_NAME: str | None = None

MAIN_SCRIPT = ROOT / "scripts" / "xxl_KDR146_run_thesis_complete.py"


# Helper to print and append monitor messages to active log
def _monitor_log(msg: str, active_log: Path) -> None:
    tsmsg = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(tsmsg)
    try:
        with open(active_log, "a") as _lf:
            _lf.write(tsmsg + "\n")
    except Exception:
        pass


def _init_env_runner(use_env: bool = True, env_name: str | None = None):
    """Initialize the ENV runner variables (mamba/conda). Default env name 'dataselector'.

    Sets global ENV_RUNNER_CMD and ENV_RUNNER_LIST.
    """
    global USE_DATASELECTOR_ENV, DATASELECTOR_ENV_NAME, ENV_RUNNER_CMD, ENV_RUNNER_LIST, ENV_RUNNER_NAME
    USE_DATASELECTOR_ENV = bool(use_env)
    if env_name:
        DATASELECTOR_ENV_NAME = env_name

    ENV_RUNNER_CMD = None
    ENV_RUNNER_LIST = None
    ENV_RUNNER_NAME = None

    if not USE_DATASELECTOR_ENV:
        return

    # Prefer mamba, fall back to conda
    try:
        if shutil.which("mamba"):
            ENV_RUNNER_CMD = f"mamba run -n {DATASELECTOR_ENV_NAME} --"
            ENV_RUNNER_LIST = ["mamba", "run", "-n", DATASELECTOR_ENV_NAME, "--"]
            ENV_RUNNER_NAME = "mamba"
        elif shutil.which("conda"):
            ENV_RUNNER_CMD = f"conda run -n {DATASELECTOR_ENV_NAME} --"
            ENV_RUNNER_LIST = ["conda", "run", "-n", DATASELECTOR_ENV_NAME, "--"]
            ENV_RUNNER_NAME = "conda"
        else:
            _monitor_log(
                "Warning: no mamba/conda found on PATH; will not use dataselector env",
                LOG_FILE,
            )
            USE_DATASELECTOR_ENV = False
    except Exception as e:
        _monitor_log(f"Warning: error detecting mamba/conda: {e}", LOG_FILE)
        USE_DATASELECTOR_ENV = False


def run_hook(
    name: str,
    cmd_str: str,
    base_log_dir: Path,
    active_log: Path,
    timeout: int,
    retries: int,
    env: dict,
    start_new_session: bool,
    pass_dry_run: bool,
) -> dict:
    """Run a hook command with retries, timeout and logging."""
    if pass_dry_run:
        cmd_str += " --dry-run"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = base_log_dir / f"{name}_{ts}.log"

    meta = {
        "name": name,
        "command": cmd_str,
        "log_file": str(log_file),
        "attempts": [],
        "success": False,
    }

    # If the command starts with a plain `python` executable name, replace it with
    # the exact interpreter used to run the monitor (sys.executable). This ensures
    # hooks run in the same environment as the monitor.
    import re
    import shlex

    rewritten = re.sub(
        r"^\s*(python3?|py)\b", shlex.quote(sys.executable), cmd_str, count=1
    )
    if rewritten != cmd_str:
        _monitor_log(
            f"[{name}] Rewrote hook command to use sys.executable: {rewritten}",
            active_log,
        )
        cmd_str = rewritten

    # If enabled and available, prefix the command with the env runner (mamba/conda)
    if USE_DATASELECTOR_ENV and ENV_RUNNER_CMD:
        _monitor_log(
            f"[{name}] Prefixing hook command to run inside env '{DATASELECTOR_ENV_NAME}' using {ENV_RUNNER_NAME}",
            active_log,
        )
        cmd_str = f"{ENV_RUNNER_CMD} {cmd_str}"

    # Record the actual command being executed (after any rewriting/prefixing)
    meta["command"] = cmd_str
    _monitor_log(f"[{name}] Running hook: {cmd_str}", active_log)
    _monitor_log(f"[{name}] Log: {log_file}", active_log)

    for i in range(retries + 1):
        attempt_info = {
            "attempt": i + 1,
            "start": datetime.now(timezone.utc).isoformat(),
            "exit_code": None,
        }

        try:
            with open(log_file, "a") as f:
                # Use shell=True for flexibility with complex commands passed as string
                proc = subprocess.Popen(
                    cmd_str,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=start_new_session,
                )

                try:
                    # Treat timeout <= 0 as None (infinite wait)
                    wait_arg = timeout if timeout > 0 else None
                    ret = proc.wait(timeout=wait_arg)
                    attempt_info["exit_code"] = ret
                    if ret == 0:
                        meta["success"] = True
                except subprocess.TimeoutExpired:
                    _monitor_log(
                        f"[{name}] Timeout ({timeout}s) expired, killing...", active_log
                    )
                    attempt_info["error"] = "timeout"
                    # Kill process group
                    if start_new_session and hasattr(os, "killpg"):
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()
                    else:
                        proc.kill()
        except Exception as e:
            attempt_info["error"] = str(e)
            _monitor_log(f"[{name}] Exception: {e}", active_log)

        attempt_info["end"] = datetime.now(timezone.utc).isoformat()
        meta["attempts"].append(attempt_info)

        if meta["success"]:
            break

        if i < retries:
            _monitor_log(f"[{name}] Retrying ({i+1}/{retries})...", active_log)
            time.sleep(2)

    # Write hook meta
    meta_file = base_log_dir / f"{name}_meta_{ts}.json"
    try:
        meta_file.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    return meta


def _reconstruct_trials_from_db(
    run_dir: Path, active_log: Path, study_name: Optional[str] = None
) -> bool:
    """Attempt to reconstruct results/trials.csv from optuna_study.db in run_dir.

    Returns True on success, False otherwise. Logs progress to active_log.
    """
    _monitor_log(
        "Starting trials.csv reconstruction from optuna_study.db...", active_log
    )

    # Prefer canonical DB filename, but allow backups like optuna_study.db.bak_*
    db_path = run_dir / "optuna_study.db"
    if not db_path.exists():
        candidates = list(run_dir.glob("optuna_study.db*"))
        if candidates:
            # Pick the most recently modified candidate
            db_path = max(candidates, key=lambda p: p.stat().st_mtime)
            _monitor_log(
                f"Using DB candidate for reconstruction: {db_path}", active_log
            )
        else:
            _monitor_log(
                "No optuna_study.db found; skipping reconstruction.", active_log
            )
            return False

    import sqlite3

    try:
        import optuna
    except Exception as e:
        _monitor_log(
            f"Optuna not available, will attempt direct sqlite parsing: {e}", active_log
        )
        optuna = None

    # Quick DB integrity check
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        res = cur.execute("PRAGMA integrity_check;").fetchone()
        if not res or (isinstance(res, tuple) and res[0] != "ok"):
            _monitor_log(f"SQLite integrity_check failed: {res}", active_log)
            conn.close()
            return False
    except Exception as e:
        _monitor_log(f"Could not run integrity_check on DB: {e}", active_log)
        return False

    rows = []

    if optuna is not None:
        # Determine study_name if not provided
        if study_name is None:
            cfg_path = run_dir / "config" / "config_optuna.yaml"
            try:
                if cfg_path.exists():
                    import yaml

                    cfg = yaml.safe_load(cfg_path.read_text()) or {}
                    study_name = cfg.get("study_name") or cfg.get("optuna", {}).get(
                        "study_name"
                    )
            except Exception:
                pass

        if not study_name:
            try:
                cur.execute("SELECT study_name FROM studies;")
                rows_st = cur.fetchall()
                names = [r[0] for r in rows_st]
                if len(names) == 1:
                    study_name = names[0]
                    _monitor_log(f"Auto-detected study name: {study_name}", active_log)
                else:
                    study_name = "kdr100_opt"
            except Exception:
                study_name = "kdr100_opt"

        try:
            study = optuna.load_study(
                study_name=study_name, storage=f"sqlite:///{db_path}"
            )
            for t in study.trials:
                rows.append(
                    {
                        "trial_number": t.number,
                        "datetime_start": (
                            t.datetime_start.isoformat()
                            if getattr(t, "datetime_start", None)
                            else None
                        ),
                        "datetime_complete": (
                            t.datetime_complete.isoformat()
                            if getattr(t, "datetime_complete", None)
                            else None
                        ),
                        "duration_sec": (
                            t.duration.total_seconds()
                            if getattr(t, "duration", None)
                            else None
                        ),
                        "value": t.value,
                        "a": t.params.get("a") if getattr(t, "params", None) else None,
                        "b": t.params.get("b") if getattr(t, "params", None) else None,
                        "c": t.params.get("c") if getattr(t, "params", None) else None,
                        "min_distance_km": (
                            t.params.get("min_distance_km")
                            if getattr(t, "params", None)
                            else None
                        ),
                        "n_samples": (
                            t.params.get("n_samples")
                            if getattr(t, "params", None)
                            and t.params.get("n_samples") is not None
                            else (
                                getattr(t, "user_attrs", {}).get("n_samples")
                                if getattr(t, "user_attrs", None)
                                else None
                            )
                        ),
                        "state": str(t.state),
                    }
                )
        except Exception as e:
            _monitor_log(
                f"Failed to load optuna study from DB: {e}; falling back to direct sqlite parsing",
                active_log,
            )

    if not rows:
        # Direct sqlite parsing fallback: query trials and join user attributes and params where available
        try:
            cur.execute(
                "SELECT trial_id, number, datetime_start, datetime_complete, state FROM trials ORDER BY number ASC;"
            )
            trials_raw = cur.fetchall()

            # Load trial params
            params_map = {}
            try:
                cur.execute(
                    "SELECT trial_id, param_name, param_value FROM trial_params;"
                )
                for tid, pname, pval in cur.fetchall():
                    params_map.setdefault(tid, {})[pname] = pval
            except Exception:
                params_map = {}

            # Load user attributes
            user_map = {}
            try:
                cur.execute("SELECT trial_id, key, value FROM trial_user_attributes;")
                for tid, key, val in cur.fetchall():
                    user_map.setdefault(tid, {})[key] = val
            except Exception:
                user_map = {}

            # Load scalar values (objective)
            value_map = {}
            try:
                cur.execute("SELECT trial_id, value FROM trial_values;")
                for tid, val in cur.fetchall():
                    value_map[tid] = val
            except Exception:
                value_map = {}

            for tid, number, dt_start, dt_complete, state in trials_raw:
                p = params_map.get(tid, {})
                u = user_map.get(tid, {})
                rows.append(
                    {
                        "trial_number": number,
                        "datetime_start": dt_start,
                        "datetime_complete": dt_complete,
                        "duration_sec": None,
                        "value": value_map.get(tid),
                        "a": (
                            p.get("a")
                            if "a" in p
                            else (u.get("alpha") if "alpha" in u else None)
                        ),
                        "b": (
                            p.get("b")
                            if "b" in p
                            else (u.get("beta") if "beta" in u else None)
                        ),
                        "c": (
                            p.get("c")
                            if "c" in p
                            else (u.get("gamma") if "gamma" in u else None)
                        ),
                        "min_distance_km": (
                            p.get("min_distance_km")
                            if "min_distance_km" in p
                            else (
                                u.get("min_distance_km")
                                if "min_distance_km" in u
                                else None
                            )
                        ),
                        # Parse n_samples as integer when possible (params may be numeric, user attrs are JSON/text)
                        "n_samples": (
                            int(p.get("n_samples"))
                            if "n_samples" in p and p.get("n_samples") is not None
                            else (
                                int(u.get("n_samples"))
                                if "n_samples" in u
                                and u.get("n_samples") not in (None, "null")
                                else None
                            )
                        ),
                        "state": state,
                    }
                )
        except Exception as e:
            _monitor_log(f"Direct DB parse failed: {e}", active_log)
            conn.close()
            return False

    # After constructing rows, attempt to backfill n_samples from direct DB user attributes if available
    try:
        # user_map was populated in direct sqlite fallback; use it when present
        if "user_map" in locals() and user_map:
            # Build mapping trial_id -> trial_number
            conn2 = sqlite3.connect(str(db_path))
            cur2 = conn2.cursor()
            cur2.execute("SELECT trial_id, number FROM trials;")
            id_to_num = {r[0]: r[1] for r in cur2.fetchall()}
            conn2.close()
            # Create mapping number -> n_samples
            num_to_ns = {}
            for tid, attrs in user_map.items():
                if "n_samples" in attrs and attrs.get("n_samples") not in (
                    None,
                    "null",
                ):
                    try:
                        num = id_to_num.get(tid)
                        if num is not None:
                            num_to_ns[num] = int(attrs.get("n_samples"))
                    except Exception:
                        pass
            # Apply backfill
            if num_to_ns:
                for r in rows:
                    tn = r.get("trial_number")
                    if tn in num_to_ns and (
                        r.get("n_samples") is None
                        or str(r.get("n_samples")).lower() in ("none", "nan")
                    ):
                        r["n_samples"] = num_to_ns[tn]
    except Exception as e:
        _monitor_log(
            f"Warning: could not backfill n_samples from user attributes: {e}",
            active_log,
        )

    conn.close()
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("trial_number")

    # Ensure n_samples column is filled from num_to_ns mapping when possible
    try:
        if "num_to_ns" in locals() and num_to_ns:
            df["n_samples"] = df["trial_number"].map(num_to_ns).astype("Int64")
    except Exception:
        pass

    out = run_dir / "results" / "trials.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")

    try:
        df.to_csv(tmp, index=False)
        # Backup existing file if present
        if out.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            bak = out.with_suffix(f".bak_reconstruct_{ts}.csv")
            try:
                out.replace(bak)
            except Exception:
                try:
                    out.rename(bak)
                except Exception:
                    pass
        # Atomic replace
        os.replace(str(tmp), str(out))

        # Post-process: ensure n_samples filled from DB user attributes when possible
        try:
            import pandas as _pd

            df2 = _pd.read_csv(out)
            if "num_to_ns" in locals() and num_to_ns:
                df2["n_samples"] = df2["trial_number"].map(num_to_ns).astype("Int64")
                df2.to_csv(out, index=False)
        except Exception as e:
            _monitor_log(
                f"Warning: failed to post-process n_samples column: {e}", active_log
            )

        # Write provenance meta
        meta = {
            "reconstructed_at": datetime.now(timezone.utc).isoformat(),
            "source_db": str(db_path),
            "n_trials": len(df),
            "best_value": (
                float(df["value"].max()) if not df["value"].isnull().all() else None
            ),
            "study_name": study_name,
            "optuna_version": (
                getattr(optuna, "__version__", None)
                if "optuna" in globals() and optuna is not None
                else None
            ),
        }
        meta_file = out.with_name("trials_reconstruct_meta.json")
        try:
            meta_file.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

        _monitor_log(
            f"Successfully reconstructed trials.csv ({len(df)} rows)", active_log
        )
        return True
    except Exception as e:
        _monitor_log(f"Failed to write reconstructed CSV: {e}", active_log)
        try:
            tmp.unlink()
        except Exception:
            pass
        return False


def _reconcile_trials(run_dir: Path, active_log: Path) -> dict:
    """
    Decide whether to use optuna_study.db or trials.csv as the source of truth.
    If DB is authoritative (newer/more complete), reconstruct trials.csv.

    Returns dict with keys: ok, source, completed_count, best_value, actions, reason, db_path.
    """
    res = {
        "ok": False,
        "source": None,
        "completed_count": 0,
        "best_value": None,
        "actions": [],
        "reason": None,
        "db_path": None,
        "attempts": [],  # structured steps: {step, ts, result, message}
    }

    # 1. Inspect DB
    db_path = run_dir / "optuna_study.db"
    if not db_path.exists():
        candidates = list(run_dir.glob("optuna_study.db*"))
        if candidates:
            db_path = max(candidates, key=lambda p: p.stat().st_mtime)

    db_info = None
    if db_path.exists():
        res["db_path"] = str(db_path)
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            integrity = cur.execute("PRAGMA integrity_check;").fetchone()
            conn.close()

            if not integrity or (isinstance(integrity, tuple) and integrity[0] != "ok"):
                msg = f"db_corrupt: {integrity}"
                res["actions"].append(msg)
                res["attempts"].append(
                    {
                        "step": "db_integrity_check",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "result": "corrupt",
                        "message": str(integrity),
                    }
                )
            else:
                # Load stats from DB
                try:
                    import optuna
                    from optuna.trial import TrialState

                    # Guess study name
                    conn = sqlite3.connect(str(db_path))
                    cur = conn.cursor()
                    rows = cur.execute("SELECT study_name FROM studies;").fetchall()
                    conn.close()
                    study_name = rows[0][0] if rows and len(rows) == 1 else "kdr100_opt"

                    study = optuna.load_study(
                        study_name=study_name, storage=f"sqlite:///{db_path}"
                    )
                    completed = len(
                        [t for t in study.trials if t.state == TrialState.COMPLETE]
                    )
                    try:
                        best = study.best_value
                    except ValueError:
                        best = None
                    db_info = {"completed": completed, "best": best}
                    res["attempts"].append(
                        {
                            "step": "db_inspect",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "result": "ok",
                            "message": f"completed={completed}, best={best}",
                        }
                    )
                except Exception as e:
                    res["actions"].append(f"db_read_failed: {e}")
                    res["attempts"].append(
                        {
                            "step": "db_inspect",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "result": "error",
                            "message": str(e),
                        }
                    )
        except Exception as e:
            res["actions"].append(f"db_check_error: {e}")

    # 2. Inspect CSV
    csv_path = run_dir / "results" / "trials.csv"
    csv_info = None
    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if "state" in df.columns:
                completed = len(df[df["state"].astype(str).str.contains("COMPLETE")])
            else:
                completed = len(df)

            best = (
                df["value"].max()
                if "value" in df.columns and not df["value"].isnull().all()
                else None
            )
            csv_info = {"completed": completed, "best": best}
            res["attempts"].append(
                {
                    "step": "csv_read",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "result": "ok",
                    "message": f"completed={completed}, best={best}",
                }
            )
        except Exception as e:
            res["actions"].append(f"csv_read_failed: {e}")
            res["attempts"].append(
                {
                    "step": "csv_read",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "result": "error",
                    "message": str(e),
                }
            )

    # If DB was corrupt but CSV exists, explicitly note the decision to use CSV
    if csv_info and any(a.startswith("db_corrupt") for a in res["actions"]):
        res["actions"].append("used_csv_due_to_db_corruption")

    # 3. Reconcile
    if db_info:
        # DB is available and valid
        if not csv_info or (db_info["completed"] > csv_info["completed"]):
            # DB is authoritative (either CSV missing or DB has more trials)
            reason_str = (
                "CSV missing"
                if not csv_info
                else f"DB has more trials ({db_info['completed']} > {csv_info['completed']})"
            )
            _monitor_log(
                f"Reconcile: {reason_str}. Reconstructing trials.csv from DB...",
                active_log,
            )

            if _reconstruct_trials_from_db(run_dir, active_log):
                res["ok"] = True
                res["source"] = "reconstructed"
                res["completed_count"] = db_info["completed"]
                res["best_value"] = db_info["best"]
                res["actions"].append("reconstructed_from_db")
                res["attempts"].append(
                    {
                        "step": "reconstruct",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "result": "ok",
                        "message": "reconstructed from db",
                    }
                )
            else:
                res["ok"] = False
                res["reason"] = "reconstruction_failed"
                res["attempts"].append(
                    {
                        "step": "reconstruct",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "result": "error",
                        "message": "reconstruction_failed",
                    }
                )
        else:
            # CSV exists and has equal or more trials -> Trust CSV (avoid overhead)
            res["ok"] = True
            res["source"] = "trials_csv"
            res["completed_count"] = csv_info["completed"]
            res["best_value"] = csv_info["best"]
            res["actions"].append("kept_existing_csv")
            if db_info["completed"] < csv_info["completed"]:
                res["actions"].append(
                    f"warning_db_has_fewer_trials({db_info['completed']})_than_csv({csv_info['completed']})"
                )

    elif csv_info:
        # Only CSV exists
        res["ok"] = True
        res["source"] = "trials_csv"
        res["completed_count"] = csv_info["completed"]
        res["best_value"] = csv_info["best"]
        res["actions"].append("db_missing_using_csv")
    else:
        res["ok"] = False
        # If DB path exists and integrity check failed, be explicit about corruption
        if res["actions"] and any(a.startswith("db_corrupt") for a in res["actions"]):
            res["reason"] = "db_corrupt"
        else:
            res["reason"] = "no_db_and_no_trials"

    return res


def run_cmd_with_retry(
    cmd: str,
    retries: int = 2,
    delay: int = 5,
    cwd: Path | None = None,
    fail_ok: bool = False,
) -> int:
    """Module-level proxy to run child commands with simple retry semantics.

    Tests can monkeypatch this symbol to simulate child process outcomes.
    """
    try:
        from scripts.xxl_KDR146_run_thesis_complete import run_cmd_with_retry as _r

        return _r(cmd, retries=retries, delay=delay, cwd=cwd, fail_ok=fail_ok)
    except Exception:
        # Fallback: use subprocess.run synchronously
        try:
            import subprocess

            res = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None)
            return res.returncode
        except Exception:
            return 1


def _resume_run(
    run_selector: str, active_log: Path, force: bool = False, dry_run: bool = False
) -> dict:
    """Attempt to safely resume a previous run identified by "last" or run name.

    Behavior:
      - Performs DB integrity checks and makes a DB backup before any modifications.
      - Computes remaining Optuna trials (counts only COMPLETE trials) and will resume
        Optuna by invoking `scripts/optuna_optimize.py --n-trials <remaining>` when
        trials remain.
      - If Optuna is already complete, the monitor will inspect downstream pipeline
        artifacts (reproducibility runs, THESIS_FINAL_SELECTION_XXL.json) and will
        sequentially run missing phases (e.g., reproducibility, finalization) so
        the *original* full run can be completed. Each phase is executed via
        `run_hook` (logs written to `run_dir/logs`) and recorded in
        `results/resume_meta.json` under the `phases` field.
      - `--dry-run-restart` shows planned actions without modifying artifacts.

    Returns a metadata dict with outcome and details.
    """
    _monitor_log(f"Resume requested for: {run_selector}", active_log)
    runs_root = ROOT / "outputs" / "runs"
    if not runs_root.exists():
        _monitor_log("No runs directory found; nothing to resume.", active_log)
        return {"ok": False, "reason": "no_runs_dir"}

    # Resolve run_dir
    if run_selector == "last":
        xxl_dirs = sorted(
            [
                p
                for p in runs_root.iterdir()
                if p.is_dir()
                and "hamburg" in p.name.lower()
                and "xxl" in p.name.lower()
            ]
        )
        if not xxl_dirs:
            _monitor_log("No previous XXL run found to restart.", active_log)
            return {"ok": False, "reason": "no_run_found"}
        run_dir = xxl_dirs[-1]
    else:
        run_dir = runs_root / run_selector
        if not run_dir.exists():
            # try fuzzy match
            candidates = [p for p in runs_root.iterdir() if run_selector in p.name]
            if candidates:
                run_dir = sorted(candidates)[-1]
            else:
                _monitor_log(f"Specified run dir not found: {run_selector}", active_log)
                return {"ok": False, "reason": "not_found"}

    _monitor_log(f"Resolved run dir: {run_dir}", active_log)

    # Check manifest status
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            status = m.get("status")
            if status == "complete":
                _monitor_log(
                    "Run already marked complete; nothing to resume.", active_log
                )
                return {"ok": False, "reason": "already_complete"}
        except Exception:
            pass

    # Load configured n_trials
    cfg_path = run_dir / "config" / "config_optuna.yaml"
    configured_n = None
    try:
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            configured_n = cfg.get("n_trials") or cfg.get("optuna", {}).get("n_trials")
            if configured_n is not None:
                configured_n = int(configured_n)
    except Exception:
        configured_n = None

    # Reconcile trials (DB vs CSV)
    rec = _reconcile_trials(run_dir, active_log)
    if not rec["ok"]:
        _monitor_log(f"Resume aborted: {rec.get('reason')}", active_log)
        return {"ok": False, "reason": rec.get("reason")}

    completed = rec["completed_count"]
    best_before = rec["best_value"]
    resume_source = rec["source"]
    db_path = Path(rec["db_path"]) if rec.get("db_path") else None

    # Use configured n_trials when available; else infer target from manifest/config
    if configured_n is None:
        # try manifest
        try:
            if manifest.exists():
                mm = json.loads(manifest.read_text())
                configured_n = (
                    int(mm.get("metadata", {}).get("n_trials"))
                    if mm.get("metadata", {}).get("n_trials")
                    else None
                )
        except Exception:
            configured_n = None

    if configured_n is None:
        _monitor_log(
            "Configured target n_trials unknown; cannot safely resume.", active_log
        )
        return {"ok": False, "reason": "unknown_target_n"}

    remaining = int(configured_n) - int(completed)
    remaining_report = max(0, remaining)
    _monitor_log(
        f"Completed trials: {completed}; Target: {configured_n}; Remaining: {remaining}",
        active_log,
    )

    if remaining <= 0:
        _monitor_log(
            "No remaining trials to run; will check downstream pipeline phases to resume.",
            active_log,
        )

    # Create resume lock
    lock = run_dir / ".resume.lock"
    if lock.exists():
        _monitor_log(
            "Resume lock exists; another resume may be running. Aborting.", active_log
        )
        return {"ok": False, "reason": "locked"}
    try:
        lock.write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    if db_path and db_path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        db_backup = db_path.with_name(f"{db_path.name}.bak_resume_{ts}")
        try:
            shutil.copy2(db_path, db_backup)
            _monitor_log(f"Backed up DB to: {db_backup}", active_log)
        except Exception as e:
            _monitor_log(f"Warning: failed to backup DB: {e}", active_log)
    else:
        db_backup = "N/A"

    # Use ExperimentStateAnalyzer and RecoveryPlanner to plan recovery tasks
    try:
        from scripts.monitor_state import ExperimentStateAnalyzer
        from scripts.recovery import RecoveryPlanner, TaskExecutor
    except Exception:
        ExperimentStateAnalyzer = None
        RecoveryPlanner = None
        TaskExecutor = None

    analyzer_state = {}
    if ExperimentStateAnalyzer is not None:
        try:
            analyzer = ExperimentStateAnalyzer(run_dir)
            analyzer_state = analyzer.inspect()
        except Exception:
            analyzer_state = {}

    # Build a state summary for planner
    state = {
        "csv_exists": analyzer_state.get("csv_exists", False)
        or (run_dir / "results" / "trials.csv").exists(),
        "csv_completed": analyzer_state.get("csv_completed", completed),
        "csv_best": analyzer_state.get("csv_best", best_before),
        "db_exists": analyzer_state.get("db_exists", False)
        or (db_path.exists() if db_path else False),
        "db_integrity_ok": analyzer_state.get("db_integrity_ok", False),
        "db_completed": analyzer_state.get("db_completed", 0),
        "db_best": analyzer_state.get("db_best", None),
        "db_path": str(db_path) if db_path else None,
        "repro_done": bool(
            list((ROOT / "outputs" / "runs").glob("*thesis_hamburg_reproducibility*"))
        ),
        "final_exists": (ROOT / "outputs" / "THESIS_FINAL_SELECTION_XXL.json").exists(),
        "final_belongs": False,
        "n_samples": None,
    }

    # Check whether global final selection belongs to this run
    final_sel = ROOT / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    if final_sel.exists():
        try:
            sel = json.loads(final_sel.read_text())
            sel_run = sel.get("run_id")
            if sel_run and (
                str(sel_run) == run_dir.name or run_dir.name in str(sel_run)
            ):
                state["final_belongs"] = True
        except Exception:
            pass

    # If best_params includes n_samples, surface it
    try:
        if best_before is not None:
            # attempt to read n_samples from best trial row in CSV
            if (run_dir / "results" / "trials.csv").exists():
                import pandas as _pd

                df_tmp = _pd.read_csv(run_dir / "results" / "trials.csv")
                best_row = df_tmp[df_tmp["value"] == df_tmp["value"].max()].iloc[0]
                state["n_samples"] = (
                    int(best_row.get("n_samples"))
                    if "n_samples" in best_row
                    and not _pd.isna(best_row.get("n_samples"))
                    else None
                )
    except Exception:
        state["n_samples"] = None

    # Compute dataset size (used for repro CLI invocation)
    n_candidates_calculated = 673
    try:
        import pandas as pd

        tiles_df = pd.read_csv(ROOT / "data" / "new_all_tiles.csv")
        n_candidates_calculated = len(tiles_df)
    except Exception:
        pass

    planner = RecoveryPlanner(configured_n=configured_n, repro_seeds=[43, 44])
    tasks = planner.plan(state)

    # Map tasks to phase dicts with representative commands (for dry-run reporting)
    phases = []
    for t in tasks:
        if t.name == "reconstruct":
            phases.append({"name": "reconstruct", "cmd": "reconstruct"})
        elif t.name == "optuna":
            n = int(t.params.get("n_trials", 0))
            phases.append(
                {
                    "name": "optuna",
                    "cmd": f"{sys.executable} -m scripts.optuna_optimize --n-trials {n}",
                }
            )
        elif t.name == "repro":
            seeds = ",".join(str(s) for s in t.params.get("seeds", []))
            phases.append(
                {
                    "name": "reproducibility",
                    "cmd": f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase repro --seeds {seeds} --n-trials {configured_n} --n-candidates {n_candidates_calculated}",
                }
            )
        elif t.name == "finalize":
            n_samples = t.params.get("n_samples")
            cmd = f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase finalize --run-dir {run_dir}"
            if n_samples:
                cmd += f" --n-samples {n_samples}"
            phases.append({"name": "finalize", "cmd": cmd})
        else:
            phases.append({"name": t.name, "cmd": None})

    _monitor_log(f"Planned resume phases: {[p['name'] for p in phases]}", active_log)

    # Dry-run: report planned actions
    if dry_run:
        try:
            lock.unlink()
        except Exception:
            pass
        return {
            "ok": True,
            "dry_run": True,
            "run_dir": str(run_dir),
            "configured_n": configured_n,
            "completed": completed,
            "remaining": remaining_report,
            "db_backup": str(db_backup),
            "phases": phases,
        }

    # Require explicit force unless interactive
    if not force:
        # non-interactive contexts (CI) should require --force-restart
        if not sys.stdin.isatty():
            _monitor_log(
                "Non-interactive shell; resume requires --force-restart", active_log
            )
            try:
                lock.unlink()
            except Exception:
                pass
            return {"ok": False, "reason": "need_force"}
        ok = input(
            f"About to resume run {run_dir.name}: {remaining} trials remaining. Proceed? (y/N): "
        )
        if ok.strip().lower() != "y":
            _monitor_log("User declined resume.", active_log)
            try:
                lock.unlink()
            except Exception:
                pass
            return {"ok": False, "reason": "user_decline"}

    try:
        env = os.environ.copy()
        env["EXPERIMENT_RUN_DIR"] = str(run_dir)

        # Execute planned tasks using TaskExecutor
        phase_results = []
        overall_ok = True
        try:
            executor = TaskExecutor(
                run_hook_func=lambda **kw: run_hook(**kw),
                run_cmd_func=run_cmd_with_retry,
            )
            _monitor_log("Executor initialized: yes", active_log)
        except Exception as e:
            executor = None
            _monitor_log(f"Executor init failed: {e}", active_log)

        _monitor_log(f"Planned tasks: {[t.name for t in tasks]}", active_log)

        if executor is None:
            _monitor_log(
                "Executor not available; using fallback sequential phases", active_log
            )
            # Fallback: run old behavior for backward compatibility
            for phase in phases:
                name = phase.get("name")
                cmd = phase.get("cmd")
                _monitor_log(f"Starting resume phase: {name}", active_log)
                hook_meta = run_hook(
                    name=f"resume_phase_{name}",
                    cmd_str=cmd,
                    base_log_dir=run_dir / "logs",
                    active_log=active_log,
                    timeout=0,
                    retries=0,
                    env=env,
                    start_new_session=True,
                    pass_dry_run=False,
                )
                if name == "optuna":
                    try:
                        _reconstruct_trials_from_db(run_dir, active_log)
                    except Exception as e:
                        _monitor_log(
                            f"Post-resume reconstruction error: {e}", active_log
                        )
                phase_results.append({"name": name, "cmd": cmd, "hook_meta": hook_meta})
                if not hook_meta.get("success"):
                    overall_ok = False
                    _monitor_log(
                        f"Phase {name} reported failure; continuing to next phase (conservative mode)",
                        active_log,
                    )
        else:
            # Convert Task objects back to a list and execute
            exec_results = executor.execute(tasks, str(run_dir))
            # Map exec_results to phase_results structure
            for r, t in zip(exec_results, tasks):
                # Use human-readable phase names in resume_meta (match dry-run output)
                entry = {"name": t.name, "cmd": None, "result": r}
                if t.name == "optuna":
                    entry["cmd"] = (
                        f"{sys.executable} -m scripts.optuna_optimize --n-trials {t.params.get('n_trials')}"
                    )
                    # After optuna, reconstruct trials from DB
                    try:
                        _reconstruct_trials_from_db(run_dir, active_log)
                    except Exception as e:
                        _monitor_log(
                            f"Post-resume reconstruction error: {e}", active_log
                        )
                elif t.name == "repro":
                    # normalize executed phase name to match dry-run naming
                    entry["name"] = "reproducibility"
                    seeds = ",".join(str(s) for s in t.params.get("seeds", []))
                    entry["cmd"] = (
                        f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase repro --seeds {seeds} --n-trials {configured_n} --n-candidates {n_candidates_calculated}"
                    )
                elif t.name == "finalize":
                    n_samples = t.params.get("n_samples")
                    entry["cmd"] = (
                        f"{sys.executable} -m scripts.xxl_KDR146_run_thesis_complete --phase finalize --run-dir {run_dir}"
                    )
                    if n_samples:
                        entry["cmd"] += f" --n-samples {n_samples}"
                elif t.name == "reconstruct":
                    entry["cmd"] = "reconstruct"
                phase_results.append(entry)
                # Interpret execution outcome
                ok_flag = True
                if isinstance(r, dict):
                    # heuristics: check 'success' or exit_code
                    ok_flag = (
                        bool(r.get("success", True))
                        if "success" in r
                        else (
                            r.get("exit_code", 0) == 0
                            if isinstance(r.get("exit_code", None), int)
                            else True
                        )
                    )
                elif isinstance(r, int):
                    ok_flag = r == 0
                if not ok_flag:
                    overall_ok = False
                    _monitor_log(
                        f"Task {t.name} reported failure; continuing to next task (conservative mode)",
                        active_log,
                    )

    except Exception as e:
        _monitor_log(f"Resume flow exception: {e}", active_log)
        # Attempt to write a resume_meta file documenting the failure for later inspection
        try:
            import traceback

            # Gather provenance: git short sha (if available) and active conda env name
            try:
                git_sha = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                git_sha = None
            conda_env = (
                os.environ.get("DATASELECTOR_ENV_NAME")
                or os.environ.get("CONDA_DEFAULT_ENV")
                or None
            )
            env_snapshot = {
                "PATH": os.environ.get("PATH"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "CONDA_ENV": conda_env,
            }

            resume_meta_err = {
                "resumed_at": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "ok": False,
                "reason": "internal_exception",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "git_sha": git_sha,
                "env": env_snapshot,
                "phases": [],
            }
            (run_dir / "results" / "resume_meta.json").write_text(
                json.dumps(resume_meta_err, indent=2)
            )
        except Exception:
            pass
        try:
            lock.unlink()
        except Exception:
            pass
        return {"ok": False, "reason": "internal_exception", "message": str(e)}

    # reload study to get final counts if optuna present
    try:
        import optuna
        from optuna.trial import TrialState

        study2 = optuna.load_study(study_name=None, storage=f"sqlite:///{db_path}")
        completed_after = len(
            [t for t in study2.trials if t.state == TrialState.COMPLETE]
        )
        best_after = study2.best_value if study2.trials else None
    except Exception:
        completed_after = None
        best_after = None

    resume_meta = {
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "configured_n": configured_n,
        "completed_before": completed,
        "remaining_requested": remaining_report,
        "completed_after": completed_after,
        "best_before": best_before,
        "best_after": best_after,
        "resume_source": resume_source,
        "reconcile_actions": rec.get("actions", []),
        "resume_attempts": rec.get("attempts", []),
        "db_backup": str(db_backup),
        "phases": phase_results,
        "ok": overall_ok,
    }
    try:
        (run_dir / "results" / "resume_meta.json").write_text(
            json.dumps(resume_meta, indent=2)
        )
    except Exception:
        pass

    try:
        lock.unlink()
    except Exception:
        pass

    _monitor_log(
        f"Resume flow completed; returning resume_meta ok={resume_meta.get('ok')}",
        active_log,
    )
    return resume_meta


def main():
    parser = argparse.ArgumentParser(description="Monitor XXL pipeline run")
    parser.add_argument(
        "--no-new-session",
        action="store_true",
        help="Do not start child in new session (better for tests)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=30, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--child-dry-run",
        action="store_true",
        help="Pass --dry-run to the child orchestrator for fast smoke tests",
    )

    # Hook arguments
    parser.add_argument(
        "--pre-run-cmd", type=str, help="Command to run before the orchestrator"
    )
    parser.add_argument(
        "--post-run-cmd", type=str, help="Command to run after the orchestrator"
    )
    parser.add_argument(
        "--pre-run-timeout",
        type=int,
        default=600,
        help="Timeout for pre-run command in seconds (0 = no timeout)",
    )
    parser.add_argument(
        "--pre-run-delay",
        type=int,
        default=0,
        help="Delay in seconds after pre-run hook before starting main run",
    )
    parser.add_argument(
        "--pre-run-retries", type=int, default=0, help="Retries for pre-run command"
    )
    parser.add_argument(
        "--pre-run-fail-mode",
        choices=["abort", "warn", "continue"],
        default="abort",
        help="Action on pre-run failure",
    )
    parser.add_argument(
        "--pre-run-dry-run",
        action="store_true",
        help="Pass --dry-run to pre-run command if present",
    )
    parser.add_argument(
        "--no-reconstruct",
        action="store_true",
        help="Do not attempt to reconstruct results/trials.csv from optuna_study.db if missing",
    )

    # Resume / restart options
    parser.add_argument(
        "--restart",
        nargs="?",
        const="last",
        help="Resume a previous run: 'last' (default when provided) or specific run_dir name",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="If set, resume without interactive confirmation",
    )
    parser.add_argument(
        "--dry-run-restart",
        action="store_true",
        help="Do not actually start resume, only display planned actions",
    )
    # Auto-resume options
    parser.add_argument(
        "--auto-resume-force",
        action="store_true",
        help="Automatically attempt to resume the last run and force restart non-interactively (safe for cron/CI)",
    )
    # Dataselector environment options
    parser.add_argument(
        "--no-dataselector-env",
        action="store_true",
        help="Do not run hooks and child processes inside the 'dataselector' mamba/conda environment (default: use env)",
    )
    parser.add_argument(
        "--dataselector-env-name",
        type=str,
        default=os.environ.get("DATASELECTOR_ENV_NAME", "dataselector"),
        help="Name of the mamba/conda environment to use (default from DATASELECTOR_ENV_NAME or 'dataselector')",
    )

    args = parser.parse_args()

    # Ensure PYTHONPATH in env
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output for real-time monitoring

    # Initialize dataselector env runner (default: use dataselector env)
    # Allow disabling with --no-dataselector-env and override name with --dataselector-env-name
    _init_env_runner(
        use_env=not args.no_dataselector_env, env_name=args.dataselector_env_name
    )

    # Create per-run timestamped log file and make `XXL_FULL_RUN.log` point to it (symlink)
    forced_ts = os.environ.get("MONITOR_FORCE_TS")
    if forced_ts:
        ts = forced_ts
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOG_FILE_TS = LOG_FILE.parent / f"XXL_FULL_RUN_{ts}.log"
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
            pass_dry_run=args.pre_run_dry_run,
        )
        if not pre_run_meta["success"]:
            msg = f"PRE-RUN hook failed (exit code: {pre_run_meta['attempts'][-1].get('exit_code')})"
            _monitor_log(msg, ACTIVE_LOG)

            # Print log excerpt for debugging
            try:
                log_path = Path(pre_run_meta["log_file"])
                if log_path.exists():
                    _monitor_log(
                        f"--- Log excerpt from {log_path.name} ---", ACTIVE_LOG
                    )
                    _monitor_log(
                        log_path.read_text(errors="replace")[-2000:], ACTIVE_LOG
                    )
                    _monitor_log(
                        "-------------------------------------------", ACTIVE_LOG
                    )
            except Exception as e:
                _monitor_log(f"Could not read hook log: {e}", ACTIVE_LOG)

            if args.pre_run_fail_mode == "abort":
                _monitor_log("Aborting run due to pre-run failure.", ACTIVE_LOG)
                sys.exit(1)
            else:
                _monitor_log(
                    f"Continuing despite pre-run failure (mode: {args.pre_run_fail_mode})",
                    ACTIVE_LOG,
                )

        if args.pre_run_delay > 0:
            _monitor_log(
                f"Waiting {args.pre_run_delay}s before starting main run...", ACTIVE_LOG
            )
            time.sleep(args.pre_run_delay)

    # If --restart requested: perform resume flow and exit
    if args.restart:
        _monitor_log(
            f"Restart requested: {args.restart} (force={args.force_restart}, dry_run={args.dry_run_restart})",
            ACTIVE_LOG,
        )
        res = _resume_run(
            args.restart,
            ACTIVE_LOG,
            force=args.force_restart,
            dry_run=args.dry_run_restart,
        )
        if res.get("ok"):
            _monitor_log(
                "Resume flow completed (or dry-run reported); exiting monitor.",
                ACTIVE_LOG,
            )
            sys.exit(0)
        else:
            _monitor_log(f"Resume aborted/skipped: {res.get('reason')}", ACTIVE_LOG)
            sys.exit(1)

    # Auto-resume (force) requested: attempt to resume last run automatically and exit
    if args.auto_resume_force:
        _monitor_log(
            "Auto-resume-force requested: attempting to resume last run (force=true)",
            ACTIVE_LOG,
        )
        res = _resume_run("last", ACTIVE_LOG, force=True, dry_run=False)
        if res.get("ok"):
            _monitor_log(
                "Auto-resume completed successfully; exiting monitor.", ACTIVE_LOG
            )
            sys.exit(0)
        else:
            _monitor_log(
                f"Auto-resume failed: {res.get('reason')}; exiting with failure.",
                ACTIVE_LOG,
            )
            sys.exit(1)

    # Start the full run subprocess
    cmd = [sys.executable, str(MAIN_SCRIPT)]

    # If a selected_sampler artifact exists, use it to instruct the main script which sampler to use
    try:
        sel = ROOT / "outputs" / "selected_sampler.json"
        # Debug visibility: print the exact path and existence check so tests can be deterministic
        print(
            f"DEBUG: selected_sampler path: {sel} ; exists={sel.exists()}", flush=True
        )
        if sel.exists():
            try:
                txt = sel.read_text()
                print(f"DEBUG: selected_sampler contents: {txt}", flush=True)
                j = __import__("json").loads(txt)
                best = j.get("best")
                if best:
                    # The pre-run compares Optuna samplers (qmc/tpe/cmaes). Pass the
                    # selected choice to the orchestrator as the Optuna sampler flag
                    # so that downstream runs use the empirically best sampler.
                    cmd.extend(["--optuna-sampler", str(best)])
                    _monitor_log(
                        f"Using selected sampler for Optuna from artifact: {best}",
                        ACTIVE_LOG,
                    )
                    # Also emit to stdout for visibility in non-interactive test runs
                    print(
                        f"Using selected sampler for Optuna from artifact: {best}",
                        flush=True,
                    )
            except Exception as e:
                _monitor_log(f"Could not parse selected_sampler.json: {e}", ACTIVE_LOG)
    except Exception as e:
        print(f"DEBUG: selected_sampler check failed: {e}", flush=True)
        pass

    # Allow monitor to instruct child to run in dry-run mode for smoke tests
    if args.child_dry_run:
        cmd.append("--dry-run")
    print(f"Starting full run: {' '.join(cmd)} (log: {ACTIVE_LOG})")
    with open(ACTIVE_LOG, "ab") as logf:
        # start_new_session=True creates a new process group, allowing us to kill the whole tree.
        # We disable this for tests via --no-new-session to allow standard signal propagation.
        start_new_session = not args.no_new_session
        process = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=start_new_session,
        )

    start_time = time.time()
    print(f"PID: {process.pid}, writing logs to: {LOG_FILE}")

    # PID Bookkeeping
    pid_file = LOG_FILE_TS.with_suffix(".pid")
    pgid = None
    try:
        # Use getpgid if available, otherwise just use PID as best-effort
        try:
            pgid = os.getpgid(process.pid) if hasattr(os, "getpgid") else process.pid
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
                with open(ACTIVE_LOG, "r") as f:
                    # Handle log rotation/truncation
                    f.seek(0, os.SEEK_END)
                    if f.tell() < last_size:
                        last_size = 0
                    f.seek(last_size)
                    new_content = f.read()
                    last_size = f.tell()
            except Exception:
                new_content = ""

            # Process only new lines to avoid repeated messages
            if new_content:
                for line in new_content.splitlines():
                    if not line or line in seen_lines:
                        continue
                    seen_lines.append(line)

                    # Detect running tasks (e.g., "Running 1/20: alpha=...")
                    m = re.search(r"Running\s+\d+/\d+:\s*(.+)", line)
                    if m:
                        _monitor_log(f"Starte: {m.group(1)}", ACTIVE_LOG)
                        continue

                    # Detect export completions
                    m = re.search(r"Auswahl exportiert nach:\s*(.+)$", line)
                    if m:
                        _monitor_log(
                            f"Prozess beendet: Auswahl exportiert nach: {m.group(1)}",
                            ACTIVE_LOG,
                        )
                        continue

                    m = re.search(r"Saved results:\s*(.+)\s*\(csv", line)
                    if m:
                        _monitor_log(
                            f"Prozess beendet: Saved results: {m.group(1)}", ACTIVE_LOG
                        )
                        continue

                    # Phase completions (localized messages)
                    if "Phase 1 ABGESCHLOSSEN" in line or "PHASE 1 COMPLETE" in line:
                        if "PHASE 1 COMPLETE" not in phase_events:
                            phase_events.append("PHASE 1 COMPLETE")
                            _monitor_log("OBSERVED: PHASE 1 COMPLETE", ACTIVE_LOG)
                        continue
                    if "Phase 2 COMPLETE" in line:
                        if "PHASE 2 COMPLETE" not in phase_events:
                            phase_events.append("PHASE 2 COMPLETE")
                            _monitor_log("OBSERVED: PHASE 2 COMPLETE", ACTIVE_LOG)
                        continue
                    if "Phase 3 COMPLETE" in line:
                        if "PHASE 3 COMPLETE" not in phase_events:
                            phase_events.append("PHASE 3 COMPLETE")
                            _monitor_log("OBSERVED: PHASE 3 COMPLETE", ACTIVE_LOG)
                        continue
                    if "Phase 4 COMPLETE" in line:
                        if "PHASE 4 COMPLETE" not in phase_events:
                            phase_events.append("PHASE 4 COMPLETE")
                            _monitor_log("OBSERVED: PHASE 4 COMPLETE", ACTIVE_LOG)
                        continue

            # Detect creation of XXL run folder
            runs_root = ROOT / "outputs" / "runs"
            xxl_dirs = []
            if runs_root.exists():
                xxl_dirs.extend(
                    [
                        p
                        for p in runs_root.iterdir()
                        if p.is_dir()
                        and "hamburg" in p.name.lower()
                        and "xxl" in p.name.lower()
                    ]
                )
            # Backward-compatible: also honor glob results (tests may patch glob.glob)
            try:
                for p in glob.glob(str(runs_root / "*")):
                    ppath = Path(p)
                    if (
                        ppath.is_dir()
                        and "hamburg" in ppath.name.lower()
                        and "xxl" in ppath.name.lower()
                    ):
                        if ppath not in xxl_dirs:
                            xxl_dirs.append(ppath)
            except Exception:
                pass
            xxl_dirs = sorted(xxl_dirs)
            if xxl_dirs:
                latest_xxl = xxl_dirs[-1]
                trials_csv = latest_xxl / "results" / "trials.csv"

                if trials_csv.exists() and trials_csv.stat().st_size > 0:
                    current_size = trials_csv.stat().st_size

                    # Check stability: size must be stable across one poll interval
                    if trials_csv == trials_csv_candidate:
                        if current_size == trials_csv_last_size:
                            # Stable!
                            if latest_xxl != detected_xxl:
                                detected_xxl = latest_xxl
                                _monitor_log(
                                    f"Detected stable XXL run directory: {latest_xxl}",
                                    ACTIVE_LOG,
                                )
                        else:
                            trials_csv_last_size = current_size
                    else:
                        trials_csv_candidate = trials_csv
                        trials_csv_last_size = current_size

            if ret is not None:
                print(f"Process exited with code: {ret}")
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt received: terminating child process group")
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
            timeout=600,  # Default timeout for post-run
            retries=0,
            env=env,
            start_new_session=not args.no_new_session,
            pass_dry_run=args.pre_run_dry_run,  # Reuse dry-run flag logic
        )

    # After process end: assemble report
    exit_code = process.returncode
    elapsed = end_time - start_time

    # Find latest XXL run dir
    runs_root = ROOT / "outputs" / "runs"
    xxl_dirs = (
        sorted(
            [
                p
                for p in runs_root.iterdir()
                if p.is_dir()
                and "hamburg" in p.name.lower()
                and "xxl" in p.name.lower()
            ]
        )
        if runs_root.exists()
        else []
    )
    if xxl_dirs:
        latest_xxl = xxl_dirs[-1]
    else:
        latest_xxl = None

    # Load final selection if exists
    final_selection = None
    final_selection_file = ROOT / "outputs" / "THESIS_FINAL_SELECTION_XXL.json"
    if final_selection_file.exists():
        with open(final_selection_file) as f:
            final_selection = json.load(f)

    # Collect some metrics
    report_lines = []
    report_lines.append("# Monitor Bericht — XXL Full Run\n")
    report_lines.append(
        f"**Run started**: {datetime.fromtimestamp(start_time, timezone.utc).isoformat()}Z"
    )
    report_lines.append(
        f"**Run finished**: {datetime.fromtimestamp(end_time, timezone.utc).isoformat()}Z"
    )
    report_lines.append(f"**Elapsed (s)**: {elapsed:.1f}")
    report_lines.append(f"**Exit code**: {exit_code}\n")
    report_lines.append("**Process Info**:")
    report_lines.append(f"- PID: {process.pid}")
    report_lines.append(f"- PGID: {pgid if pgid is not None else 'N/A'}")
    report_lines.append(f"- Log: `{ACTIVE_LOG}`\n")

    report_lines.append("## Observed phase events")
    for e in phase_events:
        report_lines.append(f"- {e}")

    report_lines.append("\n## Artifacts")
    if latest_xxl:
        report_lines.append(f"- XXL run dir: {latest_xxl}")
        trials_path = latest_xxl / "results" / "trials.csv"
        if not trials_path.exists() and not args.no_reconstruct:
            _monitor_log(
                "trials.csv missing; attempting reconstruction from optuna_study.db",
                ACTIVE_LOG,
            )
            ok = _reconstruct_trials_from_db(latest_xxl, ACTIVE_LOG)
            if ok:
                report_lines.append(
                    f"  - trials.csv: reconstructed from optuna_study.db -> {trials_path}"
                )
            else:
                report_lines.append(
                    "  - trials.csv: missing and reconstruction failed/skipped"
                )
        if trials_path.exists():
            report_lines.append(
                f"  - trials.csv: {trials_path} (size: { trials_path.stat().st_size } bytes)"
            )
    else:
        report_lines.append("- XXL run dir: Not found")

    if final_selection:
        report_lines.append(f"- Final selection JSON: {final_selection_file}")
        report_lines.append(
            f"  - Best value: {final_selection.get('best_value')} @ trial #{final_selection.get('best_trial')}"
        )
        report_lines.append(f"  - n_trials recorded: {final_selection.get('n_trials')}")
    else:
        report_lines.append("- Final selection JSON: Not found")

    # Convergence baseline analysis (attempt)
    try:
        from scripts.xxl_KDR146_run_thesis_complete import (
            _validate_convergence_from_validation_data,
        )

        conv = _validate_convergence_from_validation_data(ROOT)
        if conv:
            report_lines.append("\n## Convergence baseline analysis")
            report_lines.append(f"- n_seeds_analyzed: {conv['n_seeds_analyzed']}")
            report_lines.append(
                f"- convergence_99_trials_median: {conv['convergence_99_trials_median']}"
            )
            report_lines.append(
                f"- convergence_99_trials_min: {conv['convergence_99_trials_min']}"
            )
            report_lines.append(
                f"- convergence_99_trials_max: {conv['convergence_99_trials_max']}"
            )
        else:
            report_lines.append("\n## Convergence baseline analysis")
            report_lines.append(
                "- Could not compute convergence baseline from existing validation data (insufficient/missing runs)"
            )
    except Exception as e:
        report_lines.append("\n## Convergence baseline analysis")
        report_lines.append(f"- Error while computing baseline: {e}")

    # Basic config validation for the discovered XXL run (if any)
    config_issues = []
    if latest_xxl:
        cfg_path = latest_xxl / "config" / "config_optuna.yaml"
        if cfg_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                sampler = cfg.get("sampler")
                n_trials_cfg = cfg.get("n_trials")
                n_candidates_cfg = cfg.get("n_candidates")

                if sampler and str(sampler).lower() != "cmaes":
                    config_issues.append(f"unexpected sampler: {sampler}")

                try:
                    if n_trials_cfg is not None and int(n_trials_cfg) < 400:
                        config_issues.append(f"n_trials too small: {n_trials_cfg}")
                except Exception:
                    config_issues.append(f"n_trials not parseable: {n_trials_cfg}")

                try:
                    if n_candidates_cfg is not None and int(n_candidates_cfg) != 673:
                        config_issues.append(
                            f"n_candidates mismatch: {n_candidates_cfg}"
                        )
                except Exception:
                    config_issues.append(
                        f"n_candidates not parseable: {n_candidates_cfg}"
                    )
            except Exception as e:
                config_issues.append(f"failed to parse config: {e}")

    if config_issues:
        report_lines.append("\n## Configuration issues detected")
        for issue in config_issues:
            report_lines.append(f"- {issue}")

    # Add section with log excerpts
    report_lines.append("\n## Log excerpts (last 500 lines)")
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        excerpt = "".join(lines[-500:])
        report_lines.append("```\n" + excerpt + "\n```")
    except Exception as e:
        report_lines.append(f"Could not read log file: {e}")

    # Write report into the monitor_reports folder under the XXL run dir if available, else into outputs/monitor_reports (timestamped)
    report_text = "\n".join(report_lines)
    # Prefer forced timestamp for deterministic tests
    report_ts = os.environ.get("MONITOR_FORCE_TS") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if latest_xxl:
        reports_dir = latest_xxl / "monitor_reports"
    else:
        reports_dir = ROOT / "outputs" / "monitor_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_md = reports_dir / f"monitor_report_{report_ts}.md"
    report_meta = reports_dir / f"monitor_meta_{report_ts}.json"
    report_latest_md = reports_dir / "monitor_report.md"
    report_latest_meta = reports_dir / "monitor_meta.json"

    # Write markdown report
    report_md.write_text(report_text)
    # Write machine-readable metadata
    # Coerce meta values to basic types for safe JSON serialization
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pid": int(process.pid) if getattr(process, "pid", None) is not None else None,
        "pgid": int(pgid) if isinstance(pgid, int) else None,
        "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
        "end_time": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
        "elapsed_sec": float(elapsed),
        "exit_code": int(exit_code) if exit_code is not None else None,
        "observed_phase_events": [str(e) for e in phase_events],
        "xxl_run_dir": str(latest_xxl) if latest_xxl else None,
        "config_issues": config_issues if config_issues else [],
        "pre_run": pre_run_meta,
        "post_run": post_run_meta,
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

    print(
        f"Wrote report to: {report_md} (latest copies: {report_latest_md}, {report_latest_meta})"
    )

    # Also copy the ACTIVE log into the run folder for completeness (versioned)
    try:
        if latest_xxl and ACTIVE_LOG.exists():
            copy_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest_log = latest_xxl / f"XXL_FULL_RUN_{copy_ts}.log"
            shutil.copy(ACTIVE_LOG, dest_log)
            print(f"Copied full log into run folder: {dest_log}")
    except Exception:
        pass

    print("Monitor finished")


if __name__ == "__main__":
    main()

from pathlib import Path
import sqlite3
import pandas as pd
from datetime import datetime


class ExperimentStateAnalyzer:
    """Inspect run directory for trials CSV and Optuna DB metadata.

    This class provides small, testable methods to summarize the state of
    an experiment run directory (presence of trials.csv, counts of completed
    trials, DB integrity check, best objective, etc.).
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir / "results"
        self.trials_csv = self.results_dir / "trials.csv"
        # canonical DB filename or candidates like optuna_study.db.bak_*
        self.db_path = self.run_dir / "optuna_study.db"

    def inspect(self) -> dict:
        res = {
            "run_dir": str(self.run_dir),
            "csv_exists": False,
            "csv_completed": 0,
            "csv_best": None,
            "db_exists": False,
            "db_integrity_ok": False,
            "db_completed": 0,
            "db_best": None,
        }

        # Inspect CSV
        try:
            if self.trials_csv.exists():
                res["csv_exists"] = True
                df = pd.read_csv(self.trials_csv)
                if "state" in df.columns:
                    res["csv_completed"] = int(df[df["state"].astype(str).str.contains("COMPLETE")].shape[0])
                else:
                    res["csv_completed"] = int(df.shape[0])
                if "value" in df.columns:
                    try:
                        res["csv_best"] = float(df["value"].max()) if not df["value"].isnull().all() else None
                    except Exception:
                        res["csv_best"] = None
        except Exception:
            pass

        # Inspect DB (prefer the most recent optuna_study.db candidate)
        try:
            if not self.db_path.exists():
                candidates = list(self.run_dir.glob("optuna_study.db*"))
                if candidates:
                    self.db_path = max(candidates, key=lambda p: p.stat().st_mtime)
            if self.db_path.exists():
                res["db_exists"] = True
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                try:
                    ic = cur.execute("PRAGMA integrity_check;").fetchone()
                    res["db_integrity_ok"] = bool(ic and (ic[0] == "ok" if isinstance(ic, tuple) else ic == "ok"))
                except Exception:
                    res["db_integrity_ok"] = False

                # Count completed trials by state or by trials table row count
                try:
                    cur.execute("SELECT trial_id, number, state FROM trials ORDER BY number ASC;")
                    trials_raw = cur.fetchall()
                    if trials_raw:
                        # Try to count when state column present
                        completed = 0
                        for tid, number, state in trials_raw:
                            if state is None:
                                completed += 1
                            else:
                                s = str(state)
                                if "COMPLETE" in s:
                                    completed += 1
                        res["db_completed"] = int(completed)
                except Exception:
                    # Fallback: count rows
                    try:
                        r = cur.execute("SELECT count(*) FROM trials;").fetchone()
                        res["db_completed"] = int(r[0]) if r else 0
                    except Exception:
                        res["db_completed"] = 0

                # Best value attempt: look at trial_values (scalar objectives)
                try:
                    cur.execute("SELECT trial_id, value FROM trial_values;")
                    vals = [r[1] for r in cur.fetchall() if r[1] is not None]
                    if vals:
                        res["db_best"] = float(max(vals))
                except Exception:
                    res["db_best"] = None

                conn.close()
        except Exception:
            pass

        return res

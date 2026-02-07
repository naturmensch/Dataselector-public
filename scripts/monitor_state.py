"""Minimal ExperimentStateAnalyzer used by the monitor for resume decisions.

This implementation is intentionally lightweight and robust:
- It avoids heavy dependencies when possible
- It performs simple checks on the run folder to compute counts and best values
- It is suitable for use in tests and small deployments
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback behavior
    pd = None


class ExperimentStateAnalyzer:
    """Analyze a run directory and return a small summary dict.

    The returned dict includes the keys used by the monitor and the recovery
    planner: 'csv_exists', 'csv_completed', 'csv_best', 'db_exists',
    'db_integrity_ok', 'db_completed', 'db_best'.
    """

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)

    def inspect(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "csv_exists": False,
            "csv_completed": 0,
            "csv_best": None,
            "db_exists": False,
            "db_integrity_ok": False,
            "db_completed": 0,
            "db_best": None,
        }

        # Check CSV
        csv_path = self.run_dir / "results" / "trials.csv"
        if csv_path.exists():
            out["csv_exists"] = True
            if pd:
                try:
                    df = pd.read_csv(csv_path)
                    # Count completed trials from 'state' if present, else from non-null datetime_complete
                    if "state" in df.columns:
                        out["csv_completed"] = int((df["state"] == "COMPLETE").sum())
                    elif "datetime_complete" in df.columns:
                        out["csv_completed"] = int(
                            df["datetime_complete"].notnull().sum()
                        )
                    else:
                        out["csv_completed"] = int(len(df))

                    if "value" in df.columns and not df["value"].isnull().all():
                        out["csv_best"] = float(df["value"].max())
                except Exception:
                    # If reading fails, conservatively leave defaults
                    pass

        # Check DB
        db_path = self.run_dir / "optuna_study.db"
        if db_path.exists():
            out["db_exists"] = True
            try:
                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                res = cur.execute("PRAGMA integrity_check;").fetchone()
                if res and (isinstance(res, tuple) and res[0] == "ok") or res == "ok":
                    out["db_integrity_ok"] = True
                # Try simple trial state count in direct sqlite: some schema variants exist
                try:
                    cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE';")
                    cnt = cur.fetchone()
                    if cnt and cnt[0] is not None:
                        out["db_completed"] = int(cnt[0])
                except Exception:
                    pass
                # Try to find best value
                try:
                    cur.execute(
                        "SELECT value FROM trials WHERE value IS NOT NULL ORDER BY value DESC LIMIT 1;"
                    )
                    best = cur.fetchone()
                    if best and best[0] is not None:
                        out["db_best"] = float(best[0])
                except Exception:
                    pass
                conn.close()
            except Exception:
                # If sqlite access fails, keep defaults
                pass

        return out

"""Minimal run-state analyzer for resume decisions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class ExperimentStateAnalyzer:
    """Analyze a run directory and return monitor-friendly state."""

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

        csv_path = self.run_dir / "results" / "trials.csv"
        if csv_path.exists():
            out["csv_exists"] = True
            if pd:
                try:
                    df = pd.read_csv(csv_path)
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
                    pass

        db_path = self.run_dir / "optuna_study.db"
        if db_path.exists():
            out["db_exists"] = True
            try:
                conn = sqlite3.connect(str(db_path))
                cur = conn.cursor()
                integrity = cur.execute("PRAGMA integrity_check;").fetchone()
                if (
                    integrity
                    and isinstance(integrity, tuple)
                    and integrity[0] == "ok"
                ) or integrity == "ok":
                    out["db_integrity_ok"] = True
                try:
                    cur.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE';")
                    cnt = cur.fetchone()
                    if cnt and cnt[0] is not None:
                        out["db_completed"] = int(cnt[0])
                except Exception:
                    pass
                try:
                    cur.execute(
                        "SELECT value FROM trials WHERE value IS NOT NULL "
                        "ORDER BY value DESC LIMIT 1;"
                    )
                    best = cur.fetchone()
                    if best and best[0] is not None:
                        out["db_best"] = float(best[0])
                except Exception:
                    pass
                conn.close()
            except Exception:
                pass

        return out

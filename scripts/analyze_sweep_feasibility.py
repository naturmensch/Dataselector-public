#!/usr/bin/env python3
"""Analyze feasibility summaries from coarse/fine sweeps and produce a combined report.

Usage:
    python scripts/analyze_sweep_feasibility.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

summaries = []
for name in ["coarse_sweep", "fine_sweep"]:
    path = OUT_DIR / name / "feasibility_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["sweep"] = name
        summaries.append(df)

if not summaries:
    print("No feasibility summaries found. Run sweeps first.")
    raise SystemExit(1)

full = pd.concat(summaries, ignore_index=True)
report_dir = OUT_DIR / "feasibility_analysis"
report_dir.mkdir(exist_ok=True)

# Summary table
agg = full.groupby(["sweep", "min_distance_km"]).agg(
    total_runs=("total_runs", "sum"),
    infeasible_count=("infeasible_count", "sum"),
    median_n_selected=("median_n_selected", "median"),
)
agg["infeasible_pct"] = agg["infeasible_count"] / agg["total_runs"] * 100.0
agg = agg.reset_index()
agg.to_csv(report_dir / "feasibility_combined_summary.csv", index=False)
print(f"Wrote combined feasibility summary: {report_dir / 'feasibility_combined_summary.csv'}")

# Plot infeasible_pct by min_distance
plt.figure(figsize=(8, 4))
for sweep, grp in agg.groupby("sweep"):
    plt.plot(grp["min_distance_km"], grp["infeasible_pct"], marker="o", label=sweep)

plt.xlabel("min_distance_km")
plt.ylabel("Infeasible Runs (%)")
plt.title("Sweep Feasibility by min_distance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(report_dir / "feasibility_plot.png", dpi=200)
print(f"Wrote plot: {report_dir / 'feasibility_plot.png'}")

print("Done.")
#!/usr/bin/env python3
"""Generate a human-readable experiment report for a run folder.

Usage:
  python scripts/generate_experiment_report.py --outdir outputs/experiments/run_20260115T...

The script collects:
- run metadata (timestamp, parameters passed via environment/args)
- list of produced files in the outdir
- snippets of step logs (coarse/fine/optuna/bootstrap/final)
- the applied Optuna config (if present)
- a short summary with counts and key metrics (if present in CSV outputs)
- Optionally set `DATASELECTOR_OUTPUTS_ROOT` to override the global `outputs/` root (useful in tests)

Report written to <outdir>/experiment_report.md
"""

import argparse
import csv
import os
from pathlib import Path

import yaml


def summarize_csv_metrics(csv_path: Path) -> dict:
    # Try to extract some summary metrics from known CSVs
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if not rows:
                return {}
            # return first row numeric-summaries for common fields
            keys = [
                "n_selected",
                "temporal_std",
                "spatial_mean_km",
                "wwi_percent",
                "clusters_covered",
                "total_runs",
                "infeasible_count",
                "infeasible_pct",
                "median_n_selected",
            ]
            summary = {}
            for k in keys:
                if k in rows[0]:
                    try:
                        summary[k] = float(rows[0][k])
                    except Exception:
                        summary[k] = rows[0][k]
            return summary
    except Exception:
        return {}


def collect_logs(outdir: Path):
    logs = {}
    # include adaptive/tuning steps in logs for modern runs
    for step in [
        "coarse_sweep",
        "fine_sweep",
        "optuna",
        "bootstrap",
        "final_selection",
        "adaptive_pipeline",
        "tuning_weights",
    ]:
        p = outdir / f"{step}.log"
        if p.exists():
            try:
                logs[step] = p.read_text()[:20000]
            except Exception:
                logs[step] = f"Could not read log: {p}"
    return logs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        required=True,
        help="Experiment run folder (outputs/experiments/run_...)",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise SystemExit(f"Outdir not found: {outdir}")

    report_path = outdir / "experiment_report.md"

    # Basic header (metadata will be integrated into the written report later)
    # _meta intentionally omitted to avoid unused variable lint (was: run_dir/generator info)

    # list files (relative to outdir)
    files = sorted(
        [str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file()]
    )

    # try to load applied optuna config if present
    optuna_cfg = None
    for candidate in [
        "pipeline_config.optuna.yaml",
        "optuna_test_config.yaml",
        "pipeline_config.yaml.optuna_bak",
    ]:
        p = outdir / candidate
        if p.exists():
            try:
                optuna_cfg = yaml.safe_load(p.read_text())
                break
            except Exception:
                optuna_cfg = {"path": str(p)}
                break

    # find common CSVs and summarize (include pareto_solutions.csv with global search fallback)
    ROOT = Path(__file__).resolve().parents[1]
    env_outputs = os.environ.get("DATASELECTOR_OUTPUTS_ROOT")
    outputs_root = Path(env_outputs) if env_outputs else (ROOT / "outputs")
    summaries = {}
    csv_names = [
        "coarse_sweep_results.csv",
        "fine_sweep_results.csv",
        "optuna_results.csv",
        "bootstrap_results_summary.csv",
        "feasibility_combined_summary.csv",
        "optuna_convergence_analysis.csv",
        "pareto_solutions.csv",
    ]
    for name in csv_names:
        # prefer outdir, then parent, then global outputs (for pareto)
        p = outdir / name
        if not p.exists() and (outdir.parent / name).exists():
            p = outdir.parent / name
        if not p.exists() and name == "pareto_solutions.csv":
            # search in outputs for pareto files and prefer tuning_weights
            matches = sorted(list(outputs_root.rglob("pareto_solutions.csv")))
            if matches:
                preferred = None
                for m in matches:
                    if "tuning_weights" in str(m):
                        preferred = m
                        break
                p = Path(preferred or matches[-1])
        if p.exists():
            summaries[name] = summarize_csv_metrics(p)

    logs = collect_logs(outdir)

    # Compose report
    lines = []
    lines.append("# Experiment Report\n")
    lines.append(f"**Run folder**: `{outdir}`\n")
    lines.append("---\n")

    lines.append("## Files produced\n")
    for f in files:
        lines.append(f"- {f}")
    lines.append("\n")

    if optuna_cfg:
        lines.append("## Optuna configuration (applied)\n")
        lines.append("```yaml")
        try:
            lines.append(yaml.safe_dump(optuna_cfg, sort_keys=False))
        except Exception:
            lines.append(f"# Could not render config; path: {p}")
        lines.append("```\n")

    if summaries:
        lines.append("## Quick metrics summary\n")
        for k, v in summaries.items():
            lines.append(f"### {k}")
            for kk, vv in v.items():
                lines.append(f"- {kk}: {vv}")
            lines.append("\n")

    # Feasibility analysis section
    feasibility_dir = outdir.parent / "feasibility_analysis"
    if feasibility_dir.exists():
        lines.append("## Feasibility Analysis\n")
        feas_summary = feasibility_dir / "feasibility_combined_summary.csv"
        if feas_summary.exists():
            lines.append(
                f"Combined feasibility summary available: `{feas_summary.name}`\n"
            )
        feas_plot = feasibility_dir / "feasibility_plot.png"
        if feas_plot.exists():
            lines.append(f"Feasibility plot: `{feas_plot.name}`\n")
        lines.append("\n")

    # Convergence analysis section
    if (
        "optuna_convergence_analysis.csv" in summaries
        or (outdir.parent / "optuna_convergence_analysis.csv").exists()
    ):
        lines.append("## Convergence Analysis\n")
        lines.append(
            "Optuna convergence analysis available. Check CSV and plots in outputs.\n\n"
        )

    # Pareto solutions found (LHS or Fine)
    pareto_files = sorted([str(p) for p in outputs_root.rglob("pareto_solutions.csv")])
    if pareto_files:
        lines.append("## Pareto solutions found\n")
        for p in pareto_files:
            lines.append(f"- `{p}`")
        lines.append("\n")

    if logs:
        lines.append("## Step log snippets\n")
        for step, content in logs.items():
            lines.append(f"### {step}")
            # include only first 2000 chars
            lines.append("```")
            lines.append(content[:2000])
            lines.append("```\n")

    lines.append("---\n")
    lines.append("Report generated by `scripts/generate_experiment_report.py`\n")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()

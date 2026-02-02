from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ExplorationPlan:
    method: str
    n_initial_raw: int
    n_initial_final: int
    adjustment_rule: str
    benchmark_csv: str
    benchmark_plot: str
    n_dims: int
    n_samples_list: list[int]
    selected_score: float | None
    timestamp_utc: str
    git_commit: str | None
    python_executable: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit(repo_root: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode == 0:
            return (p.stdout or "").strip() or None
    except Exception:
        return None
    return None


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _round_up_to_multiple(n: int, m: int) -> int:
    if m <= 0:
        return n
    return int(math.ceil(n / m) * m)


def _adjust_n_initial(
    method: str, n_raw: int, *, min_sobol: int = 32, lhs_multiple: int = 8
) -> tuple[int, str]:
    method_norm = method.strip().lower()
    if method_norm in {"sobol", "qmc"}:
        adjusted = max(min_sobol, _next_power_of_two(n_raw))
        return adjusted, "sobol_power_of_two_round_up_min32"
    if method_norm in {"lhs"}:
        adjusted = _round_up_to_multiple(n_raw, lhs_multiple)
        return adjusted, f"lhs_round_up_to_multiple_{lhs_multiple}"
    # random / fallback
    return n_raw, "no_adjustment"


def _choose_best_method(
    csv_path: Path, *, prefer: str | None = None
) -> tuple[str, float | None]:
    """Choose method using benchmark CSV.

    The legacy benchmark writes rows with columns:
      - method
      - mean_time
      - std_time
      - discrepancy
      - min_distance

    We choose the method that minimizes discrepancy, using mean_time as a tie-breaker.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Benchmark CSV is empty: {csv_path}")
    # Normalize method names
    df["method_norm"] = df["method"].astype(str).str.strip().str.lower()

    if prefer:
        pref = prefer.strip().lower()
        cand = df[df["method_norm"] == pref]
        if not cand.empty:
            # still return its mean discrepancy as score
            score = (
                float(cand["discrepancy"].mean())
                if "discrepancy" in cand.columns
                else None
            )
            return pref, score

    # Filter to valid numeric discrepancy
    if "discrepancy" not in df.columns:
        raise RuntimeError(f"Benchmark CSV missing 'discrepancy' column: {csv_path}")
    cand = df[df["discrepancy"].notna()].copy()
    if cand.empty:
        raise RuntimeError(f"No valid discrepancy values in benchmark CSV: {csv_path}")

    # Aggregate per method
    agg = (
        cand.groupby("method_norm")
        .agg(discrepancy_mean=("discrepancy", "mean"), mean_time=("mean_time", "mean"))
        .reset_index()
        .sort_values(["discrepancy_mean", "mean_time"], ascending=[True, True])
    )
    best = str(agg.iloc[0]["method_norm"])
    best_score = float(agg.iloc[0]["discrepancy_mean"])
    return best, best_score


def main(argv: list[str] | None = None) -> int:
    """Benchmark sampling methods and write an exploration plan artifact.

    Canonical usage:
        python -m dataselector benchmark-sampling -- <args>
    """

    parser = argparse.ArgumentParser(
        prog="dataselector benchmark-sampling",
        description=(
            "Benchmark initial sampling methods and persist an exploration plan artifact"
        ),
    )
    parser.add_argument(
        "--n-initial-raw",
        type=int,
        default=20,
        help="Raw heuristic sample count before method-specific adjustment (default: 20)",
    )
    parser.add_argument(
        "--n-dims",
        type=int,
        default=4,
        help=(
            "Dimensionality for the benchmark (default: 4; matches "
            "alpha/beta/gamma/min_distance)"
        ),
    )
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Sample sizes to benchmark (default: 32 64 128)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Repeats per method/sample size (default: 5)",
    )
    parser.add_argument(
        "--prefer-method",
        type=str,
        default=None,
        help=(
            "If set and available in results, force the chosen method "
            "(e.g. sobol, lhs, random)"
        ),
    )
    parser.add_argument(
        "--require-methods",
        nargs="+",
        default=None,
        help=(
            "If set, fail if any of these methods are missing from results "
            "(e.g. sobol lhs)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help=(
            "Output directory for benchmark artifacts and exploration plan "
            "(default: outputs)"
        ),
    )
    parser.add_argument(
        "--min-sobol",
        type=int,
        default=32,
        help="Minimum Sobol sample size when adjusting n_initial (default: 32)",
    )
    parser.add_argument(
        "--lhs-multiple",
        type=int,
        default=8,
        help="Round LHS sample count up to a multiple of this value (default: 8)",
    )
    args = parser.parse_args(list(argv or []))

    repo_root = _repo_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the legacy benchmark script in a subprocess (same interpreter)
    bench_csv = out_dir / "sampling_benchmark_results.csv"
    bench_plot = out_dir / "sampling_benchmark_plots.png"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "benchmark_sampling_methods.py"),
        "--n-samples",
        *[str(x) for x in args.n_samples],
        "--n-dims",
        str(args.n_dims),
        "--n-repeats",
        str(args.n_repeats),
        "--output-dir",
        str(out_dir),
    ]

    p = subprocess.run(cmd, cwd=str(repo_root))
    if p.returncode != 0:
        return int(p.returncode)

    if not bench_csv.exists():
        raise RuntimeError(f"Benchmark did not write expected CSV: {bench_csv}")
    if not bench_plot.exists():
        # Plot may be missing if matplotlib failed; treat as error for thesis-grade usage.
        raise RuntimeError(f"Benchmark did not write expected plot: {bench_plot}")

    chosen_method, score = _choose_best_method(bench_csv, prefer=args.prefer_method)

    # Ensure required methods are present if requested
    if args.require_methods:
        import pandas as pd

        df = pd.read_csv(bench_csv)
        present = set(df["method"].astype(str).str.strip().str.lower().unique())
        missing = [
            m
            for m in [x.strip().lower() for x in args.require_methods]
            if m not in present
        ]
        if missing:
            raise RuntimeError(
                "Required benchmark methods missing from results: " + ", ".join(missing)
            )

    n_final, rule = _adjust_n_initial(
        chosen_method,
        int(args.n_initial_raw),
        min_sobol=int(args.min_sobol),
        lhs_multiple=int(args.lhs_multiple),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = ExplorationPlan(
        method=chosen_method,
        n_initial_raw=int(args.n_initial_raw),
        n_initial_final=int(n_final),
        adjustment_rule=rule,
        benchmark_csv=str(bench_csv),
        benchmark_plot=str(bench_plot),
        n_dims=int(args.n_dims),
        n_samples_list=[int(x) for x in args.n_samples],
        selected_score=score,
        timestamp_utc=ts,
        git_commit=_git_commit(repo_root),
        python_executable=sys.executable,
    )

    plans_dir = out_dir / "exploration_plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    latest = out_dir / "exploration_plan_latest.json"
    stamped = plans_dir / f"exploration_plan_{ts}.json"

    payload = asdict(plan)
    latest.write_text(json.dumps(payload, indent=2))
    stamped.write_text(json.dumps(payload, indent=2))

    # Also export to environment for downstream wrappers if this subcommand is used in-process.
    os.environ["DATASELECTOR_EXPLORATION_PLAN"] = str(latest)

    print(f"Wrote exploration plan: {latest}")
    print(f"Wrote exploration plan (timestamped): {stamped}")
    print(
        f"Chosen method: {chosen_method} | n_initial: {args.n_initial_raw} -> "
        f"{n_final} ({rule})"
    )
    return 0

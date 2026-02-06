from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataselector")
    sub = p.add_subparsers(dest="cmd", required=True)

    autoscale = sub.add_parser(
        "autoscale", help="Run Optuna autoscale (direct workflow)"
    )
    autoscale.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV metadata file (default: generate synthetic)",
    )
    autoscale.add_argument(
        "--n-trials",
        type=int,
        nargs="+",
        default=[20, 40, 80, 160],
        help="Number of trials per stage",
    )
    autoscale.add_argument(
        "--stages",
        nargs="+",
        default=["50", "100", "300", "full"],
        help="Sample sizes per stage ('full' = all candidates)",
    )
    autoscale.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    autoscale.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of candidates (fallback if CSV not provided)",
    )
    autoscale.add_argument("--dim", type=int, default=256, help="Feature dimension")
    autoscale.add_argument("--seed", type=int, default=42, help="Random seed")
    autoscale.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of stages without improvement to trigger early stopping",
    )
    autoscale.add_argument(
        "--pre-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional pre-selected tile names",
    )
    autoscale.add_argument(
        "--pre-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional pre-selected tile indices",
    )

    suite = sub.add_parser(
        "sampler-suite", help="Run the sampler suite (optionally includes autoscale)"
    )
    suite.add_argument("args", nargs=argparse.REMAINDER)

    xxl = sub.add_parser("xxl", help="Run XXL thesis pipeline")
    xxl.add_argument(
        "--best-sampler",
        type=str,
        default="tpe",
        help="Best sampler from suite (qmc/tpe/cmaes)",
    )
    xxl.add_argument(
        "--phase",
        type=str,
        choices=["full", "finalize"],
        default="full",
        help="Run only a sub-phase: finalize (bootstrap+finalization) or full (default)",
    )
    xxl.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to operate on (for finalize phase)",
    )
    xxl.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )
    xxl.add_argument(
        "--n-candidates",
        type=int,
        default=676,
        help="Number of candidates",
    )
    xxl.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (reduced settings for testing)",
    )
    xxl.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    xxl_mon = sub.add_parser(
        "xxl-monitor",
        help="Run XXL full-run monitor (scripts/xxl_full_run_monitor.py)",
    )
    xxl_mon.add_argument("args", nargs=argparse.REMAINDER)

    opt = sub.add_parser(
        "optuna-optimize",
        help="Run Optuna optimization (direct workflow)",
    )
    opt.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    opt.add_argument(
        "--n-candidates", type=int, default=500, help="Number of candidates"
    )
    opt.add_argument("--dim", type=int, default=256, help="Feature dimension")
    opt.add_argument(
        "--n-samples", type=int, default=34, help="Number of samples to select"
    )
    opt.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode with reduced trials/candidates",
    )
    opt.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Alternate workspace path for outputs/data",
    )
    opt.add_argument(
        "--n-samples-min",
        type=int,
        default=None,
        help="Min samples for range (overrides n-samples if set)",
    )
    opt.add_argument(
        "--n-samples-max",
        type=int,
        default=None,
        help="Max samples for range (ignored if n-samples-min not set)",
    )
    opt.add_argument(
        "--min-distance-km", type=int, default=28, help="Minimum distance in km"
    )
    opt.add_argument("--seed", type=int, default=42, help="Random seed")
    opt.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save Optuna study and results every N trials (0 disables)",
    )
    opt.add_argument(
        "--sampler", type=str, default="tpe", help="Optuna sampler (qmc, tpe, cmaes)"
    )
    opt.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    opt.add_argument(
        "--use-study-db",
        action="store_true",
        help="Create/use default outputs/optuna_study.db for persistent storage",
    )
    opt.add_argument(
        "--study-db",
        type=str,
        default=None,
        help="Path to SQLite DB file for Optuna storage",
    )
    opt.add_argument(
        "--constrain-a-min", type=float, default=None, help="Constrain alpha min"
    )
    opt.add_argument(
        "--constrain-a-max", type=float, default=None, help="Constrain alpha max"
    )
    opt.add_argument(
        "--constrain-b-min", type=float, default=None, help="Constrain beta min"
    )
    opt.add_argument(
        "--constrain-b-max", type=float, default=None, help="Constrain beta max"
    )
    opt.add_argument(
        "--constrain-c-min", type=float, default=None, help="Constrain gamma min"
    )
    opt.add_argument(
        "--constrain-c-max", type=float, default=None, help="Constrain gamma max"
    )
    opt.add_argument(
        "--constrain-min-dist-min",
        type=int,
        default=None,
        help="Constrain min_distance lower bound",
    )
    opt.add_argument(
        "--constrain-min-dist-max",
        type=int,
        default=None,
        help="Constrain min_distance upper bound",
    )

    fin = sub.add_parser(
        "final-selection",
        help="Run final selection runner (scripts/final_selection.py)",
    )
    fin.add_argument("args", nargs=argparse.REMAINDER)

    pipe = sub.add_parser(
        "thesis-pipeline",
        help=(
            "Run full thesis pipeline orchestrator "
            "(scripts/run_complete_thesis_pipeline.sh)"
        ),
    )
    pipe.add_argument("args", nargs=argparse.REMAINDER)

    adaptive = sub.add_parser(
        "adaptive-pipeline",
        help="Run adaptive multi-stage pipeline (direct workflow)",
    )
    adaptive.add_argument(
        "--experiment-name",
        type=str,
        default="adaptive_pipeline",
        help="Name for this experiment run",
    )
    adaptive.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to tile metadata CSV (default: data/new_all_tiles.csv)",
    )
    adaptive.add_argument(
        "--n-lhs",
        type=int,
        default=None,
        help="Number of LHS samples (None = adaptive)",
    )
    adaptive.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials"
    )
    adaptive.add_argument(
        "--n-boot", type=int, default=500, help="Number of bootstrap resamples"
    )
    adaptive.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Total candidate pool size (None = dataset size)",
    )
    adaptive.add_argument(
        "--n-dimensions",
        type=int,
        default=9,
        help="Number of parameter dimensions for adaptive sizing",
    )
    adaptive.add_argument(
        "--sampler",
        type=str,
        choices=["lhs", "sobol"],
        default="lhs",
        help="Exploration sampler type",
    )
    adaptive.add_argument(
        "--optuna-sampler",
        type=str,
        default="TPESampler",
        help="Optuna sampler class name",
    )
    adaptive.add_argument("--seed", type=int, default=42, help="Random seed")
    adaptive.add_argument(
        "--n-initial-strategy",
        type=str,
        choices=["conservative", "moderate", "aggressive"],
        default="conservative",
        help="Strategy for adaptive n_lhs computation",
    )
    adaptive.add_argument(
        "--n-samples", type=int, default=None, help="Fixed n_samples for Optuna"
    )
    adaptive.add_argument(
        "--n-samples-min", type=int, default=None, help="Min n_samples (range mode)"
    )
    adaptive.add_argument(
        "--n-samples-max", type=int, default=None, help="Max n_samples (range mode)"
    )
    adaptive.add_argument(
        "--fine-max-runs", type=int, default=None, help="Max runs for fine sweep"
    )
    adaptive.add_argument(
        "--skip-exploration",
        action="store_true",
        help="Skip exploration phase",
    )
    adaptive.add_argument("--skip-fine", action="store_true", help="Skip fine sweep")
    adaptive.add_argument(
        "--skip-optuna", action="store_true", help="Skip Optuna phase"
    )
    adaptive.add_argument(
        "--skip-bootstrap-injection",
        action="store_true",
        help="Skip bootstrap-best config injection",
    )
    adaptive.add_argument(
        "--continue-on-analysis-failure",
        action="store_true",
        help="Continue pipeline on analysis errors",
    )
    adaptive.add_argument("--dry-run", action="store_true", help="Dry run mode")
    adaptive.add_argument(
        "--pre-names", nargs="*", default=None, help="Pre-selected tile names"
    )
    adaptive.add_argument(
        "--pre-indices",
        nargs="*",
        type=int,
        default=None,
        help="Pre-selected indices",
    )
    adaptive.add_argument(
        "--hamburg", action="store_true", help="Add Hamburg to pre-names"
    )
    adaptive.add_argument(
        "--KDR146", action="store_true", help="Add KDR_146 to pre-names"
    )

    thesis_suite = sub.add_parser(
        "thesis-sampler-suite",
        help="Run thesis-grade sampler evaluation (direct workflow)",
    )
    thesis_suite.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        help="Random seeds for reproducibility (default: 10 seeds)",
    )
    thesis_suite.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Trials per sampler (default: 1000)",
    )
    thesis_suite.add_argument(
        "--datasets",
        nargs="+",
        default=["hamburg", "kdr100"],
        help="Datasets to compare on (default: hamburg + kdr100)",
    )
    thesis_suite.add_argument(
        "--samplers",
        nargs="+",
        default=["qmc", "tpe", "cmaes"],
        help="Samplers to compare (default: QMC, TPE, CMA-ES)",
    )
    thesis_suite.add_argument(
        "--sequential", action="store_true", help="Run sequentially"
    )
    thesis_suite.add_argument(
        "--n-trials-full",
        type=int,
        default=2000,
        help="Trials for full adaptive runs",
    )
    thesis_suite.add_argument("--n-candidates", type=int, default=None)
    thesis_suite.add_argument(
        "--autoscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run autoscale before sampler suite",
    )

    cs = sub.add_parser(
        "compare-samplers",
        help="Compare samplers across seeds (direct workflow)",
    )
    cs.add_argument("args", nargs=argparse.REMAINDER)

    bench = sub.add_parser(
        "benchmark-sampling",
        help=(
            "Benchmark initial sampling methods and write an exploration plan artifact"
        ),
    )
    bench.add_argument("args", nargs=argparse.REMAINDER)
    # report (alias for generate-reports with subcommands)
    report_cmd = sub.add_parser("report", help="Generate reports and visualizations")
    report_cmd.add_argument("args", nargs=argparse.REMAINDER)
    rep = sub.add_parser(
        "generate-reports",
        help="Generate summary plots/reports (direct workflow)",
    )
    rep.add_argument("args", nargs=argparse.REMAINDER)

    # build-tiles
    build_tiles_cmd = sub.add_parser(
        "build-tiles", help="Build new_all_tiles.csv from image directory scan"
    )
    build_tiles_cmd.add_argument(
        "--image-dir", required=True, help="Directory containing image files"
    )
    build_tiles_cmd.add_argument(
        "--out", default="data/new_all_tiles.csv", help="Output CSV path"
    )
    build_tiles_cmd.add_argument(
        "--force-source", help="Force a specific source CSV to use for provenance"
    )

    # bootstrap
    bootstrap_cmd = sub.add_parser(
        "bootstrap", help="Bootstrap uncertainty quantification workflows"
    )
    bootstrap_cmd.add_argument("args", nargs=argparse.REMAINDER)

    # Tools subcommand group
    tools_parser = sub.add_parser("tools", help="Administrative & maintenance tools")
    tools_sub = tools_parser.add_subparsers(dest="tool_cmd", required=True)

    # check-protected
    check_prot = tools_sub.add_parser(
        "check-protected", help="Check for modifications inside protected paths"
    )
    check_prot.add_argument(
        "--list", action="store_true", help="List protected paths and exit"
    )
    check_prot.add_argument(
        "--all", action="store_true", help="Check all tracked files (git ls-files)"
    )
    check_prot.add_argument(
        "--protect", action="append", help="Add protected path (repeatable)"
    )

    # check-env
    check_env = tools_sub.add_parser(
        "check-env", help="Check environment usage in scripts/CI"
    )
    check_env.add_argument(
        "paths",
        nargs="*",
        help="Paths to scan (defaults to scripts, Makefile, .github/workflows)",
    )

    # check-geo
    check_geo_cmd = tools_sub.add_parser(
        "check-geo",
        help="Check geo dependencies (geopandas, pyproj, shapely, fiona, rtree)",
    )

    # verify-archive
    verify_arch = tools_sub.add_parser(
        "verify-archive", help="Verify no references to archived tests"
    )
    verify_arch.add_argument(
        "--fail-on-reference",
        action="store_true",
        help="Exit with code 1 if references found",
    )

    # archive-outputs
    archive_out = tools_sub.add_parser(
        "archive-outputs", help="Archive outputs directory"
    )
    archive_out.add_argument("--outputs", required=True, help="Directory to archive")
    archive_out.add_argument(
        "--dest", default="data/archive", help="Destination directory"
    )
    archive_out.add_argument("--exclude", nargs="*", help="Glob patterns to exclude")

    # list-archives
    list_arch = tools_sub.add_parser("list-archives", help="List available archives")
    list_arch.add_argument(
        "--dir", default="data/archive", help="Directory containing archives"
    )

    # align-audit
    align = tools_sub.add_parser("align-audit", help="Audit CSV vs raster alignment")
    align.add_argument(
        "--csv", default="data/new_all_tiles.csv", help="CSV with tile metadata"
    )
    align.add_argument("--base-dir", default=".", help="Base dir for image paths")
    align.add_argument("--aux-dir", help="Optional directory for .aux.xml files")
    align.add_argument(
        "--target-crs", default="EPSG:25832", help="Target CRS for metric comparisons"
    )
    align.add_argument(
        "--max-offset-m",
        type=float,
        default=1000.0,
        help="Threshold (m) for outlier reporting",
    )
    align.add_argument("--out", help="Path for JSON report")
    align.add_argument("--plot", help="Path for PNG plot")

    # clean-workspace
    clean = tools_sub.add_parser("clean-workspace", help="Clean workspace files")
    clean.add_argument(
        "--delete-outputs",
        action="store_true",
        help="Delete outputs/ (except protected)",
    )
    clean.add_argument(
        "--delete-cache",
        action="store_true",
        help="Delete __pycache__ and .pytest_cache",
    )
    clean.add_argument(
        "--delete-venvs", action="store_true", help="Delete .venv and venv"
    )
    clean.add_argument("--archive", help="Archive instead of delete (path to .tar.gz)")
    clean.add_argument(
        "--yes",
        dest="dry_run",
        action="store_false",
        help="Actually perform cleanup (default is dry-run)",
    )

    # docs-link-check
    docs_check = tools_sub.add_parser(
        "docs-link-check", help="Check for broken documentation links"
    )

    # docs-link-autofix
    docs_fix = tools_sub.add_parser(
        "docs-link-autofix", help="Auto-fix broken documentation links"
    )
    docs_fix.add_argument(
        "--yes",
        dest="dry_run",
        action="store_false",
        help="Actually fix links (default is dry-run)",
    )
    docs_fix.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Don't backup original files",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    ns = p.parse_args(argv)

    if ns.cmd == "autoscale":
        from dataselector.workflows.autoscale import main as run

        # Convert namespace to argv format for the workflow main()
        argv = []
        if ns.csv:
            argv.extend(["--csv", ns.csv])
        if ns.n_trials:
            argv.append("--n-trials")
            argv.extend(str(t) for t in ns.n_trials)
        if ns.stages:
            argv.append("--stages")
            argv.extend(str(s) for s in ns.stages)
        if ns.output_dir:
            argv.extend(["--output-dir", ns.output_dir])
        if ns.n_candidates:
            argv.extend(["--n-candidates", str(ns.n_candidates)])
        if ns.dim:
            argv.extend(["--dim", str(ns.dim)])
        if ns.seed:
            argv.extend(["--seed", str(ns.seed)])
        if ns.patience:
            argv.extend(["--patience", str(ns.patience)])
        if ns.pre_names:
            argv.append("--pre-names")
            argv.extend(ns.pre_names)
        if ns.pre_indices:
            argv.append("--pre-indices")
            argv.extend(str(i) for i in ns.pre_indices)
        return run(argv)

    if ns.cmd == "sampler-suite":
        from dataselector.workflows.sampler_suite import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "xxl":
        from dataselector.workflows.xxl import main as run

        # Convert namespace to argv format
        argv = ["--best-sampler", ns.best_sampler]
        argv.extend(["--phase", ns.phase])
        if ns.run_dir:
            argv.extend(["--run-dir", ns.run_dir])
        argv.extend(["--output-dir", ns.output_dir])
        argv.extend(["--n-candidates", str(ns.n_candidates)])
        if ns.smoke:
            argv.append("--smoke")
        if ns.seed is not None:
            argv.extend(["--seed", str(ns.seed)])
        return run(argv)

    if ns.cmd == "xxl-monitor":
        from dataselector.workflows.xxl_monitor import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "optuna-optimize":
        from dataselector.workflows.optuna_optimize import main as run

        # Convert namespace to argv format
        argv = []
        argv.extend(["--n-trials", str(ns.n_trials)])
        argv.extend(["--n-candidates", str(ns.n_candidates)])
        argv.extend(["--dim", str(ns.dim)])
        argv.extend(["--n-samples", str(ns.n_samples)])
        if ns.smoke:
            argv.append("--smoke")
        if ns.workspace:
            argv.extend(["--workspace", ns.workspace])
        if ns.n_samples_min is not None:
            argv.extend(["--n-samples-min", str(ns.n_samples_min)])
        if ns.n_samples_max is not None:
            argv.extend(["--n-samples-max", str(ns.n_samples_max)])
        argv.extend(["--min-distance-km", str(ns.min_distance_km)])
        argv.extend(["--seed", str(ns.seed)])
        if ns.checkpoint_every:
            argv.extend(["--checkpoint-every", str(ns.checkpoint_every)])
        argv.extend(["--sampler", ns.sampler])
        if ns.exp_name:
            argv.extend(["--exp-name", ns.exp_name])
        if ns.use_study_db:
            argv.append("--use-study-db")
        if ns.study_db:
            argv.extend(["--study-db", ns.study_db])
        if ns.constrain_a_min is not None:
            argv.extend(["--constrain-a-min", str(ns.constrain_a_min)])
        if ns.constrain_a_max is not None:
            argv.extend(["--constrain-a-max", str(ns.constrain_a_max)])
        if ns.constrain_b_min is not None:
            argv.extend(["--constrain-b-min", str(ns.constrain_b_min)])
        if ns.constrain_b_max is not None:
            argv.extend(["--constrain-b-max", str(ns.constrain_b_max)])
        if ns.constrain_c_min is not None:
            argv.extend(["--constrain-c-min", str(ns.constrain_c_min)])
        if ns.constrain_c_max is not None:
            argv.extend(["--constrain-c-max", str(ns.constrain_c_max)])
        if ns.constrain_min_dist_min is not None:
            argv.extend(["--constrain-min-dist-min", str(ns.constrain_min_dist_min)])
        if ns.constrain_min_dist_max is not None:
            argv.extend(["--constrain-min-dist-max", str(ns.constrain_min_dist_max)])
        return run(argv)

    if ns.cmd == "final-selection":
        from dataselector.workflows.final_selection import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "thesis-pipeline":
        from dataselector.workflows.thesis_pipeline import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "adaptive-pipeline":
        from dataselector.workflows.adaptive_pipeline import main as run

        # Convert namespace to argv
        argv = []
        if ns.experiment_name != "adaptive_pipeline":
            argv.extend(["--experiment-name", ns.experiment_name])
        if ns.csv_path:
            argv.extend(["--csv-path", ns.csv_path])
        if ns.n_lhs is not None:
            argv.extend(["--n-lhs", str(ns.n_lhs)])
        if ns.n_trials != 100:
            argv.extend(["--n-trials", str(ns.n_trials)])
        if ns.n_boot != 500:
            argv.extend(["--n-boot", str(ns.n_boot)])
        if ns.n_candidates is not None:
            argv.extend(["--n-candidates", str(ns.n_candidates)])
        if ns.n_dimensions != 9:
            argv.extend(["--n-dimensions", str(ns.n_dimensions)])
        if ns.sampler != "lhs":
            argv.extend(["--sampler", ns.sampler])
        if ns.optuna_sampler != "TPESampler":
            argv.extend(["--optuna-sampler", ns.optuna_sampler])
        if ns.seed != 42:
            argv.extend(["--seed", str(ns.seed)])
        if ns.n_initial_strategy != "conservative":
            argv.extend(["--n-initial-strategy", ns.n_initial_strategy])
        if ns.n_samples is not None:
            argv.extend(["--n-samples", str(ns.n_samples)])
        if ns.n_samples_min is not None:
            argv.extend(["--n-samples-min", str(ns.n_samples_min)])
        if ns.n_samples_max is not None:
            argv.extend(["--n-samples-max", str(ns.n_samples_max)])
        if ns.fine_max_runs is not None:
            argv.extend(["--fine-max-runs", str(ns.fine_max_runs)])
        if ns.skip_exploration:
            argv.append("--skip-exploration")
        if ns.skip_fine:
            argv.append("--skip-fine")
        if ns.skip_optuna:
            argv.append("--skip-optuna")
        if ns.skip_bootstrap_injection:
            argv.append("--skip-bootstrap-injection")
        if ns.continue_on_analysis_failure:
            argv.append("--continue-on-analysis-failure")
        if ns.dry_run:
            argv.append("--dry-run")
        if ns.pre_names:
            argv.append("--pre-names")
            argv.extend(ns.pre_names)
        if ns.pre_indices:
            argv.append("--pre-indices")
            argv.extend(str(i) for i in ns.pre_indices)
        if ns.hamburg:
            argv.append("--hamburg")
        if ns.KDR146:
            argv.append("--KDR146")
        return run(argv)

    if ns.cmd == "thesis-sampler-suite":
        from dataselector.workflows.thesis_sampler_suite import main as run

        # Convert namespace to argv
        argv = []
        if ns.seeds != [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
            argv.append("--seeds")
            argv.extend(str(s) for s in ns.seeds)
        if ns.n_trials != 1000:
            argv.extend(["--n-trials", str(ns.n_trials)])
        if ns.datasets != ["hamburg", "kdr100"]:
            argv.append("--datasets")
            argv.extend(ns.datasets)
        if ns.samplers != ["qmc", "tpe", "cmaes"]:
            argv.append("--samplers")
            argv.extend(ns.samplers)
        if ns.sequential:
            argv.append("--sequential")
        if ns.n_trials_full != 2000:
            argv.extend(["--n-trials-full", str(ns.n_trials_full)])
        if ns.n_candidates is not None:
            argv.extend(["--n-candidates", str(ns.n_candidates)])
        if not ns.autoscale:
            argv.append("--no-autoscale")
        return run(argv)

    if ns.cmd == "compare-samplers":
        from dataselector.workflows.compare_samplers import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "benchmark-sampling":
        from dataselector.workflows.benchmark_sampling import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "generate-reports":
        from dataselector.workflows.generate_reports import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "report":
        from dataselector.workflows.generate_reports import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "build-tiles":
        from dataselector.data.build_tiles import build_tiles

        return build_tiles(
            image_dir=ns.image_dir, out=ns.out, force_source=ns.force_source
        )

    if ns.cmd == "bootstrap":
        from dataselector.workflows.bootstrap import main as run

        args = list(ns.args)
        if args and args[0] == "--":
            args = args[1:]
        return run(args)

    if ns.cmd == "tools":
        from dataselector.tools import archive, audit, check, clean, docs_link

        if ns.tool_cmd == "check-protected":
            return check.check_protected(
                list_only=ns.list, all_files=ns.all, extra_protected=ns.protect
            )

        if ns.tool_cmd == "check-env":
            return check.check_env_usage(paths=ns.paths if ns.paths else None)

        if ns.tool_cmd == "check-geo":
            return check.check_geo()

        if ns.tool_cmd == "verify-archive":
            return archive.verify_archive(fail_on_reference=ns.fail_on_reference)

        if ns.tool_cmd == "archive-outputs":
            from pathlib import Path

            archive_path = archive.archive_outputs(
                outputs_dir=Path(ns.outputs), dest_dir=Path(ns.dest), exclude=ns.exclude
            )
            return 0

        if ns.tool_cmd == "list-archives":
            from pathlib import Path

            archives = archive.list_archives(Path(ns.dir))
            if not archives:
                print(f"No archives found in {ns.dir}")
            else:
                print(f"Archives in {ns.dir}:")
                for arch in archives:
                    print(f"  - {arch.name}")
            return 0

        if ns.tool_cmd == "align-audit":
            return audit.align_audit(
                csv_path=ns.csv,
                base_dir=ns.base_dir,
                aux_dir=ns.aux_dir,
                target_crs=ns.target_crs,
                max_offset_m=ns.max_offset_m,
                out_json=ns.out,
                out_plot=ns.plot,
            )

        if ns.tool_cmd == "clean-workspace":
            return clean.clean_workspace(
                delete_outputs=ns.delete_outputs,
                delete_cache=ns.delete_cache,
                delete_venvs=ns.delete_venvs,
                archive_path=ns.archive,
                dry_run=ns.dry_run,
            )

        if ns.tool_cmd == "docs-link-check":
            return docs_link.check_links()

        if ns.tool_cmd == "docs-link-autofix":
            return docs_link.autofix_links(dry_run=ns.dry_run, backup=ns.backup)

        raise SystemExit(f"Unknown tools command: {ns.tool_cmd}")

    raise SystemExit(f"Unknown command: {ns.cmd}")

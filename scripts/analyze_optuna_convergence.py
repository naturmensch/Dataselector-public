#!/usr/bin/env python3
<<<<<<< HEAD
"""Analyze Optuna Study Convergence.

This script analyzes the convergence of Optuna optimization studies
by plotting objective values over trials and computing convergence metrics.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import optuna


def analyze_optuna_convergence(
    study_db: str,
    output_dir: Optional[str] = None
):
    """Analyze Optuna study convergence.

    Args:
        study_db: Path to Optuna study database
        output_dir: Output directory for plots
    """
    if output_dir is None:
        output_dir = Path(study_db).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load study
    storage = optuna.storages.RDBStorage(f"sqlite:///{study_db}")
    study = optuna.load_study(study_name="dataselector", storage=storage)

    print(f"Loaded study with {len(study.trials)} trials")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Get trial data
    trials = study.trials
    if not trials:
        print("No trials found")
        return

    # Extract data
    trial_numbers = []
    values = []
    for trial in trials:
        if trial.value is not None:
            trial_numbers.append(trial.number)
            values.append(trial.value)

    if not values:
        print("No completed trials with values")
        return

    # Sort by trial number
    sorted_indices = np.argsort(trial_numbers)
    trial_numbers = np.array(trial_numbers)[sorted_indices]
    values = np.array(values)[sorted_indices]

    # Compute rolling statistics
    window_size = max(5, len(values) // 20)  # Adaptive window
    rolling_mean = pd.Series(values).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(values).rolling(window=window_size, center=True).std()

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot all values
    ax1.plot(trial_numbers, values, 'b.', alpha=0.6, label='Trial values')
    ax1.plot(trial_numbers, rolling_mean, 'r-', linewidth=2, label=f'Rolling mean (window={window_size})')
    ax1.fill_between(trial_numbers,
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.3, color='red', label='±1 STD')
    ax1.axhline(y=study.best_value, color='green', linestyle='--', linewidth=2,
               label=f'Best value: {study.best_value:.4f}')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optuna Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot improvement over time
    best_so_far = np.minimum.accumulate(values)
    ax2.plot(trial_numbers, best_so_far, 'g-', linewidth=2, label='Best so far')
    ax2.plot(trial_numbers, values, 'b.', alpha=0.6, label='Trial values')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Objective Value')
    ax2.set_title('Best Value Progression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'optuna_convergence.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot to {plot_path}")

    # Save convergence data
    conv_df = pd.DataFrame({
        'trial_number': trial_numbers,
        'value': values,
        'rolling_mean': rolling_mean.values,
        'rolling_std': rolling_std.values,
        'best_so_far': best_so_far
    })
    csv_path = output_dir / 'optuna_convergence_data.csv'
    conv_df.to_csv(csv_path, index=False)
    print(f"Saved convergence data to {csv_path}")

    # Compute convergence metrics
    final_window = min(50, len(values) // 4)
    if final_window > 0:
        final_values = values[-final_window:]
        final_std = np.std(final_values)
        final_mean = np.mean(final_values)
        improvement_rate = (values[0] - study.best_value) / len(values)

        print("\nConvergence Metrics:")
        print(f"Final window size: {final_window}")
        print(f"Final mean: {final_mean:.4f}")
        print(f"Final std: {final_std:.4f}")
        print(f"Improvement rate: {improvement_rate:.6f} per trial")


def main():
    parser = argparse.ArgumentParser(description='Analyze Optuna Study Convergence')
    parser.add_argument('--study-db', required=True,
                       help='Path to Optuna study database (SQLite file)')

    args = parser.parse_args()

    analyze_optuna_convergence(args.study_db)


if __name__ == '__main__':
    main()
=======
"""Analyse Optuna Konvergenz um optimales n_trials zu bestimmen.

Usage:
    python scripts/analyze_optuna_convergence.py outputs/optuna_comparison/
    python scripts/analyze_optuna_convergence.py --study-db optuna_study.db
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib placeholder assigned by _ensure_matplotlib()
plt = None


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    globals()["plt"] = plt


def plot_convergence_from_csv(csv_path: Path, output_dir: Path):
    """Analysiere Konvergenz aus CSV-Datei."""
    # Ensure matplotlib backend and `plt` is available
    _ensure_matplotlib()
    df = pd.read_csv(csv_path)

    if "value" not in df.columns:
        print(f"Warnung: {csv_path} hat keine 'value' Spalte")
        return None

    # Filter out NaN values
    df = df[df["value"].notna()]

    if len(df) == 0:
        print(f"Warnung: {csv_path} hat keine validen Values")
        return None

    # Sortiere nach trial number falls vorhanden
    if "number" in df.columns:
        df = df.sort_values("number")
    elif "trial_number" in df.columns:
        df = df.sort_values("trial_number")

    best_values = df["value"].values
    cumulative_best = np.maximum.accumulate(best_values)

    # Berechne Konvergenzpunkt (99% vom Maximum)
    threshold = cumulative_best[-1] * 0.99
    conv_indices = np.where(cumulative_best >= threshold)[0]
    conv_idx = conv_indices[0] if len(conv_indices) > 0 else len(cumulative_best) - 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_best, linewidth=2, label="Best Value", color="#2E86AB")
    ax.axhline(
        y=threshold,
        color="#A23B72",
        linestyle="--",
        linewidth=1.5,
        label=f"99% von Maximum ({threshold:.3f})",
    )
    ax.axvline(
        x=conv_idx,
        color="#F18F01",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Konvergenz bei Trial {conv_idx}",
    )

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(
        f"Optuna Konvergenzanalyse: {csv_path.stem}", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = output_dir / f"{csv_path.stem}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print(f"Analyse: {csv_path.name}")
    print(f"{'='*60}")
    print(f"Gesamt Trials:        {len(best_values)}")
    print(f"Finales Best Value:   {cumulative_best[-1]:.4f}")
    print(f"99% Threshold:        {threshold:.4f}")
    print(f"Konvergenz bei Trial: {conv_idx} ({conv_idx/len(best_values)*100:.1f}%)")

    if conv_idx < len(best_values) * 0.5:
        print(
            "✅ EMPFEHLUNG: Konvergenz früh erreicht! n_trials könnte reduziert werden."
        )
    elif conv_idx < len(best_values) * 0.8:
        print(f"✅ EMPFEHLUNG: Aktuelle n_trials={len(best_values)} ist angemessen.")
    else:
        print("⚠️  WARNUNG: Konvergenz spät erreicht! Mehr Trials empfohlen.")

    print(f"Plot gespeichert: {output_path}")

    return {
        "csv_path": str(csv_path),
        "n_trials": len(best_values),
        "best_value": cumulative_best[-1],
        "convergence_trial": conv_idx,
        "convergence_ratio": conv_idx / len(best_values),
    }


def plot_convergence_from_study(study_path: Path, output_dir: Path):
    """Analysiere Konvergenz aus Optuna Study (pickle)."""
    # Ensure matplotlib backend and `plt` is available
    _ensure_matplotlib()
    # optuna is not required for reading a pickled study; remove import to satisfy linters
    try:
        import optuna  # noqa: F401
    except ImportError:
        print(
            "Fehler: optuna nicht installiert. Bitte installieren: pip install optuna"
        )
        return None

    with open(study_path, "rb") as f:
        study = pickle.load(f)

    trials = study.trials
    best_values = [t.value for t in trials if t.value is not None]

    if len(best_values) == 0:
        print(f"Warnung: Keine validen Trials in {study_path}")
        return None

    cumulative_best = np.maximum.accumulate(best_values)
    threshold = cumulative_best[-1] * 0.99
    conv_indices = np.where(cumulative_best >= threshold)[0]
    conv_idx = conv_indices[0] if len(conv_indices) > 0 else len(cumulative_best) - 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_best, linewidth=2, label="Best Value", color="#2E86AB")
    ax.axhline(
        y=threshold,
        color="#A23B72",
        linestyle="--",
        linewidth=1.5,
        label=f"99% von Maximum ({threshold:.3f})",
    )
    ax.axvline(
        x=conv_idx,
        color="#F18F01",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Konvergenz bei Trial {conv_idx}",
    )

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Best Objective Value", fontsize=12)
    ax.set_title(
        f"Optuna Konvergenzanalyse: {study.study_name}", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = output_dir / f"{study.study_name}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print(f"Analyse: {study.study_name}")
    print(f"{'='*60}")
    print(f"Gesamt Trials:        {len(best_values)}")
    print(f"Finales Best Value:   {cumulative_best[-1]:.4f}")
    print(f"99% Threshold:        {threshold:.4f}")
    print(f"Konvergenz bei Trial: {conv_idx} ({conv_idx/len(best_values)*100:.1f}%)")

    if conv_idx < len(best_values) * 0.5:
        print(
            "✅ EMPFEHLUNG: Konvergenz früh erreicht! n_trials könnte reduziert werden."
        )
    elif conv_idx < len(best_values) * 0.8:
        print(f"✅ EMPFEHLUNG: Aktuelle n_trials={len(best_values)} ist angemessen.")
    else:
        print("⚠️  WARNUNG: Konvergenz spät erreicht! Mehr Trials empfohlen.")

    print(f"Plot gespeichert: {output_path}")

    return {
        "study_name": study.study_name,
        "n_trials": len(best_values),
        "best_value": cumulative_best[-1],
        "convergence_trial": conv_idx,
        "convergence_ratio": conv_idx / len(best_values),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analysiere Optuna Konvergenz für n_trials Optimierung"
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Pfad zu Optuna Study (pickle) oder Verzeichnis mit CSVs",
    )
    parser.add_argument("--study-db", type=str, help="Optuna Study Database (sqlite)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output Verzeichnis für Plots (default: outputs/)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    results = []

    if args.study_db:
        # Load from database
        try:
            # Attempt to plot convergence from the provided DB path. The direct `load_study`
            # call is unnecessary here and was removed to avoid an unused-variable lint.
            result = plot_convergence_from_study(Path(args.study_db), args.output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Fehler beim Laden der Study DB: {e}")

    elif args.input:
        input_path = Path(args.input)

        if input_path.is_file():
            if input_path.suffix == ".pkl":
                result = plot_convergence_from_study(input_path, args.output_dir)
                if result:
                    results.append(result)
            elif input_path.suffix == ".csv":
                result = plot_convergence_from_csv(input_path, args.output_dir)
                if result:
                    results.append(result)

        elif input_path.is_dir():
            # Suche alle CSV-Dateien im Verzeichnis
            csv_files = sorted(input_path.glob("*.csv"))
            pkl_files = sorted(input_path.glob("*.pkl"))

            for csv_file in csv_files:
                result = plot_convergence_from_csv(csv_file, args.output_dir)
                if result:
                    results.append(result)

            for pkl_file in pkl_files:
                result = plot_convergence_from_study(pkl_file, args.output_dir)
                if result:
                    results.append(result)

    else:
        # Default: Suche in outputs/...
        print("Suche Optuna-Ergebnisse in outputs/...")

        # NEW: Search in versioned run directories first
        runs_dir = Path("outputs/runs")
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.glob("*")):
                if not run_dir.is_dir():
                    continue
                trials_csv = run_dir / "results" / "trials.csv"
                if trials_csv.exists():
                    result = plot_convergence_from_csv(trials_csv, args.output_dir)
                    if result:
                        results.append(result)

        # Fallback: legacy search directories
        search_dirs = [
            Path("outputs/optuna_comparison"),
            Path("outputs/optuna_comparison_v2"),
            Path("outputs/optuna_comparison_v3"),
            Path("outputs"),
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            csv_files = sorted(search_dir.glob("*trials*.csv"))
            # Also accept the canonical optuna_results.csv filename produced by the optimizer
            opt_csv = search_dir / "optuna_results.csv"
            if opt_csv.exists():
                csv_files = [opt_csv] + csv_files

            for csv_file in csv_files:
                result = plot_convergence_from_csv(csv_file, args.output_dir)
                if result:
                    results.append(result)

    # Zusammenfassung
    if results:
        print(f"\n{'='*60}")
        print("GESAMTZUSAMMENFASSUNG")
        print(f"{'='*60}")

        df_summary = pd.DataFrame(results)
        print(df_summary.to_string(index=False))

        summary_path = args.output_dir / "optuna_convergence_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\nZusammenfassung gespeichert: {summary_path}")

        # Gesamtempfehlung
        avg_conv_ratio = df_summary["convergence_ratio"].mean()
        print(
            f"\n📊 Durchschnittliche Konvergenz bei {avg_conv_ratio*100:.1f}% der Trials"
        )

        if avg_conv_ratio < 0.5:
            print("✅ EMPFEHLUNG: n_trials könnte auf ~100-150 reduziert werden")
        elif avg_conv_ratio < 0.8:
            print("✅ EMPFEHLUNG: Aktuelle n_trials=200 ist optimal")
        else:
            print("⚠️  EMPFEHLUNG: n_trials auf 300-500 erhöhen für bessere Konvergenz")

    else:
        print("\n❌ Keine Optuna-Ergebnisse gefunden.")
        print("Bitte Pfad angeben oder Optuna-Experimente durchführen.")


if __name__ == "__main__":
    main()
>>>>>>> ci/add-smoke-tests

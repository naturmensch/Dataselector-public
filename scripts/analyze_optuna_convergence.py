#!/usr/bin/env python3
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
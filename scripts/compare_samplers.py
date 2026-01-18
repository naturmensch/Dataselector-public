#!/usr/bin/env python3
"""Scientific sampler comparison: QMC vs TPE vs CMA-ES.

Runs identical optimization problems with different samplers to empirically
validate sampler choice for thesis.

Usage:
    python scripts/compare_samplers.py --n-trials 500 --n-candidates 800 --hamburg
    
Generates:
    outputs/runs/sampler_comparison_<timestamp>/
        ├── qmc_500trials/
        ├── tpe_500trials/
        ├── cmaes_500trials/
        └── comparison_summary.csv
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def run_sampler_comparison(
    samplers=['qmc', 'tpe', 'cmaes'],
    n_trials=500,
    n_candidates=673,
    seed=42,
    preselection_flag='--hamburg',
    exp_description="Sampler comparison study"
):
    """Run Optuna optimization with different samplers and compare results."""
    import subprocess
    
    timestamp = datetime.now().strftime("%Y%m%d_T%H%M%S")
    base_run = Path('outputs') / 'runs' / f'sampler_comparison_{timestamp}'
    base_run.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for sampler in samplers:
        print(f"\n{'='*70}")
        print(f"Running optimization with {sampler.upper()} sampler")
        print(f"{'='*70}\n")
        
        exp_name = f"{sampler}_{n_trials}trials"
        
        cmd = [
            sys.executable,
            'scripts/optuna_optimize.py',
            '--n-trials', str(n_trials),
            '--n-candidates', str(n_candidates),
            '--n-samples-min', '30',
            '--n-samples-max', '50',
            '--sampler', sampler,
            '--seed', str(seed),
            '--exp-name', exp_name,
            '--exp-desc', f"{exp_description} ({sampler})",
        ]
        
        if preselection_flag:
            cmd.append(preselection_flag)
        
        # Run optimization
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: {sampler} run failed!")
            print(result.stderr)
            continue
        
        print(result.stdout)
        
        # Load results
        run_dirs = sorted(Path('outputs/runs').glob(f'*{exp_name}'))
        if not run_dirs:
            print(f"WARNING: Could not find run directory for {sampler}")
            continue
        
        run_dir = run_dirs[-1]
        trials_csv = run_dir / 'results' / 'trials.csv'
        
        if trials_csv.exists():
            df = pd.read_csv(trials_csv)
            df = df[df['value'].notna()]
            
            if len(df) > 0:
                best_value = df['value'].max()
                best_trial_num = df.loc[df['value'].idxmax(), 'trial_number']
                mean_value = df['value'].mean()
                std_value = df['value'].std()
                
                # Convergence: trial where 99% of best is reached
                cumulative_best = df['value'].expanding().max()
                threshold = best_value * 0.99
                conv_idx = (cumulative_best >= threshold).idxmax() if (cumulative_best >= threshold).any() else len(df) - 1
                
                results.append({
                    'sampler': sampler,
                    'n_trials': len(df),
                    'best_value': best_value,
                    'best_trial_number': best_trial_num,
                    'mean_value': mean_value,
                    'std_value': std_value,
                    'convergence_trial': conv_idx,
                    'convergence_ratio': conv_idx / len(df),
                    'run_dir': str(run_dir),
                })
                
                print(f"\n✓ {sampler.upper()} Results:")
                print(f"  Best value: {best_value:.4f} (trial {best_trial_num})")
                print(f"  Mean ± std: {mean_value:.4f} ± {std_value:.4f}")
                print(f"  Convergence: trial {conv_idx} ({conv_idx/len(df)*100:.1f}%)")
    
    if not results:
        print("\nERROR: No results collected!")
        return 1
    
    # Save comparison summary
    df_results = pd.DataFrame(results)
    summary_file = base_run / 'comparison_summary.csv'
    df_results.to_csv(summary_file, index=False)
    print(f"\n✓ Saved comparison summary: {summary_file}")
    
    # Create comparison plots
    create_comparison_plots(df_results, results, base_run)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("SAMPLER COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(df_results[['sampler', 'best_value', 'convergence_trial', 'convergence_ratio']].to_string(index=False))
    print(f"{'='*70}\n")
    
    # Recommendation
    best_sampler = df_results.loc[df_results['best_value'].idxmax(), 'sampler']
    fastest_conv = df_results.loc[df_results['convergence_ratio'].idxmin(), 'sampler']
    
    print(f"📊 Scientific Assessment:")
    print(f"  Best objective value: {best_sampler.upper()}")
    print(f"  Fastest convergence: {fastest_conv.upper()}")
    
    if best_sampler == fastest_conv:
        print(f"  ✅ RECOMMENDATION: Use {best_sampler.upper()} (best performance + fastest convergence)")
    else:
        print(f"  ⚖️  TRADE-OFF: {best_sampler.upper()} for quality, {fastest_conv.upper()} for speed")
    
    print(f"\n{'='*70}\n")
    
    return 0


def create_comparison_plots(df_summary, results, output_dir):
    """Create visualization plots for sampler comparison."""
    # Load trial histories
    histories = {}
    for res in results:
        run_dir = Path(res['run_dir'])
        trials_csv = run_dir / 'results' / 'trials.csv'
        if trials_csv.exists():
            df = pd.read_csv(trials_csv)
            df = df[df['value'].notna()]
            histories[res['sampler']] = df
    
    if not histories:
        return
    
    # Plot 1: Convergence curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for sampler, df in histories.items():
        cumulative_best = df['value'].expanding().max()
        ax.plot(cumulative_best.values, label=sampler.upper(), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Best Objective Value', fontsize=12)
    ax.set_title('Sampler Comparison: Convergence Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    conv_plot = output_dir / 'convergence_comparison.png'
    plt.savefig(conv_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved convergence plot: {conv_plot}")
    
    # Plot 2: Box plot of objective values
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data_for_box = []
    labels_for_box = []
    for sampler, df in histories.items():
        data_for_box.append(df['value'].values)
        labels_for_box.append(sampler.upper())
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Sampler Comparison: Value Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    box_plot = output_dir / 'distribution_comparison.png'
    plt.savefig(box_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plot: {box_plot}")
    
    # Plot 3: Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Best values
    ax1.bar(df_summary['sampler'], df_summary['best_value'], color='#2ecc71', alpha=0.7)
    ax1.set_ylabel('Best Objective Value', fontsize=11)
    ax1.set_title('Best Performance', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Convergence ratios
    ax2.bar(df_summary['sampler'], df_summary['convergence_ratio'], color='#e74c3c', alpha=0.7)
    ax2.set_ylabel('Convergence Ratio (lower = faster)', fontsize=11)
    ax2.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    summary_plot = output_dir / 'summary_comparison.png'
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved summary plot: {summary_plot}")


def main():
    parser = argparse.ArgumentParser(
        description='Scientific sampler comparison for thesis'
    )
    parser.add_argument('--samplers', nargs='+', default=['qmc', 'tpe', 'cmaes'],
                        help='Samplers to compare')
    parser.add_argument('--n-trials', type=int, default=500,
                        help='Number of trials per sampler')
    parser.add_argument('--n-candidates', type=int, default=673,
                        help='Candidate pool size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--hamburg', action='store_true',
                        help='Use Hamburg preselection')
    parser.add_argument('--KDR146', action='store_true',
                        help='Use KDR_146 preselection')
    
    args = parser.parse_args()
    
    preselection_flag = None
    if args.hamburg:
        preselection_flag = '--hamburg'
    elif args.KDR146:
        preselection_flag = '--KDR146'
    
    return run_sampler_comparison(
        samplers=args.samplers,
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        seed=args.seed,
        preselection_flag=preselection_flag,
        exp_description=f"Sampler comparison (n_trials={args.n_trials})"
    )


if __name__ == '__main__':
    sys.exit(main())

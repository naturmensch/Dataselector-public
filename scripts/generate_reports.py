"""Generate summary plots and reports for Optuna and experiments.
Saves outputs to outputs/ with date suffix.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

def save_fig(fig, name):
    date = datetime.now().strftime('%Y%m%d')
    out = OUT / f"{name}_{date}.png"
    fig.savefig(out, bbox_inches='tight')
    print(f"Saved: {out}")


def make_optuna_plots():
    p = OUT / 'optuna_results.csv'
    if not p.exists():
        print('No optuna_results.csv found; skipping optuna plots')
        return {}

    df = pd.read_csv(p)
    stats = {}
    if 'value' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(df['number'], df['value'], marker='o')
        ax.set_xlabel('trial')
        ax.set_ylabel('value')
        ax.set_title('Optuna: objective per trial')
        save_fig(fig, 'optuna_history')
        stats['optuna_trials'] = len(df)
        stats['optuna_best_value'] = float(df['value'].max())
        # min_distance histogram
        if 'params_min_distance_km' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df['params_min_distance_km'], bins=10, kde=False, ax=ax)
            ax.set_title('Optuna: min_distance distribution')
            save_fig(fig, 'optuna_min_distance_hist')
    return stats


def make_experiments_plots():
    p = OUT / 'experiments_20_runs_100_samples.csv'
    if not p.exists():
        print('No experiments summary found; skipping experiments plots')
        return {}

    df = pd.read_csv(p)
    stats = {}
    if 'n_selected' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df['n_selected'], bins=10, kde=False, ax=ax)
        ax.set_title('Experiments: n_selected distribution')
        save_fig(fig, 'experiments_n_selected_hist')
        stats['experiments_runs'] = len(df)
        stats['experiments_mean_n_selected'] = float(df['n_selected'].mean())

    if 'min_distance_km' in df.columns and 'diversity' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='min_distance_km', y='diversity', data=df, ax=ax)
        ax.set_title('Experiments: diversity vs min_distance')
        save_fig(fig, 'experiments_diversity_vs_min_distance')

    return stats


def write_report(stats_optuna, stats_exp):
    date = datetime.now().strftime('%Y%m%d')
    out = OUT / f'report_{date}.md'
    lines = ['# Selection Reports', '', f'Date: {datetime.now().isoformat()}', '']
    lines.append('## Optuna')
    if stats_optuna:
        lines.append(f"- trials: {stats_optuna.get('optuna_trials')}")
        lines.append(f"- best_value: {stats_optuna.get('optuna_best_value')}")
    else:
        lines.append('- no optuna results')

    lines.append('')
    lines.append('## Experiments')
    if stats_exp:
        lines.append(f"- runs: {stats_exp.get('experiments_runs')}")
        lines.append(f"- mean_n_selected: {stats_exp.get('experiments_mean_n_selected')}")
    else:
        lines.append('- no experiments results')

    out.write_text('\n'.join(lines))
    print(f"Report written to {out}")


if __name__ == '__main__':
    s1 = make_optuna_plots()
    s2 = make_experiments_plots()
    write_report(s1, s2)

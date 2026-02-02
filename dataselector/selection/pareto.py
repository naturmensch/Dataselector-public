"""
Pareto-Optimierung für Multi-Criteria Selection.

Wissenschaftlich fundierte Methode zur Identifikation optimaler Gewichtungen
ohne ad-hoc Scoring-Funktionen.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def is_dominated(
    metrics_a: dict, metrics_b: dict, objectives: List[Tuple[str, str]]
) -> bool:
    """
    Prüft ob Lösung A von Lösung B dominiert wird.

    Args:
        metrics_a: Metriken von Lösung A
        metrics_b: Metriken von Lösung B
        objectives: Liste von (metric_name, direction) Tupeln
                   direction: 'maximize' oder 'minimize'

    Returns:
        True wenn B A dominiert (B ist in allen Objectives >= A und in mind. einem >)
    """
    better_in_any = False
    worse_in_any = False

    for metric, direction in objectives:
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]

        if direction == "maximize":
            if val_b > val_a:
                better_in_any = True
            elif val_b < val_a:
                worse_in_any = True
        else:  # minimize
            if val_b < val_a:
                better_in_any = True
            elif val_b > val_a:
                worse_in_any = True

    # B dominiert A wenn B in mindestens einem Objective besser und in keinem schlechter
    return better_in_any and not worse_in_any


def compute_pareto_front(
    results: pd.DataFrame, objectives: List[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Berechnet Pareto-Front aus Ergebnissen.

    Args:
        results: DataFrame mit Metriken und Parametern
        objectives: Liste von (metric, direction) für Multi-Objective
                   Default: clusters_covered (max), temporal_std (min), spatial_mean_km (max)

    Returns:
        DataFrame nur mit Pareto-optimalen Lösungen
    """
    if objectives is None:
        objectives = [
            ("clusters_covered", "maximize"),
            ("temporal_std", "maximize"),  # Höhere STD = bessere zeitliche Diversität
            (
                "spatial_mean_km",
                "maximize",
            ),  # Höhere Mean Distance = bessere räumliche Verteilung
        ]

    pareto_indices = []

    for i, row_i in results.iterrows():
        metrics_i = row_i.to_dict()
        is_dominated_flag = False

        for j, row_j in results.iterrows():
            if i == j:
                continue
            metrics_j = row_j.to_dict()

            if is_dominated(metrics_i, metrics_j, objectives):
                is_dominated_flag = True
                break

        if not is_dominated_flag:
            pareto_indices.append(i)

    return results.loc[pareto_indices].copy()


def visualize_pareto_front(
    results: pd.DataFrame,
    pareto_front: pd.DataFrame,
    output_dir: str = "outputs/pareto",
):
    # Use a non-interactive backend and import pyplot at runtime to avoid
    # module-level imports that break lint rules (E402).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    """
    Visualisiert Pareto-Front in 2D-Projektionen.

    Args:
        results: Alle Ergebnisse
        pareto_front: Pareto-optimale Ergebnisse
        output_dir: Output-Verzeichnis für Plots
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2D Projection: temporal_std vs spatial_mean_km
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Temporal vs Spatial
    ax = axes[0]
    ax.scatter(
        results["temporal_std"],
        results["spatial_mean_km"],
        alpha=0.5,
        label="All solutions",
        s=50,
    )
    ax.scatter(
        pareto_front["temporal_std"],
        pareto_front["spatial_mean_km"],
        color="red",
        marker="*",
        s=200,
        label="Pareto-optimal",
        edgecolors="black",
    )

    for _, row in pareto_front.iterrows():
        ax.annotate(
            f"α={row['alpha']:.1f}\nβ={row['beta']:.2f}\nγ={row['gamma']:.2f}",
            (row["temporal_std"], row["spatial_mean_km"]),
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Temporal STD (Jahre)", fontsize=12)
    ax.set_ylabel("Spatial Mean Distance (km)", fontsize=12)
    ax.set_title("Pareto-Front: Temporal vs Spatial Diversity", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: n_selected vs temporal_std (colored by cluster_coverage)
    ax = axes[1]
    scatter = ax.scatter(
        results["n_selected"],
        results["temporal_std"],
        c=results["clusters_covered"],
        cmap="viridis",
        alpha=0.5,
        s=50,
    )
    ax.scatter(
        pareto_front["n_selected"],
        pareto_front["temporal_std"],
        color="red",
        marker="*",
        s=200,
        edgecolors="black",
        label="Pareto-optimal",
    )

    plt.colorbar(scatter, ax=ax, label="Clusters Covered")
    ax.set_xlabel("N Selected Samples", fontsize=12)
    ax.set_ylabel("Temporal STD (Jahre)", fontsize=12)
    ax.set_title("Selection Size vs Temporal Diversity", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "pareto_front.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Pareto-Front visualisiert: {out_path / 'pareto_front.png'}")


def export_pareto_report(
    pareto_front: pd.DataFrame, output_path: str = "outputs/pareto/pareto_solutions.csv"
):
    """
    Exportiert Pareto-optimale Lösungen als CSV mit Empfehlungen.

    Args:
        pareto_front: DataFrame mit Pareto-optimalen Lösungen
        output_path: Pfad für CSV-Export
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sortiere nach temporal_std (höhere STD = bessere zeitliche Diversität)
    pareto_sorted = pareto_front.sort_values("temporal_std", ascending=False)

    pareto_sorted.to_csv(out, index=False)
    print(f"Pareto-Lösungen exportiert: {out}")

    print("\n" + "=" * 70)
    print("PARETO-OPTIMALE LÖSUNGEN (sortiert nach zeitlicher Diversität)")
    print("=" * 70)
    for i, row in pareto_sorted.iterrows():
        print(f"\nLösung {i+1}:")
        print(
            f"  Gewichtung: α={row['alpha']:.2f}, β={row['beta']:.2f}, γ={row['gamma']:.2f}"
        )
        print(f"  Cluster Coverage: {row['clusters_covered']}/8")
        print(f"  Temporal STD: {row['temporal_std']:.2f} Jahre")
        print(f"  Temporal Range: {row['temporal_range']} Jahre")
        print(f"  Spatial Mean Dist: {row['spatial_mean_km']:.1f} km")
        print(f"  N Selected: {row['n_selected']}")

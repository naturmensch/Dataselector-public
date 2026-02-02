"""
Visualisierungen für Clustering und Selection Ergebnisse.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class Visualizer:
    """Erstellt Visualisierungen der Pipeline-Ergebnisse."""

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialisiert den Visualizer.

        Args:
            output_dir: Verzeichnis für Output-Dateien
        """
        # Ensure a non-interactive backend and import pyplot at runtime so we
        # don't have module-level imports that trigger E402 in linters.
        global plt
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def plot_umap_clusters(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        selected_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualisiert UMAP-Embeddings mit Cluster-Farben.

        Args:
            embeddings_2d: 2D UMAP-Embeddings
            cluster_labels: Cluster-Zuordnungen
            selected_indices: Optional, Indizes der ausgewählten Samples
            save_path: Optional, Pfad zum Speichern

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Diskrete Colormap basierend auf der tatsächlichen Anzahl an Clustern
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        from matplotlib.colors import BoundaryNorm, ListedColormap

        try:
            palette = sns.color_palette("tab10", n_colors=n_clusters)
        except Exception:
            palette = sns.color_palette(n_colors=n_clusters)
        cmap = ListedColormap(palette)
        norm = BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)

        # Scatter plot aller Punkte (farbig nach Cluster)
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=cluster_labels,
            cmap=cmap,
            norm=norm,
            alpha=0.8,
            s=40,
            linewidths=0,
            zorder=1,
        )

        # Markiere ausgewählte Samples (Sterne in Clusterfarbe)
        if selected_indices is not None and len(selected_indices) > 0:
            sel_clusters = cluster_labels[selected_indices]
            ax.scatter(
                embeddings_2d[selected_indices, 0],
                embeddings_2d[selected_indices, 1],
                c=sel_clusters,
                cmap=cmap,
                norm=norm,
                marker="*",
                s=200,
                edgecolors="black",
                linewidths=1.5,
                label="Ausgewählte Samples",
                zorder=5,
            )

        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.set_title(
            "UMAP Clustering der KDR100 Kacheln", fontsize=14, fontweight="bold"
        )

        # Colorbar für Cluster (diskret, nur vorhandene Cluster anzeigen)
        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_clusters)
        cbar.set_label("Cluster ID", fontsize=12)
        cbar.set_ticks(unique_clusters)
        cbar.set_ticklabels(unique_clusters)

        if selected_indices is not None:
            ax.legend(fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot gespeichert: {save_path}")

        return fig

    def plot_temporal_distribution(
        self,
        metadata: pd.DataFrame,
        selected_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualisiert die zeitliche Verteilung der Kacheln.

        Args:
            metadata: DataFrame mit 'year' Spalte
            selected_indices: Optional, Indizes der Auswahl
            save_path: Optional, Pfad zum Speichern

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Histogramm aller Jahre
        all_years = metadata["year"].dropna()
        ax.hist(
            all_years,
            bins=30,
            alpha=0.5,
            label="Alle Kacheln",
            color="gray",
            edgecolor="black",
        )

        # Histogramm der ausgewählten Jahre
        if selected_indices is not None:
            selected_years = metadata.loc[selected_indices, "year"].dropna()
            ax.hist(
                selected_years,
                bins=30,
                alpha=0.7,
                label="Ausgewählte Kacheln",
                color="red",
                edgecolor="black",
            )

        ax.set_xlabel("Jahr", fontsize=12)
        ax.set_ylabel("Anzahl Kacheln", fontsize=12)
        ax.set_title(
            "Zeitliche Verteilung der KDR100 Kacheln", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot gespeichert: {save_path}")

        return fig

    def plot_spatial_distribution(
        self,
        metadata: pd.DataFrame,
        selected_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualisiert die räumliche Verteilung der Kacheln.

        Unterstützt Geo-aware Darstellung: falls `metadata` ein Objekt mit
        `gdf_metric` (projizierte GeoDataFrame mit `_proj_x/_proj_y`) enthält,
        werden diese metrischen Koordinaten für eine akkurate Darstellung
        in Metern verwendet und `ax.set_aspect('equal')` gesetzt.

        Args:
            metadata: DataFrame oder Objekt mit GeoDataFrame-Attribute
                (z.B. `gdf_metric` oder `geometry`)
            selected_indices: Optional, Indizes der Auswahl
            save_path: Optional, Pfad zum Speichern

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Geo-aware: prefer projected metric coords if available
        plotted = False
        try:
            from dataselector.data.io import get_metric_gdf

            gdf_metric = get_metric_gdf(metadata)
        except Exception:
            gdf_metric = getattr(metadata, "gdf_metric", None)

        if gdf_metric is not None:
            try:
                xs = gdf_metric["_proj_x"].values
                ys = gdf_metric["_proj_y"].values
                ax.scatter(xs, ys, alpha=0.4, s=50, label="Alle Kacheln", color="gray")
                if selected_indices is not None and len(selected_indices) > 0:
                    ax.scatter(
                        xs[selected_indices],
                        ys[selected_indices],
                        alpha=0.8,
                        s=200,
                        label="Ausgewählte Kacheln",
                        color="red",
                        marker="*",
                        edgecolors="black",
                        linewidths=1.5,
                    )
                ax.set_xlabel("X (meters)", fontsize=12)
                ax.set_ylabel("Y (meters)", fontsize=12)
                ax.set_aspect("equal", adjustable="box")
                plotted = True
            except Exception:
                plotted = False

        # If metadata looks like a GeoDataFrame with geometry
        if not plotted and "geometry" in getattr(metadata, "columns", []):
            try:
                geom = metadata["geometry"]
                xs = geom.x
                ys = geom.y
                ax.scatter(xs, ys, alpha=0.4, s=50, label="Alle Kacheln", color="gray")
                if selected_indices is not None and len(selected_indices) > 0:
                    ax.scatter(
                        xs.iloc[selected_indices],
                        ys.iloc[selected_indices],
                        alpha=0.8,
                        s=200,
                        label="Ausgewählte Kacheln",
                        color="red",
                        marker="*",
                        edgecolors="black",
                        linewidths=1.5,
                    )
                ax.set_xlabel("X", fontsize=12)
                ax.set_ylabel("Y", fontsize=12)
                ax.set_aspect("equal", adjustable="box")
                plotted = True
            except Exception:
                plotted = False

        # Fallback: plain pandas lat/lon scatter
        if not plotted:
            ax.scatter(
                metadata["left"],
                metadata["N"],
                alpha=0.4,
                s=50,
                label="Alle Kacheln",
                color="gray",
            )

            # Ausgewählte Kacheln
            if selected_indices is not None and len(selected_indices) > 0:
                ax.scatter(
                    metadata.loc[selected_indices, "left"],
                    metadata.loc[selected_indices, "N"],
                    alpha=0.8,
                    s=200,
                    label="Ausgewählte Kacheln",
                    color="red",
                    marker="*",
                    edgecolors="black",
                    linewidths=1.5,
                )

            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)

        ax.set_title(
            "Räumliche Verteilung der KDR100 Kacheln", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot gespeichert: {save_path}")

        return fig

    def plot_cluster_distribution(
        self,
        cluster_labels: np.ndarray,
        selected_indices: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Visualisiert die Verteilung über Cluster.

        Args:
            cluster_labels: Cluster-Zuordnungen aller Samples
            selected_indices: Optional, Indizes der Auswahl
            save_path: Optional, Pfad zum Speichern

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        unique_clusters = np.unique(cluster_labels)

        # Zähle Samples pro Cluster
        all_counts = [np.sum(cluster_labels == c) for c in unique_clusters]

        x = np.arange(len(unique_clusters))
        width = 0.35

        ax.bar(x - width / 2, all_counts, width, label="Alle Kacheln", alpha=0.7)

        if selected_indices is not None:
            selected_labels = cluster_labels[selected_indices]
            selected_counts = [np.sum(selected_labels == c) for c in unique_clusters]
            ax.bar(
                x + width / 2,
                selected_counts,
                width,
                label="Ausgewählte Kacheln",
                alpha=0.7,
            )

        ax.set_xlabel("Cluster ID", fontsize=12)
        ax.set_ylabel("Anzahl Kacheln", fontsize=12)
        ax.set_title("Verteilung über Cluster", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(unique_clusters)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot gespeichert: {save_path}")

        return fig

    def create_summary_report(
        self,
        embeddings_2d: np.ndarray,
        cluster_labels: np.ndarray,
        metadata: pd.DataFrame,
        selected_indices: np.ndarray,
        output_prefix: str = "selection_report",
    ):
        """
        Erstellt einen vollständigen Visualisierungs-Report.

        Args:
            embeddings_2d: UMAP-Embeddings
            cluster_labels: Cluster-Labels
            metadata: Metadaten-DataFrame
            selected_indices: Ausgewählte Indizes
            output_prefix: Prefix für Output-Dateien
        """
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(exist_ok=True, parents=True)

        # 1. UMAP Cluster Plot
        self.plot_umap_clusters(
            embeddings_2d,
            cluster_labels,
            selected_indices,
            save_path=output_dir / "umap_clusters.png",
        )
        plt.close()

        # 2. Temporal Distribution
        self.plot_temporal_distribution(
            metadata,
            selected_indices,
            save_path=output_dir / "temporal_distribution.png",
        )
        plt.close()

        # 3. Spatial Distribution
        self.plot_spatial_distribution(
            metadata,
            selected_indices,
            save_path=output_dir / "spatial_distribution.png",
        )
        plt.close()

        # 4. Cluster Distribution
        self.plot_cluster_distribution(
            cluster_labels,
            selected_indices,
            save_path=output_dir / "cluster_distribution.png",
        )
        plt.close()

        print(f"\nVisualisierungs-Report erstellt in: {output_dir}")

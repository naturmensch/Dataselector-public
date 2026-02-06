"""
Main Pipeline für die KDR100 Datenselektion.

Dieses Skript orchestriert den gesamten Workflow:
1. Metadaten-Verarbeitung
2. Feature Extraction
3. Clustering
4. Diversity Selection
5. Visualisierung
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from dataselector.analysis.visualizer import Visualizer
from dataselector.data.metadata_processor import MetadataProcessor
from dataselector.features.feature_extractor import FeatureExtractor
from dataselector.selection.clustering import ClusteringPipeline
from dataselector.selection.diversity_selector import DiversitySelector


class KDR100SelectionPipeline:
    """Hauptpipeline für die algorithmische Datenselektion."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialisiert die Pipeline.

        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)

        # Komponenten
        self.metadata_processor = None
        self.feature_extractor = None
        self.clustering_pipeline = None
        self.diversity_selector = None
        self.visualizer = None

        # Daten
        self.metadata_df: Optional[pd.DataFrame] = None
        self.features: Optional[np.ndarray] = None
        self.embeddings_2d: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.selected_indices: Optional[np.ndarray] = None

    def _load_config(self, config_path: str) -> dict:
        """Lädt die Konfiguration aus YAML."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        else:
            # Default-Konfiguration
            return {
                "data": {
                    "metadata_path": "data/KDR100_foliage_with_files_epsg3857.csv",
                    "csv_path": "data/KDR100_foliage_with_files_epsg3857.csv",
                    "image_dir": "data/images",
                },
                "feature_extraction": {
                    "model": "resnet50",
                    "batch_size": 8,
                    "crop_size": [2048, 2048],
                },
                "clustering": {"n_clusters": 8, "umap_components": 2},
                "selection": {
                    "n_samples": 34,
                    "temporal_weight": 0.2,
                    "min_distance_km": 50.0,
                },
                "output": {"dir": "outputs", "prefix": "kdr100_selection"},
            }

    def run(self):
        """Führt die komplette Pipeline aus."""
        print("=" * 80)
        print("KDR100 Datenselektion Pipeline")
        print("=" * 80)

        # Schritt 1: Metadaten verarbeiten
        print("\n[1/5] Verarbeite Metadaten...")
        self._process_metadata()

        # Schritt 2: Features extrahieren
        print("\n[2/5] Extrahiere Deep Learning Features...")
        self._extract_features()

        # Schritt 3: Clustering
        print("\n[3/5] Führe Clustering durch...")
        self._perform_clustering()

        # Schritt 4: Diversity Selection
        print("\n[4/5] Wähle diverse Samples aus...")
        self._select_diverse_samples()

        # Schritt 5: Visualisierung
        print("\n[5/5] Erstelle Visualisierungen...")
        self._create_visualizations()

        # Zusammenfassung
        self._print_summary()

        print("\n" + "=" * 80)
        print("Pipeline erfolgreich abgeschlossen!")
        print("=" * 80)

    def _process_metadata(self):
        """Verarbeitet die Metadaten (CSV oder DBF)."""
        metadata_path = self.config["data"].get(
            "metadata_path", self.config["data"].get("csv_path")
        )
        if metadata_path is None:
            raise ValueError(
                "Kein Metadatenpfad in der Konfiguration gefunden "
                "('metadata_path' oder 'csv_path')."
            )

        self.metadata_processor = MetadataProcessor(metadata_path)
        self.metadata_df = self.metadata_processor.load_csv()
        self.metadata_df = self.metadata_processor.add_temporal_metadata()

        # Activate GIS logic: Transform to UTM (Metric CRS)
        # This enables precise distance calculations and correct aspect ratios in plots
        if self.config.get("features", {}).get("geo", True):
            print("  [GIS] Aktiviere metrisches CRS (UTM Zone 32N)...")
            self.metadata_df.gdf_metric = self.metadata_processor.ensure_metric_crs()

        stats = self.metadata_processor.get_summary_statistics()
        print(f"  Geladene Kacheln: {stats['total_tiles']}")
        start_year, end_year = stats["temporal_range"]
        print(f"  Zeitspanne: {start_year} - {end_year}")
        lat_min = stats["spatial_extent"]["lat_min"]
        lat_max = stats["spatial_extent"]["lat_max"]
        print(f"  Räumliche Ausdehnung: Lat {lat_min:.2f} - {lat_max:.2f}")

    def _extract_features(self):
        """Extrahiert visuelle Features aus den Bildern."""
        model = self.config["feature_extraction"]["model"]
        batch_size = self.config["feature_extraction"]["batch_size"]
        image_dir = Path(self.config["data"]["image_dir"])

        self.feature_extractor = FeatureExtractor(model_name=model)

        # Resolve image paths (bevorzugt shortName, fallback longName)
        self.metadata_df = self.metadata_processor.resolve_image_paths(
            image_dir, prefer_shortname=True
        )

        # Extrahiere Features für alle Bilder (verwende image_filename)

        # Prüfe auf gecachte Features (schneller Re-Run)
        output_dir = Path(self.config["output"]["dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        metadata_path = output_dir / "metadata.csv"

        from dataselector.data.io import load_or_extract_features

        csv_meta = metadata_path if metadata_path.exists() else None
        self.features = load_or_extract_features(
            out_dir=output_dir,
            csv_meta=str(csv_meta) if csv_meta is not None else None,
            batch_size=batch_size,
            cache=True,
        )
        # ensure metadata CSV is written if missing
        if not metadata_path.exists():
            self.metadata_df.to_csv(metadata_path, index=False)

        print(f"  Feature-Dimension: {self.features.shape}")

    def _perform_clustering(self):
        """Führt UMAP und K-Means Clustering durch."""
        n_clusters = self.config["clustering"]["n_clusters"]
        umap_components = self.config["clustering"]["umap_components"]

        self.clustering_pipeline = ClusteringPipeline(
            n_clusters=n_clusters, umap_n_components=umap_components
        )

        (
            self.embeddings_2d,
            self.cluster_labels,
        ) = self.clustering_pipeline.fit_transform(self.features)

        stats = self.clustering_pipeline.get_cluster_statistics()
        print(f"  Cluster-Größen: {stats['cluster_sizes']}")

    def _select_diverse_samples(self):
        """Wählt diverse Samples mittels Facility Location."""
        n_samples = self.config["selection"]["n_samples"]
        temporal_weight = self.config["selection"]["temporal_weight"]

        self.diversity_selector = DiversitySelector(n_samples=n_samples)

        self.selected_indices = self.diversity_selector.select(
            self.features,
            metadata=self.metadata_df,
            temporal_weight=temporal_weight,
            spatial_constraint=True,
        )

        # Exportiere Auswahl
        output_dir = Path(self.config["output"]["dir"])
        output_prefix = self.config["output"]["prefix"]
        output_path = output_dir / f"{output_prefix}_selected.csv"

        self.diversity_selector.export_selection(self.metadata_df, str(output_path))

        # Coverage-Statistiken
        coverage = self.diversity_selector.get_coverage_statistics(
            self.features, self.cluster_labels
        )
        print(f"  Ausgewählte Samples: {coverage['n_selected']}")
        n_clusters_total = self.config["clustering"]["n_clusters"]
        print(
            f"  Abgedeckte Cluster: {coverage['clusters_covered']}/{n_clusters_total}"
        )
        print(f"  Diversitäts-Score: {coverage['diversity_score']:.4f}")

    def _create_visualizations(self):
        """Erstellt alle Visualisierungen."""
        output_dir = self.config["output"]["dir"]
        output_prefix = self.config["output"]["prefix"]

        self.visualizer = Visualizer(output_dir=output_dir)

        self.visualizer.create_summary_report(
            self.embeddings_2d,
            self.cluster_labels,
            self.metadata_df,
            self.selected_indices,
            output_prefix=output_prefix,
        )

    def _print_summary(self):
        """Gibt eine Zusammenfassung der Ergebnisse aus."""
        print("\n" + "=" * 80)
        print("ZUSAMMENFASSUNG")
        print("=" * 80)

        selected_df = self.metadata_df.iloc[self.selected_indices]

        print(f"\nAusgewählte Kacheln: {len(self.selected_indices)}")
        min_year = selected_df["year"].min()
        max_year = selected_df["year"].max()
        print(f"Zeitliche Abdeckung: {min_year:.0f} - {max_year:.0f}")

        # Top 5 ausgewählte Kacheln
        print("\nTop 5 ausgewählte Kacheln:")
        for i, idx in enumerate(self.selected_indices[:5]):
            row = self.metadata_df.iloc[idx]
            print(f"  {i+1}. {row['longName']} (Jahr: {row['year']:.0f})")


def main():
    """Haupteinstiegspunkt."""
    pipeline = KDR100SelectionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

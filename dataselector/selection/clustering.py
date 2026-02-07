"""
Clustering und Dimensionsreduktion mit UMAP und K-Means.

Dieses Modul reduziert die Feature-Dimensionalität und gruppiert
ähnliche Kacheln in Cluster.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import umap

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ClusteringPipeline:
    """Pipeline für Dimensionsreduktion und Clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        umap_n_components: int = 2,
        random_state: int = 42,
        umap_random_state: int | None = 42,
        umap_n_jobs: int = 1,
        umap_n_neighbors: int | None = None,
    ):
        """
        Initialisiert die Clustering Pipeline.

        Args:
            n_clusters: Anzahl Cluster für K-Means
            umap_n_components: Ziel-Dimensionalität für UMAP
            random_state: Seed für KMeans-Reproduzierbarkeit
            umap_random_state: Seed für UMAP (set to None to allow parallelism)
            umap_n_jobs: n_jobs for UMAP when non-deterministic (-1 uses all CPUs)
        """
        self.n_clusters = n_clusters
        self.umap_n_components = umap_n_components
        self.random_state = random_state
        self.umap_random_state = umap_random_state
        self.umap_n_jobs = umap_n_jobs

        self.scaler = StandardScaler()
        self.umap_reducer = None
        self.kmeans = None

        # user-configurable override for n_neighbors; if None, use adaptive heuristic
        self.umap_n_neighbors = umap_n_neighbors

        self.embeddings_2d: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None

    def fit_transform(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline aus: Skalierung, UMAP, K-Means.

        Args:
            features: 2D array (n_samples, n_features)

        Returns:
            Tupel (2D embeddings, cluster labels)
        """
        # 1. Standardisierung
        features_scaled = self.scaler.fit_transform(features)

        # 2. Dimensionsreduktion (UMAP primär, PCA fallback)
        self.embeddings_2d = None
        if UMAP_AVAILABLE:
            try:
                n_neighbors = self.umap_n_neighbors
                if n_neighbors is None:
                    n_neighbors = max(2, min(15, features_scaled.shape[0] - 1))
                n_jobs = self.umap_n_jobs if self.umap_random_state is None else 1
                self.umap_reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    random_state=self.umap_random_state,
                    n_jobs=n_jobs,
                    n_neighbors=n_neighbors,
                )
                emb = self.umap_reducer.fit_transform(features_scaled)
                if np.ndim(emb) != 2 or not np.isfinite(emb).all():
                    raise ValueError("UMAP produced invalid embeddings")
                self.embeddings_2d = emb
            except Exception as exc:
                print(
                    f"UMAP failed ({exc}); falling back to PCA for deterministic embeddings."
                )

        if self.embeddings_2d is None:
            # PCA fallback is deterministic and keeps tests stable across envs.
            n_components = min(
                self.umap_n_components, features_scaled.shape[0], features_scaled.shape[1]
            )
            n_components = max(1, int(n_components))
            self.umap_reducer = PCA(
                n_components=n_components, random_state=self.random_state
            )
            emb = self.umap_reducer.fit_transform(features_scaled)
            if emb.shape[1] < self.umap_n_components:
                pad = self.umap_n_components - emb.shape[1]
                emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant")
            self.embeddings_2d = emb

        # 3. K-Means Clustering
        print(f"Führe K-Means Clustering durch ({self.n_clusters} Cluster)...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",  # Moderne API: automatische Wahl basierend auf Datengröße
        )
        self.cluster_labels = self.kmeans.fit_predict(self.embeddings_2d)

        return self.embeddings_2d, self.cluster_labels

    def get_cluster_statistics(self) -> dict:
        """
        Berechnet Statistiken über die Cluster.

        Returns:
            Dictionary mit Cluster-Statistiken
        """
        if self.cluster_labels is None:
            raise ValueError("Pipeline muss zuerst ausgeführt werden (fit_transform)")

        unique, counts = np.unique(self.cluster_labels, return_counts=True)

        stats = {
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
            "total_samples": len(self.cluster_labels),
            "n_clusters": self.n_clusters,
        }

        return stats

    def get_cluster_centers(self) -> np.ndarray:
        """
        Gibt die Cluster-Zentren zurück.

        Returns:
            Array der Cluster-Zentren
        """
        if self.kmeans is None:
            raise ValueError("K-Means muss zuerst ausgeführt werden")

        return self.kmeans.cluster_centers_

    def predict_cluster(self, features: np.ndarray) -> np.ndarray:
        """
        Weist neue Samples bestehenden Clustern zu.

        Args:
            features: Neue Feature-Vektoren

        Returns:
            Cluster-Labels
        """
        if self.kmeans is None or self.umap_reducer is None:
            raise ValueError("Modell muss zuerst trainiert werden")

        # Skaliere und transformiere
        features_scaled = self.scaler.transform(features)
        embeddings = self.umap_reducer.transform(features_scaled)
        expected_dim = int(getattr(self.kmeans, "n_features_in_", embeddings.shape[1]))
        if embeddings.shape[1] < expected_dim:
            pad = expected_dim - embeddings.shape[1]
            embeddings = np.pad(embeddings, ((0, 0), (0, pad)), mode="constant")
        elif embeddings.shape[1] > expected_dim:
            embeddings = embeddings[:, :expected_dim]
        labels = self.kmeans.predict(embeddings)

        return labels

    def get_samples_per_cluster(self, cluster_id: int) -> np.ndarray:
        """
        Gibt die Indizes aller Samples in einem bestimmten Cluster zurück.

        Args:
            cluster_id: ID des Clusters

        Returns:
            Array von Sample-Indizes
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering muss zuerst ausgeführt werden")

        return np.where(self.cluster_labels == cluster_id)[0]

    def calculate_intra_cluster_distances(self) -> dict:
        """
        Berechnet die durchschnittliche Distanz innerhalb jedes Clusters.

        Returns:
            Dictionary {cluster_id: avg_distance}
        """
        if self.embeddings_2d is None or self.cluster_labels is None:
            raise ValueError("Pipeline muss zuerst ausgeführt werden")

        distances = {}
        centers = self.get_cluster_centers()

        for i in range(self.n_clusters):
            cluster_samples = self.embeddings_2d[self.cluster_labels == i]
            if len(cluster_samples) > 0:
                # Euklidische Distanz zum Zentrum
                dists = np.linalg.norm(cluster_samples - centers[i], axis=1)
                distances[i] = float(np.mean(dists))
            else:
                distances[i] = 0.0

        return distances

"""
Multi-Criteria Facility Location mit Custom Distance Metric.

Wissenschaftlich fundierte Lösung: Kombiniere visuelle, räumliche und
temporale Distanzen in einer unified metric statt Feature-Augmentation.

Theoretische Basis:
- Multi-criteria optimization via weighted distance aggregation
- Explicit trade-off control zwischen Visual/Spatial/Temporal
- Keine Dominanz einzelner Dimensionalitäten
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from dataselector.data.spatial_schema import (
    coordinates_look_projected,
    normalize_spatial_schema,
)
from dataselector.data.io import get_metric_gdf
from dataselector.selection.spatial_facility_location import (
    haversine_distance,
    haversine_matrix,
)


class MultiCriteriaFacilityLocation:
    """
    Facility Location mit gewichteter Multi-Criteria Distance.

    Kombiniert drei Distanztypen:
    1. Visual: Feature-basierte Ähnlichkeit (ResNet50)
    2. Spatial: Geografische Distanz (Haversine)
    3. Temporal: Zeitliche Distanz (Jahre)

    Unified Distance:
        d(i,j) = α·d_visual(i,j) + β·d_spatial(i,j) + γ·d_temporal(i,j)

    Mit Constraints:
        - α + β + γ = 1 (normalisiert)
        - Alle Distanzen normalisiert auf [0,1]
    """

    def __init__(
        self,
        n_samples: int,
        metadata: pd.DataFrame,
        alpha_visual: float = 0.7,
        beta_spatial: float = 0.15,
        gamma_temporal: float = 0.15,
        min_distance_km: float = 50.0,
        metric: str = "euclidean",
        random_state: Optional[int] = None,
        preselected_indices: Optional[list] = None,
    ):
        """
        Initialisiert Multi-Criteria Selector.

        Args:
            n_samples: Anzahl zu selektierender Samples
            metadata: DataFrame mit ['ul_x', 'ul_y', 'lr_x', 'lr_y', 'year']
            alpha_visual: Gewicht für visuelle Ähnlichkeit [0,1]
            beta_spatial: Gewicht für räumliche Diversität [0,1]
            gamma_temporal: Gewicht für zeitliche Diversität [0,1]
            min_distance_km: Hard spatial constraint (km)
            metric: 'euclidean' oder 'cosine' für visuelle Features
            random_state: Seed

        Note:
            alpha + beta + gamma muss ~1.0 sein für normalisierte Gewichte.
            Falls nicht, wird ein `ValueError` ausgelöst — der Nutzer muss
            explizit normierte Gewichte angeben.
        """
        self.n_samples = n_samples
        self.metadata = normalize_spatial_schema(metadata, require_bounds=True, copy=True)
        self.min_distance_km = min_distance_km
        self.metric = metric
        self.random_state = random_state

        # Validiere Gewichte (keine automatische Normalisierung mehr)
        total_weight = alpha_visual + beta_spatial + gamma_temporal
        # Toleranz für Floating-Point Abweichungen
        if not (0.99999 < total_weight < 1.00001):
            raise ValueError(
                f"Gewichte müssen sich zu 1.0 addieren (aktuell: {total_weight:.6f}). "
                f"Eingegeben: α={alpha_visual:.4f}, β={beta_spatial:.4f}, γ={gamma_temporal:.4f}\n"
                f"Beispiel: alpha=0.7, beta=0.15, gamma=0.15"
            )

        # Speichere Gewichte (werden angenommen als normiert)
        self.alpha = alpha_visual
        self.beta = beta_spatial
        self.gamma = gamma_temporal

        self.ranking = None
        self.n_samples_ = 0

        # Pre-selected seeds (positional indices into candidate arrays)
        self.preselected_indices = (
            list(preselected_indices) if preselected_indices is not None else []
        )

        # Cache coordinates
        self.y_coords = self.metadata["center_y"].values
        self.x_coords = self.metadata["center_x"].values
        self._coords_are_projected = coordinates_look_projected(self.metadata)
        self.years = self.metadata["year"].fillna(self.metadata["year"].median()).values

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Berechnet gewichtete Multi-Criteria Distanzmatrix.

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Pairwise distance matrix (n_samples, n_samples)
        """
        n = X.shape[0]

        # 1. Visual distances (normalisiert auf [0,1])
        if self.metric == "euclidean":
            d_visual = euclidean_distances(X)
            d_visual = d_visual / d_visual.max() if d_visual.max() > 0 else d_visual
        elif self.metric == "cosine":
            sim_visual = cosine_similarity(X)
            d_visual = 1 - sim_visual  # Cosine distance
            d_visual = np.clip(d_visual, 0, 1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # 2. Spatial distances (compute vectorized km matrix and normalize)
        # Use vectorized haversine_matrix to avoid python-level loops
        d_spatial_km = None

        # Prefer projected metric coordinates if available via metadata.gdf_metric
        gdf_metric = get_metric_gdf(self.metadata)
        if gdf_metric is not None:
            try:
                xs = gdf_metric["_proj_x"].values.astype(float)
                ys = gdf_metric["_proj_y"].values.astype(float)
                coords = np.stack([xs, ys], axis=1)
                # use sklearn's euclidean_distances (imported at module scope)
                from sklearn.metrics import pairwise as _pairwise

                d_m = _pairwise.euclidean_distances(coords)
                d_spatial_km = d_m / 1000.0
            except Exception:
                d_spatial_km = None

        if d_spatial_km is None:
            try:
                if self._coords_are_projected:
                    coords = np.stack([self.x_coords, self.y_coords], axis=1)
                    d_spatial_km = euclidean_distances(coords) / 1000.0
                else:
                    d_spatial_km = haversine_matrix(self.y_coords, self.x_coords)
            except Exception:
                # Fallback (should rarely happen) — compute with loops
                d_spatial_km = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        if self._coords_are_projected:
                            dx = self.x_coords[i] - self.x_coords[j]
                            dy = self.y_coords[i] - self.y_coords[j]
                            dist_km = float((dx * dx + dy * dy) ** 0.5 / 1000.0)
                        else:
                            dist_km = haversine_distance(
                                self.y_coords[i],
                                self.x_coords[i],
                                self.y_coords[j],
                                self.x_coords[j],
                            )
                        d_spatial_km[i, j] = dist_km
                        d_spatial_km[j, i] = dist_km

        # Keep raw km distances for constraint checks
        self._spatial_km = d_spatial_km

        d_spatial = d_spatial_km.copy()
        max_spatial = d_spatial.max()
        if max_spatial > 0:
            d_spatial = d_spatial / max_spatial

        # 3. Temporal distances (normalisiert auf [0,1])
        year_diff = np.abs(self.years[:, None] - self.years[None, :])
        max_temporal = year_diff.max()
        if max_temporal > 0:
            d_temporal = year_diff / max_temporal
        else:
            d_temporal = year_diff

        # 4. Combine weighted
        d_combined = (
            self.alpha * d_visual + self.beta * d_spatial + self.gamma * d_temporal
        )

        return d_combined

    def _violates_spatial_constraint(
        self, candidate_idx: int, selected_indices: np.ndarray
    ) -> bool:
        """Prüft hard spatial constraint (min_distance_km)."""
        if self.min_distance_km is None:
            return False
        if len(selected_indices) == 0:
            return False

        cand_y = self.y_coords[candidate_idx]
        cand_x = self.x_coords[candidate_idx]

        # Use precomputed spatial km matrix if available for O(1) checks
        if hasattr(self, "_spatial_km") and self._spatial_km is not None:
            # vectorized selection: check if any selected distance is below threshold
            dists = self._spatial_km[candidate_idx, selected_indices]
            return np.any(dists < self.min_distance_km)

        # Fallback to scalar haversine if no precomputed matrix
        for sel_idx in selected_indices:
            sel_y = self.y_coords[sel_idx]
            sel_x = self.x_coords[sel_idx]

            if self._coords_are_projected:
                dx = cand_x - sel_x
                dy = cand_y - sel_y
                distance = float((dx * dx + dy * dy) ** 0.5 / 1000.0)
            else:
                distance = haversine_distance(cand_y, cand_x, sel_y, sel_x)

            if distance < self.min_distance_km:
                return True

        return False

    def _greedy_selection(self, distances: np.ndarray) -> np.ndarray:
        """
        Greedy Facility Location auf Multi-Criteria Distanzen.

        Args:
            distances: Pairwise distance matrix

        Returns:
            Selected indices
        """
        n = distances.shape[0]
        selected = []
        remaining = set(range(n))

        # Convert distances to similarities (Facility Location nutzt similarities)
        max_dist = distances.max()
        similarities = max_dist - distances  # Höhere sim für kleinere dist

        # Initialize with pre-selected seeds if any
        if len(self.preselected_indices) > 0:
            # Validate and keep only indices in-range
            sel = [int(i) for i in self.preselected_indices if 0 <= int(i) < n]
            selected = list(dict.fromkeys(sel))  # preserve order, remove duplicates
            remaining = set(range(n)) - set(selected)
        else:
            selected = []
            remaining = set(range(n))

        # If more seeds than requested, trim
        if len(selected) >= self.n_samples:
            return np.array(selected[: self.n_samples])

        num_to_select = self.n_samples - len(selected)

        for _ in range(num_to_select):
            if len(remaining) == 0:
                break

            best_gain = -np.inf
            best_idx = None

            for candidate in remaining:
                # Check spatial constraint relative to ALL currently selected (including seeds)
                if self._violates_spatial_constraint(candidate, np.array(selected)):
                    continue

                # Facility Location gain
                if len(selected) == 0:
                    gain = similarities[candidate].sum()
                else:
                    current_max_sims = similarities[selected].max(axis=0)
                    new_max_sims = np.maximum(current_max_sims, similarities[candidate])
                    gain = (new_max_sims - current_max_sims).sum()

                if gain > best_gain:
                    best_gain = gain
                    best_idx = candidate

            if best_idx is None:
                print(
                    f"  [WARNING] Spatial constraint zu restriktiv - nur {len(selected)} von {self.n_samples} selektiert"
                )
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.array(selected[: self.n_samples])

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "MultiCriteriaFacilityLocation":
        """
        Führt Multi-Criteria Selection aus.

        Args:
            X: Feature-Matrix (visual features only, no augmentation needed)
            y: Ignoriert

        Returns:
            self
        """
        print(
            f"  [Multi-Criteria] Weights validated: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}"
        )

        # Guard: features must align with metadata length
        if X.shape[0] != len(self.y_coords):
            # Versuche, Hash-Infos für bessere Fehlermeldung zu laden
            meta_hash = None
            meta_path = None
            try:
                import inspect

                from dataselector.pipeline.cache import (
                    compute_meta_hash,
                    features_path_for_hash,
                )

                # Versuche, den Pfad zur Metadaten-CSV zu erraten
                frame = inspect.currentframe()
                csv_meta = None
                while frame:
                    local_vars = frame.f_locals
                    if "csv_meta" in local_vars:
                        csv_meta = local_vars["csv_meta"]
                        break
                    frame = frame.f_back
                if csv_meta:
                    meta_hash = compute_meta_hash(csv_meta, params=None)
                    meta_path = features_path_for_hash("outputs", meta_hash)
            except Exception:
                pass
            msg = (
                f"Feature rows ({X.shape[0]}) != metadata rows ({len(self.y_coords)}). "
                "This will cause broadcasting errors (visual vs spatial matrices). "
                "Bitte stelle sicher, dass Feature-Cache und Metadaten exakt zusammengehören.\n"
            )
            if meta_hash:
                msg += f"[Debug] metadata_hash: {meta_hash}\n"
            if meta_path:
                msg += f"[Debug] expected features-cache: {meta_path}\n"
            msg += (
                "Regeneriere den Feature-Cache passend zu den Metadaten, z.B.:\n"
                '  rm outputs/features-*.npy && python -c "from dataselector.data.io import load_or_extract_features; load_or_extract_features(cache=True)"'
            )
            raise ValueError(msg)

        distances = self._compute_pairwise_distances(X)

        if self.min_distance_km == 0.0:
            md_note = f"{self.min_distance_km}km (disabled)"
        else:
            md_note = f"{self.min_distance_km}km"
        print(f"  [Multi-Criteria] Greedy selection with min_dist={md_note}...")
        self.ranking = self._greedy_selection(distances)
        self.n_samples_ = len(self.ranking)

        return self

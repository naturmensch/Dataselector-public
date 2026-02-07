"""
Spatial-Constraint-Integrated Facility Location Selection.

Implementiert eine wissenschaftlich fundierte Lösung zur Integration
räumlicher Constraints direkt in die submodulare Optimierung statt
als Post-Processing-Filter.

Theoretische Basis:
- Submodular maximization under matroid constraints (Nemhauser et al., 1978)
- Spatial constraints als feasibility check während Greedy-Selektion
- Garantiert monotone submodulare Funktion unter harten Constraints
"""

from math import atan2, cos, radians, sin, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from apricot import FacilityLocationSelection

from dataselector.data.io import get_metric_gdf
from dataselector.data.spatial_schema import (
    coordinates_look_projected,
    normalize_spatial_schema,
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Berechnet Haversine-Distanz zwischen zwei Geopunkten.

    Args:
        lat1, lon1: Koordinaten Punkt 1 (Grad)
        lat2, lon2: Koordinaten Punkt 2 (Grad)

    Returns:
        Distanz in Kilometern
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def haversine_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Haversine distance matrix (in km).

    Args:
        latitudes: 1D array of latitudes (degrees)
        longitudes: 1D array of longitudes (degrees)

    Returns:
        2D array shape (n, n) of pairwise distances in kilometers.
    """
    R = 6371.0
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


class SpatialConstrainedFacilityLocation(FacilityLocationSelection):
    """
    Facility Location Selection mit integrierten räumlichen Constraints.

    Im Gegensatz zu Post-Processing-Filtern wird die räumliche Constraint
    direkt in die Greedy-Optimierung integriert: Kandidaten, die die
    Mindestdistanz verletzen, werden von der Selektion ausgeschlossen.

    Algorithmus:
        Für jeden Greedy-Schritt:
        1. Berechne Gains für alle nicht-selektierten Samples
        2. Setze Gain = -∞ für Samples, die spatial constraint verletzen
        3. Wähle Sample mit maximalem validen Gain

    Dies garantiert:
        - Submodularität bleibt erhalten
        - Spatial constraint ist hart eingehalten
        - Temporal/Visual-Optimierung nicht nachträglich überschrieben
    """

    def __init__(
        self,
        n_samples: int,
        metadata: pd.DataFrame,
        min_distance_km: float = 50.0,
        metric: str = "euclidean",
        random_state: Optional[int] = None,
    ):
        """
        Initialisiert Constraint-integrierten Selector.

        Args:
            n_samples: Anzahl zu selektierender Samples
            metadata: DataFrame mit kanonischen Spatial-Spalten (ul/lr)
            min_distance_km: Minimale Distanz zwischen Samples (km)
            metric: Distanzmetrik für Feature-Space
            random_state: Seed für Reproduzierbarkeit
        """
        super().__init__(n_samples=n_samples, metric=metric, random_state=random_state)

        self.metadata = normalize_spatial_schema(
            metadata, require_bounds=True, copy=True
        )
        self.min_distance_km = min_distance_km
        self._precompute_coordinates()

    def _precompute_coordinates(self) -> None:
        """Extrahiert und cached Koordinaten für schnelle Distanzberechnung.

        If the provided metadata includes a GeoDataFrame projected in meters
        (columns `_proj_x` and `_proj_y`), we cache those projected
        coordinates for fast Euclidean distance computations.
        """
        self.y_coords = self.metadata["center_y"].values
        self.x_coords = self.metadata["center_x"].values
        self._coords_are_projected = coordinates_look_projected(self.metadata)

        # If metadata provides a cached metric GeoDataFrame, use that
        self._has_proj = False
        gdf_metric = get_metric_gdf(self.metadata)
        if gdf_metric is not None:
            try:
                self._proj_x = gdf_metric["_proj_x"].values
                self._proj_y = gdf_metric["_proj_y"].values
                self._has_proj = True
            except Exception:
                self._has_proj = False

    def _violates_spatial_constraint(
        self, candidate_idx: int, selected_indices: np.ndarray
    ) -> bool:
        """
        Prüft ob Kandidat räumliche Constraint verletzt.

        Args:
            candidate_idx: Index des Kandidaten
            selected_indices: Bereits selektierte Indizes

        Returns:
            True wenn Kandidat zu nah an einem selektierten Sample liegt
        """
        if len(selected_indices) == 0:
            return False

        # If projected coordinates are available, compute planar distances (meters -> km)
        if getattr(self, "_has_proj", False):
            cx = self._proj_x[candidate_idx]
            cy = self._proj_y[candidate_idx]
            for sel_idx in selected_indices:
                sx = self._proj_x[sel_idx]
                sy = self._proj_y[sel_idx]
                dx = cx - sx
                dy = cy - sy
                dist_km = (dx * dx + dy * dy) ** 0.5 / 1000.0
                if dist_km < self.min_distance_km:
                    return True
            return False

        # fallback to Haversine distance
        cand_y = self.y_coords[candidate_idx]
        cand_x = self.x_coords[candidate_idx]

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

    def _greedy_selection(self, X: np.ndarray) -> np.ndarray:
        """
        Überschreibt Greedy-Selektion mit räumlicher Constraint-Integration.

        Modifiziert den Standard Greedy-Algorithmus:
        - Berechnet Gains wie üblich
        - Maskiert räumlich invalide Kandidaten
        - Selektiert nur valide Samples

        Args:
            X: Feature-Matrix (n_samples, n_features)

        Returns:
            Array der selektierten Indizes (kann < n_samples sein bei
            restriktiven Constraints)
        """
        n_samples = X.shape[0]
        selected = []
        remaining = set(range(n_samples))

        # Similarity/distance matrix berechnen (nur einmal)
        if self.metric == "euclidean":
            # Pairwise distances für Facility Location
            from sklearn.metrics.pairwise import euclidean_distances

            distances = euclidean_distances(X)
            similarities = -distances  # Facility Location nutzt similarities
        elif self.metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(X)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Greedy-Selektion mit Constraint-Check
        for _ in range(self.n_samples):
            if len(remaining) == 0:
                break

            best_gain = -np.inf
            best_idx = None

            for candidate in remaining:
                # Spatial constraint check
                if self._violates_spatial_constraint(candidate, np.array(selected)):
                    continue

                # Berechne Facility Location Gain
                if len(selected) == 0:
                    # Erstes Sample: wähle Sample mit höchster Gesamtsimilarität
                    gain = similarities[candidate].sum()
                else:
                    # Marginaler Gain: max(sim(candidate, all)) - max(sim(best_so_far, all))
                    # Vereinfachte Facility Location: Sum of max similarities
                    current_max_sims = similarities[selected].max(axis=0)
                    new_max_sims = np.maximum(current_max_sims, similarities[candidate])
                    gain = (new_max_sims - current_max_sims).sum()

                if gain > best_gain:
                    best_gain = gain
                    best_idx = candidate

            if best_idx is None:
                # Keine validen Kandidaten mehr
                print(
                    f"  [WARNING] Spatial constraint zu restriktiv - nur {len(selected)} von {self.n_samples} selektiert"
                )
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.array(selected)

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "SpatialConstrainedFacilityLocation":
        """
        Führt Constraint-integrierte Selektion aus.

        Args:
            X: Feature-Matrix (n_samples, n_features)
            y: Ignoriert (für sklearn API-Kompatibilität)

        Returns:
            self
        """
        self.ranking = self._greedy_selection(X)
        self.n_samples_ = len(self.ranking)
        return self

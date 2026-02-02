"""
Diversity Sampling mittels Facility Location Function.

Dieses Modul implementiert die Core-Set Selection zur Auswahl
der diversesten 34 Kacheln aus dem Datensatz.

Unterstützt drei Modi:
1. Legacy: Facility Location + Post-Processing spatial filter
2. Constraint-integrated: Spatial constraint direkt in Optimierung
3. Multi-Criteria: Unified distance metric (Visual+Spatial+Temporal)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
try:
    from apricot import FacilityLocationSelection
except Exception:
    FacilityLocationSelection = None
    print("[WARN] apricot FacilityLocationSelection not available; falling back to pure-Python methods where possible")

try:
    from dataselector.selection.multi_criteria_facility_location import MultiCriteriaFacilityLocation
except ImportError:
    MultiCriteriaFacilityLocation = None

try:
    from dataselector.selection.spatial_facility_location import SpatialConstrainedFacilityLocation
except ImportError:
    SpatialConstrainedFacilityLocation = None


class DiversitySelector:
    """Wählt diverse Samples mittels submodularer Optimierung."""

    def __init__(
        self,
        n_samples: Optional[int] = None,
        metric: str = "euclidean",
        random_state: int = 42,
        use_constraint_integration: bool = False,
        use_multi_criteria: bool = True,
        use_lazy_greedy: bool = False,
    ):
        """
        Initialisiert den Diversity Selector.

        Args:
            n_samples: Anzahl zu selektierender Samples
            metric: Distanzmetrik ('euclidean', 'cosine')
            random_state: Seed für Reproduzierbarkeit
            use_constraint_integration: Wenn True, nutze Constraint-integrierte
                Optimierung (wissenschaftlich korrekt, aber spatial-dominant).
            use_multi_criteria: Wenn True, nutze Multi-Criteria distance metric
                (empfohlen: explizite Visual/Spatial/Temporal Trade-offs).
                Überschreibt use_constraint_integration.
        """
        self.n_samples = n_samples
        self.metric = metric
        self.random_state = random_state
        self.use_constraint_integration = use_constraint_integration
        self.use_multi_criteria = use_multi_criteria
        self.use_lazy_greedy = use_lazy_greedy

        self.selector = None
        self.selected_indices: Optional[np.ndarray] = None

    def select(
        self,
        features: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
        temporal_weight: float = 0.2,
        spatial_constraint: bool = True,
        min_distance_km: float = 50.0,
        adaptive_min_distance: bool = False,
        adaptive_step_km: float = 5.0,
        adaptive_min_allowed_km: float = 0.0,
        alpha_visual: float = 0.7,
        beta_spatial: float = 0.15,
        gamma_temporal: float = 0.15,
        override_n_samples: Optional[int] = None,
        pre_selected: Optional[list] = None,
        pre_selected_names: Optional[list] = None,
    ) -> np.ndarray:
        """
        Führt die Diversity Selection aus.

        Args:
            features: Feature-Matrix (n_samples, n_features)
            metadata: Optional DataFrame mit year, N, left Spalten
            temporal_weight: Gewichtung für zeitliche Diversität (Legacy/Constraint mode)
            spatial_constraint: Ob räumlicher Sperrfilter angewandt wird
            min_distance_km: Minimale Distanz zwischen Samples (km)
            alpha_visual: Gewicht für visuelle Ähnlichkeit (Multi-Criteria mode)
            beta_spatial: Gewicht für räumliche Diversität (Multi-Criteria mode)
            gamma_temporal: Gewicht für zeitliche Diversität (Multi-Criteria mode)
            override_n_samples: Optionaler Überschreibungswert für die Anzahl zu selektierender Samples.
                Wenn angegeben, hat dieser Vorrang vor dem bei der Instanziierung gesetzten `n_samples`.

        Returns:
            Array von ausgewählten Indizes
        """
        """
        Führt die Diversity Selection aus.

        Args:
            features: Feature-Matrix (n_samples, n_features)
            metadata: Optional DataFrame mit year, N, left Spalten
            temporal_weight: Gewichtung für zeitliche Diversität (Legacy/Constraint mode)
            spatial_constraint: Ob räumlicher Sperrfilter angewandt wird
            min_distance_km: Minimale Distanz zwischen Samples (km)
            alpha_visual: Gewicht für visuelle Ähnlichkeit (Multi-Criteria mode)
            beta_spatial: Gewicht für räumliche Diversität (Multi-Criteria mode)
            gamma_temporal: Gewicht für zeitliche Diversität (Multi-Criteria mode)

        Returns:
            Array von ausgewählten Indizes
        """

        # Anzahl der verfügbaren Kandidaten und effektive Auswahlgröße
        n_candidates = features.shape[0]
        # Resolve effective number of samples: explicit override wins, then instance attribute
        effective_n = (
            int(override_n_samples)
            if override_n_samples is not None
            else self.n_samples
        )
        if effective_n is None:
            raise ValueError(
                "n_samples ist nicht gesetzt. Bitte übergeben Sie `n_samples` (z.B. via `select(..., override_n_samples=...)`) "
                "oder setzen Sie `n_samples` beim Konstruktor; alternativ verwenden Sie `src.pipeline_utils.compute_adaptive_n_initial`."
            )
        # Persist resolved n_samples to ensure internal helpers (spatial constraints) can access it
        self.n_samples = int(effective_n)
        n_to_select = min(self.n_samples, n_candidates)

        # Wähle Selektionsmodus
        if self.use_multi_criteria and metadata is not None and MultiCriteriaFacilityLocation is not None:
            # Multi-Criteria: Unified distance metric (EMPFOHLEN)
            print(
                f"Führe Multi-Criteria Facility Location durch ({n_to_select} Samples)..."
            )
            # Resolve pre-selected indices by name (if provided)
            preselected_indices = []
            if pre_selected is not None:
                preselected_indices = list(pre_selected)
            elif pre_selected_names is not None and metadata is not None:
                for nm in pre_selected_names:
                    # match shortName exact (if exists) or longName substring (case-insensitive)
                    mask = metadata["longName"].astype(str).str.lower().str.contains(
                        str(nm).lower()
                    )
                    if "shortName" in metadata.columns:
                        mask = mask | (
                            metadata["shortName"].astype(str).str.lower() == str(nm).lower()
                        )
                    idxs = list(mask[mask].index)
                    if len(idxs) == 0:
                        print(
                            f"  [WARN] Pre-selected name '{nm}' not found in metadata"
                        )
                    else:
                        preselected_indices.extend(idxs)

            # Logging: report resolved preselected indices and their names (if any)
            if len(preselected_indices) > 0 and metadata is not None:
                try:
                    names = [
                        str(metadata.loc[i, "shortName"])
                        + "/"
                        + str(metadata.loc[i, "longName"])
                        for i in preselected_indices
                    ]
                    print(
                        f"  [INFO] Resolved pre-selected indices: {preselected_indices} -> {names}"
                    )
                except Exception:
                    print(
                        f"  [INFO] Resolved pre-selected indices: {preselected_indices}"
                    )

            self.selector = MultiCriteriaFacilityLocation(
                n_samples=n_to_select,
                metadata=metadata,
                alpha_visual=alpha_visual,
                beta_spatial=beta_spatial,
                gamma_temporal=gamma_temporal,
                min_distance_km=min_distance_km,
                metric=self.metric,
                random_state=self.random_state,
                preselected_indices=preselected_indices,
            )
            # NO feature augmentation for multi-criteria
            self.selector.fit(features)
            self.selected_indices = self.selector.ranking

        elif (
            self.use_constraint_integration
            and spatial_constraint
            and metadata is not None
            and SpatialConstrainedFacilityLocation is not None
        ):
            # Constraint-integrierte Optimierung
            # Erweitere Features mit Metadaten
            if temporal_weight > 0:
                features = self._augment_features_with_metadata(
                    features, metadata, temporal_weight
                )

            print(
                f"Führe Constraint-integrierte Facility Location durch ({n_to_select} Samples, min_dist={min_distance_km}km)..."
            )
            self.selector = SpatialConstrainedFacilityLocation(
                n_samples=n_to_select,
                metadata=metadata,
                min_distance_km=min_distance_km,
                metric=self.metric,
                random_state=self.random_state,
            )
            self.selector.fit(features)
            self.selected_indices = self.selector.ranking

        else:
            # Legacy: Facility Location + Post-Processing-Filter
            if metadata is not None and temporal_weight > 0:
                features = self._augment_features_with_metadata(
                    features, metadata, temporal_weight
                )

            print(f"Führe Facility Location Selection durch ({n_to_select} Samples)...")
            if self.use_lazy_greedy:
                try:
                    from dataselector.selection.lazy_facility_location import LazyFacilityLocationSelection

                    self.selector = LazyFacilityLocationSelection(
                        n_samples=n_to_select,
                        metric=self.metric,
                        random_state=self.random_state,
                    )
                    self.selector.fit(features)
                    self.selected_indices = self.selector.ranking
                except Exception as e:
                    print(f"[WARN] LazyFacilityLocation unavailable or failed: {e}; falling back to simple greedy selection")

            else:
                if FacilityLocationSelection is not None:
                    self.selector = FacilityLocationSelection(
                        n_samples=n_to_select, metric=self.metric
                    )
                    # Fit the selected selector (both expose `ranking`)
                    self.selector.fit(features)
                    self.selected_indices = self.selector.ranking
                else:
                    # Fallback greedy diversity selection (farthest-point greedy on normalized features)
                    print("[WARN] FacilityLocationSelection not available; using greedy farthest-point selection")
                    # Normalize features for cosine-like behavior
                    try:
                        feats = features.copy().astype(float)
                        # L2-normalize
                        norms = np.linalg.norm(feats, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        feats = feats / norms
                        selected = []
                        # Initialize with highest variance sample
                        var_idx = np.argmax(np.var(feats, axis=1))
                        selected.append(int(var_idx))
                        while len(selected) < n_to_select:
                            # compute min similarity (dot) to current selected set and pick min
                            sim = feats @ feats[selected].T
                            min_sim = sim.min(axis=1)
                            # pick index with smallest max similarity (most dissimilar)
                            cand = int(np.nanargmin(min_sim))
                            if cand in selected:
                                # fallback: random selection
                                remaining = [i for i in range(feats.shape[0]) if i not in selected]
                                if not remaining:
                                    break
                                cand = remaining[0]
                            selected.append(cand)
                        self.selected_indices = np.array(selected, dtype=int)
                    except Exception as e:
                        print(f"[ERROR] Greedy fallback selection failed: {e}")
                        self.selected_indices = np.array([], dtype=int)

            # Wende räumlichen Constraint an, falls aktiviert
            if spatial_constraint and metadata is not None:
                if adaptive_min_distance:
                    self.selected_indices = self._apply_spatial_constraint_adaptive(
                        self.selected_indices,
                        metadata,
                        min_distance_km=min_distance_km,
                        step_km=adaptive_step_km,
                        min_allowed_km=adaptive_min_allowed_km,
                    )
                else:
                    self.selected_indices = self._apply_spatial_constraint(
                        self.selected_indices, metadata, min_distance_km=min_distance_km
                    )

        # Ensure integer dtype for indices to avoid float-indexing issues in downstream code
        if self.selected_indices is None:
            return np.array([], dtype=int)
        self.selected_indices = np.asarray(self.selected_indices, dtype=int)
        return self.selected_indices

    def _augment_features_with_metadata(
        self, features: np.ndarray, metadata: pd.DataFrame, weight: float
    ) -> np.ndarray:
        """
        Erweitert die Feature-Matrix mit normalisierten Metadaten.

        Args:
            features: Ursprüngliche Features
            metadata: DataFrame mit year Spalte
            weight: Gewichtung der Metadaten

        Returns:
            Erweiterte Feature-Matrix

        Note:
            Die Gewichtung wurde verbessert: Temporal-Dimensionen werden
            repliziert (50x), um mit 2048 visuellen Dimensionen zu konkurrieren.
            Effektive Gewichtung: weight * sqrt(50) * features.std()
        """
        if "year" not in metadata.columns:
            return features

        # Normalisiere Jahre auf [0, 1]
        years = metadata["year"].fillna(metadata["year"].median()).values.astype(float)
        denom = years.max() - years.min()
        if denom == 0:
            years_normalized = np.zeros_like(years, dtype=float)
        else:
            years_normalized = (years - years.min()) / denom

        # NEUE STRATEGIE: Repliziere temporal dimension mehrfach
        # Grund: 1 temporal dim vs 2048 visual dims → temporal wird ignoriert
        # Lösung: Repliziere 50x → effektiv ~2.5% der Feature-Dimensionalität
        n_temporal_dims = 50  # Tunable parameter
        years_replicated = np.tile(
            years_normalized.reshape(-1, 1), (1, n_temporal_dims)
        )

        # Skaliere auf Feature-Standardabweichung
        years_weighted = years_replicated * weight * features.std()
        augmented = np.hstack([features, years_weighted])

        return augmented

    def _apply_spatial_constraint(
        self, indices: np.ndarray, metadata: pd.DataFrame, min_distance_km: float = 50.0
    ) -> np.ndarray:
        """
        Wendet räumlichen Sperrfilter auf die Auswahl an.
        Ersetzt zu nahe Samples durch die nächst-diversen Alternativen.

        Diese Version garantiert, wenn möglich, genau `self.n_samples` zurückzugeben:
        - Erst wird versucht, `n_samples` einzusammeln, die den min_distance constraint einhalten.
        - Falls nicht genug valide Samples gefunden werden, werden die nächsten besten
          Kandidaten aus dem vollständigen Ranking hinzugefügt (Constraint wird gelockert).

        Args:
            indices: Ausgewählte Indizes (Top-K Ranking oder Kandidatenliste)
            metadata: DataFrame mit 'N' (Lat) und 'left' (Lon) Spalten
            min_distance_km: Minimale Distanz zwischen Samples

        Returns:
            Bereinigte Indizes (Länge = min(self.n_samples, verfügbare_samples))
        """

        # Verfügbare Kandidaten: falls selector.ranking vorhanden ist, nutze es als vollständiges Ranking
        if (
            hasattr(self, "selector")
            and self.selector is not None
            and hasattr(self.selector, "ranking")
        ):
            candidate_pool = list(self.selector.ranking)
        else:
            candidate_pool = list(indices)

        # Guard: falls metadata keine notwendigen Spalten hat
        if "N" not in metadata.columns or "left" not in metadata.columns:
            raise ValueError(
                "metadata muss 'N' und 'left' Spalten enthalten für räumliche Constraints"
            )

        required = min(self.n_samples, len(candidate_pool))
        valid_indices: list = []

        # 1) First pass: nur solche wählen, die den min_distance Constraint einhalten
        for idx in candidate_pool:
            if len(valid_indices) >= required:
                break

            lat, lon = metadata.loc[idx, "N"], metadata.loc[idx, "left"]
            is_valid = True

            for valid_idx in valid_indices:
                lat2 = metadata.loc[valid_idx, "N"]
                lon2 = metadata.loc[valid_idx, "left"]

                # If projected coordinates are available on metadata, compute Euclidean (metric) distance
                if gdf_metric is not None:
                    a = gdf_metric.loc[idx, ["_proj_x", "_proj_y"]].values.astype(float)
                    b = gdf_metric.loc[valid_idx, ["_proj_x", "_proj_y"]].values.astype(
                        float
                    )
                    distance = float(((a - b) ** 2).sum() ** 0.5) / 1000.0
                else:
                    if processor is None:
                        from dataselector.data.metadata_processor import MetadataProcessor

                        processor = MetadataProcessor("")
                    distance = processor.calculate_spatial_distance(
                        lat, lon, lat2, lon2
                    )

                if distance < min_distance_km:
                    is_valid = False
                    break

            if is_valid:
                valid_indices.append(idx)

        # 2) Falls nicht genug valide Indizes gefunden wurden: fülle mit nächsten Kandidaten
        if len(valid_indices) < required:
            remaining_needed = required - len(valid_indices)
            # Füge Kandidaten hinzu, die noch nicht in valid_indices (Constraint wird gelockert)
            for idx in candidate_pool:
                if len(valid_indices) >= required:
                    break
                if idx in valid_indices:
                    continue
                valid_indices.append(idx)

            print(
                f"⚠️  Warnung: Nur {len(valid_indices)-remaining_needed}/{required} Samples erfüllten den Abstand; "
                f"füge {remaining_needed} Kandidaten nach (Constraint gelockert)."
            )

        return np.array(valid_indices)

    def _apply_spatial_constraint_adaptive(
        self,
        indices: np.ndarray,
        metadata: pd.DataFrame,
        min_distance_km: float = 50.0,
        step_km: float = 5.0,
        min_allowed_km: float = 0.0,
    ) -> np.ndarray:
        """
        Adaptive spatial constraint: reduziert schrittweise `min_distance_km` um `step_km`
        bis `n_samples` gefunden wurden oder `min_allowed_km` erreicht ist.

        Falls nach Reduktion immer noch nicht genug valide Indizes gefunden wurden,
        fällt die Methode auf die standardmäßige Lockerung zurück (fügt verbleibende
        Kandidaten aus dem Ranking hinzu).
        """
        current_min = float(min_distance_km)

        # Try progressively relaxing the constraint
        while current_min >= min_allowed_km:
            valid = self._apply_spatial_constraint(
                indices, metadata, min_distance_km=current_min
            )
            if len(valid) >= min(self.n_samples, len(indices)):
                if current_min != min_distance_km:
                    print(
                        f"ℹ️  adaptive fallback: reduced min_distance from {min_distance_km}km to {current_min}km to reach required samples"
                    )
                return valid[: min(self.n_samples, len(indices))]
            current_min = max(current_min - float(step_km), min_allowed_km)

        # If still not enough, call the standard function which will relax by filling
        print(
            f"⚠️  adaptive fallback: reached min_allowed_km={min_allowed_km}km but still insufficient; falling back to relaxed addition"
        )
        return self._apply_spatial_constraint(
            indices, metadata, min_distance_km=min_allowed_km
        )

    def get_selection_scores(self) -> np.ndarray:
        """
        Gibt die Facility Location Scores zurück.

        Returns:
            Array von Scores für jedes Sample
        """
        if self.selector is None:
            raise ValueError("Selection muss zuerst ausgeführt werden")

        return self.selector.gains_

    def get_coverage_statistics(
        self, features: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict:
        """
        Berechnet Coverage-Statistiken der Auswahl.

        Args:
            features: Vollständige Feature-Matrix
            cluster_labels: Cluster-Zuordnung aller Samples

        Returns:
            Dictionary mit Coverage-Metriken
        """
        if self.selected_indices is None:
            raise ValueError("Selection muss zuerst ausgeführt werden")

        selected_clusters = cluster_labels[self.selected_indices]

        stats = {
            "n_selected": len(self.selected_indices),
            "clusters_covered": len(np.unique(selected_clusters)),
            "cluster_distribution": dict(
                zip(*np.unique(selected_clusters, return_counts=True))
            ),
            "diversity_score": self._calculate_diversity_score(
                features[self.selected_indices]
            ),
        }

        return stats

    def _calculate_diversity_score(self, features: np.ndarray) -> float:
        """
        Berechnet einen Diversitäts-Score basierend auf Paarweisen Distanzen.

        Args:
            features: Feature-Matrix der ausgewählten Samples

        Returns:
            Durchschnittliche paarweise Distanz (höher = diverser)
        """
        from scipy.spatial.distance import pdist

        if len(features) < 2:
            return 0.0

        pairwise_distances = pdist(features, metric=self.metric)
        return float(np.mean(pairwise_distances))

    def export_selection(
        self, metadata: pd.DataFrame, output_path: str
    ) -> pd.DataFrame:
        """
        Exportiert die ausgewählten Samples als CSV.

        Args:
            metadata: Vollständiger Metadaten-DataFrame
            output_path: Pfad für Output-CSV

        Returns:
            DataFrame der ausgewählten Samples
        """
        if self.selected_indices is None:
            raise ValueError("Selection muss zuerst ausgeführt werden")

        selected_df = metadata.iloc[self.selected_indices].copy()
        selected_df["selection_rank"] = range(len(selected_df))

        selected_df.to_csv(output_path, index=False)
        print(f"Auswahl exportiert nach: {output_path}")

        return selected_df

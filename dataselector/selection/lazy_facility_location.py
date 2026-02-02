import heapq
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class LazyFacilityLocationSelection:
    """
    Lazy Greedy (CELF) implementation for Facility Location selection.

    This class implements the CELF optimization to reduce marginal gain
    evaluations while producing the same greedy solution as the standard
    greedy algorithm.
    """

    def __init__(
        self,
        n_samples: int,
        metric: str = "euclidean",
        random_state: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.metric = metric
        self.random_state = random_state

        self.ranking = None
        self.gains_ = None

    def _compute_similarities(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            d = euclidean_distances(X)
            # Convert distances to similarities (higher is better)
            max_d = d.max()
            # Avoid division by zero
            if max_d > 0:
                sims = max_d - d
            else:
                sims = -d
        elif self.metric == "cosine":
            sims = cosine_similarity(X)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        return sims

    def _initial_gain(self, sims: np.ndarray, idx: int) -> float:
        # For Facility Location, initial gain of picking idx first is sum of similarities
        return float(sims[idx].sum())

    def _marginal_gain(
        self,
        sims: np.ndarray,
        idx: int,
        selected: np.ndarray,
        current_max_sims: np.ndarray,
    ) -> float:
        if selected.size == 0:
            return self._initial_gain(sims, idx)

        new_max = np.maximum(current_max_sims, sims[idx])
        gain = float(new_max.sum() - current_max_sims.sum())
        return gain

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        k = min(self.n_samples, n)

        sims = self._compute_similarities(X)

        # Priority queue entries: (-upper_bound_gain, last_updated_round, idx)
        pq = []
        current_max_sims = np.zeros(n)
        selected = []
        gains = np.zeros(n)

        # Initial upper bounds
        for i in range(n):
            g = self._initial_gain(sims, i)
            heapq.heappush(pq, (-g, 0, i))

        round_num = 0
        while len(selected) < k and pq:
            neg_upper, last_round, idx = heapq.heappop(pq)
            if last_round == round_num:
                # This entry is up-to-date
                selected.append(idx)
                gains[idx] = -neg_upper

                # Update current_max_sims
                if len(selected) == 1:
                    current_max_sims = sims[idx].copy()
                else:
                    current_max_sims = np.maximum(current_max_sims, sims[idx])

                round_num += 1
                continue

            # Recompute true marginal gain with respect to current selection
            true_gain = self._marginal_gain(
                sims, idx, np.array(selected), current_max_sims
            )
            # Push back with updated round marker
            heapq.heappush(pq, (-true_gain, round_num, idx))

        self.ranking = np.array(selected, dtype=int)
        self.gains_ = gains
        return self

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.ranking

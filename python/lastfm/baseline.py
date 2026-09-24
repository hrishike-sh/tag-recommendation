"""Popularity-based baseline recommender."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import scipy.sparse as sp


class PopularityRecommender:
    """Popularity recommender based purely on training-period interaction frequency."""

    def __init__(self) -> None:
        self.item_counts: Optional[np.ndarray] = None
        self.n_items: int = 0
        self.popular_item_indices: Optional[np.ndarray] = None

    def fit(self, counts_matrix: Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix]) -> "PopularityRecommender":
        """Fit popularity from training-period counts or interaction matrix.
        
        Parameters
        ----------
        counts_matrix : sparse matrix (n_users x n_items)
            Training interaction counts or binary preferences.
        """
        csr = counts_matrix.tocsr()
        self.n_items = csr.shape[1]
        
        # Sum columns over training users
        # Shape: (n_items,)
        counts = np.asarray(csr.sum(axis=0)).ravel()
        self.item_counts = counts
        # Precompute sorted item indices descending by frequency
        self.popular_item_indices = np.argsort(-counts)
        return self

    def recommend(
        self,
        user_index: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[Union[Sequence[int], Set[int], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-K most popular items, excluding training-seen items."""
        if self.popular_item_indices is None or self.item_counts is None:
            raise RuntimeError("Model is not fitted.")
        if k <= 0:
            return []

        seen_set = set(seen_items) if (exclude_seen and seen_items is not None) else set()

        recs: List[Tuple[int, float]] = []
        for idx in self.popular_item_indices:
            if idx in seen_set:
                continue
            recs.append((int(idx), float(self.item_counts[idx])))
            if len(recs) >= k:
                break

        return recs

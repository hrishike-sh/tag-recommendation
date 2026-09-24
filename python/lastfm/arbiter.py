"""Adaptive Multi-Stream Recommendation Arbiter and Re-Ranking Layer (Phase 8)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD

logger = logging.getLogger(__name__)


def normalize_scores(
    scores: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize score vector per user into [0, 1].

    Parameters
    ----------
    scores : np.ndarray (shape: n_items)
        Raw score vector for a single user.
    method : str, default='minmax'
        Normalization method ('minmax', 'zscore', 'raw').
    """
    if method == "raw":
        return scores

    if method == "minmax":
        # Ignore -inf values from excluded seen items if present
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            return np.zeros_like(scores)
        s_min = np.min(scores[valid_mask])
        s_max = np.max(scores[valid_mask])
        rng = s_max - s_min
        if rng == 0.0:
            return np.zeros_like(scores)
        norm_scores = np.zeros_like(scores)
        norm_scores[valid_mask] = (scores[valid_mask] - s_min) / rng
        norm_scores[~valid_mask] = -np.inf
        return norm_scores

    elif method == "zscore":
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            return np.zeros_like(scores)
        mu = np.mean(scores[valid_mask])
        std = np.std(scores[valid_mask])
        if std == 0.0:
            return np.zeros_like(scores)
        # Standardize and apply sigmoid to map to [0, 1]
        z = (scores[valid_mask] - mu) / std
        norm_scores = np.zeros_like(scores)
        norm_scores[valid_mask] = 1.0 / (1.0 + np.exp(-z))
        norm_scores[~valid_mask] = -np.inf
        return norm_scores

    raise ValueError(f"Unknown normalization method: {method}")


class MultiStreamArbiter:
    """Adaptive Multi-Stream Recommender Arbiter (MSVD + Tags + Temporal)."""

    def __init__(
        self,
        base_msvd: ImplicitMSVD,
        tag_matrix: Optional[sp.csr_matrix] = None,
        temporal_matrix: Optional[sp.csr_matrix] = None,
        weights: Optional[Sequence[float]] = None,
        normalization: str = "minmax",
        gating_mode: str = "global",
        temperature: float = 1.0,
    ) -> None:
        """Initialize MultiStreamArbiter.

        Parameters
        ----------
        base_msvd : ImplicitMSVD
            Fitted base collaborative MSVD model.
        tag_matrix : sp.csr_matrix or None
            L2-normalized item tag matrix (I x T).
        temporal_matrix : sp.csr_matrix or None
            L2-normalized sparse temporal affinity matrix (U x I).
        weights : Sequence of 3 floats or None
            Fixed [w_msvd, w_tag, w_temporal] prior weights summing to 1.
        normalization : str, default='minmax'
            Per-user normalization technique for MSVD and combined streams.
        gating_mode : str, default='global'
            Gating strategy: 'global' (fixed weights), 'user_aware' (activity-scaled), or 'item_aware'.
        temperature : float, default=1.0
            Softmax temperature for gating activations.
        """
        self.base_msvd = base_msvd
        self.tag_matrix = tag_matrix.tocsr().astype(np.float32) if tag_matrix is not None else None
        self.temporal_matrix = temporal_matrix.tocsr().astype(np.float32) if temporal_matrix is not None else None
        
        if weights is not None:
            w_arr = np.array(weights, dtype=np.float32)
            if np.sum(w_arr) > 0:
                w_arr = w_arr / np.sum(w_arr)
            self.weights = w_arr
        else:
            self.weights = np.array([0.5, 0.25, 0.25], dtype=np.float32)

        self.normalization = normalization
        self.gating_mode = gating_mode
        self.temperature = max(1e-4, float(temperature))

        self.user_tag_profiles: Optional[np.ndarray] = None
        self.user_features: Optional[Dict[int, Dict[str, float]]] = None
        self.n_users = base_msvd.n_users
        self.n_items = base_msvd.n_items

    def fit_user_profiles(
        self,
        pref_csr: sp.csr_matrix,
        conf_delta_csr: sp.csr_matrix,
        user_features: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> "MultiStreamArbiter":
        """Compute user tag profiles and cache user gating features."""
        # 1. User Tag Profiles
        if self.tag_matrix is not None:
            n_users = pref_csr.shape[0]
            c_sparse = pref_csr.copy().astype(np.float32)
            c_sparse.data = (1.0 + conf_delta_csr.data.astype(np.float32)) * pref_csr.data.astype(np.float32)
            raw_profiles = c_sparse @ self.tag_matrix
            profiles = raw_profiles.toarray() if sp.issparse(raw_profiles) else np.asarray(raw_profiles)
            norms = np.linalg.norm(profiles, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.user_tag_profiles = (profiles / norms).astype(np.float32)

        # 2. Extract deterministic user gating features
        self.user_features = {}
        for u in range(pref_csr.shape[0]):
            u_nnz = pref_csr.indptr[u + 1] - pref_csr.indptr[u]
            has_tag = 1.0 if (self.user_tag_profiles is not None and np.linalg.norm(self.user_tag_profiles[u]) > 0) else 0.0
            has_temporal = 1.0 if (self.temporal_matrix is not None and (self.temporal_matrix.indptr[u + 1] > self.temporal_matrix.indptr[u])) else 0.0
            
            self.user_features[u] = {
                "log_interactions": math.log(1.0 + u_nnz),
                "has_tag": has_tag,
                "has_temporal": has_temporal,
            }
        return self

    def get_user_weights(self, user_index: int) -> np.ndarray:
        """Compute normalized [w_msvd, w_tag, w_temporal] gating weights for a user."""
        if self.gating_mode == "global" or self.user_features is None:
            return self.weights

        feats = self.user_features.get(user_index, {"log_interactions": 3.0, "has_tag": 1.0, "has_temporal": 1.0})
        log_cnt = feats["log_interactions"]

        # Adaptive heuristics:
        # - Low-activity/sparse users benefit heavily from recency (temporal) and tags
        # - High-activity users with rich collaborative history rely primarily on MSVD
        logits = np.zeros(3, dtype=np.float32)
        logits[0] = 0.5 * log_cnt  # MSVD logit increases with interaction volume
        logits[1] = 2.0 * feats["has_tag"]  # Tag logit if user has tag coverage
        logits[2] = 3.5 * feats["has_temporal"] / (1.0 + 0.3 * log_cnt)  # Temporal logit higher for sparse users

        # Softmax with temperature
        exp_l = np.exp((logits - np.max(logits)) / self.temperature)
        return exp_l / np.sum(exp_l)

    def recommend(
        self,
        user_index: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[Union[Sequence[int], Set[int], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate Top-K recommendations using adaptive multi-stream fusion."""
        if self.base_msvd.user_factors is None or self.base_msvd.item_factors is None:
            raise RuntimeError("Base MSVD is not fitted.")
        if k <= 0:
            return []

        w = self.get_user_weights(user_index)
        w_msvd, w_tag, w_temp = float(w[0]), float(w[1]), float(w[2])

        # 1. Base MSVD scores (normalized)
        u_vec = self.base_msvd.user_factors[user_index]
        s_msvd_raw = u_vec @ self.base_msvd.item_factors.T
        s_msvd = normalize_scores(s_msvd_raw, method=self.normalization)

        combined_scores = w_msvd * s_msvd

        # 2. Tag similarity stream
        if w_tag > 0.0 and self.tag_matrix is not None and self.user_tag_profiles is not None:
            z_u = self.user_tag_profiles[user_index]
            s_tag = self.tag_matrix @ z_u  # Shape: (n_items,), already in [0, 1]
            combined_scores = combined_scores + (w_tag * s_tag)

        # 3. Temporal affinity stream
        if w_temp > 0.0 and self.temporal_matrix is not None and user_index < self.temporal_matrix.shape[0]:
            start = self.temporal_matrix.indptr[user_index]
            end = self.temporal_matrix.indptr[user_index + 1]
            if end > start:
                cols = self.temporal_matrix.indices[start:end]
                data = self.temporal_matrix.data[start:end]
                combined_scores[cols] += w_temp * data

        # 4. Seen item exclusion
        if exclude_seen and seen_items is not None:
            for idx in seen_items:
                if 0 <= idx < self.n_items:
                    combined_scores[idx] = -np.inf

        num_candidates = min(k, self.n_items)
        if num_candidates == 0:
            return []

        if num_candidates < self.n_items:
            candidate_indices = np.argpartition(combined_scores, -num_candidates)[-num_candidates:]
            sorted_indices = candidate_indices[np.argsort(-combined_scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-combined_scores)

        results: List[Tuple[int, float]] = []
        for idx in sorted_indices:
            sc = float(combined_scores[idx])
            if np.isneginf(sc):
                continue
            results.append((int(idx), sc))
            if len(results) >= k:
                break

        return results

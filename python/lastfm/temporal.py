"""Temporal preference and nostalgia scoring extension (Stream B)."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .als import ImplicitMSVD
from .ingest import connect, literal

logger = logging.getLogger(__name__)


def compute_temporal_affinity_matrix(
    root: Path,
    user_ids: Sequence[str],
    item_ids: Sequence[str],
    cutoff_iso: str,
    gamma: float = 0.01,
    mode: str = "recency",
    track_meta_parquet: Optional[Path] = None,
    kappa: float = 40.0,
) -> sp.csr_matrix:
    """Build sparse user-item temporal affinity matrix (U x I) strictly before cutoff.

    Parameters
    ----------
    root : Path
        Project root directory.
    user_ids : Sequence[str]
        Ordered list of user IDs for matrix row alignment.
    item_ids : Sequence[str]
        Ordered list of item IDs for matrix column alignment.
    cutoff_iso : str
        UTC ISO timestamp marking the exclusive upper bound for training history.
    gamma : float, default=0.01
        Temporal decay rate (per day).
    mode : str, default="recency"
        Decay mode: 'recency' (weight decays with age) or 'nostalgia' (weight increases with age).
    track_meta_parquet : Path or None
        Path to track_metadata.parquet for item_id -> artist_id mapping.
    kappa : float, default=40.0
        Confidence scaling parameter.

    Returns
    -------
    S : sp.csr_matrix (shape: n_users x n_items)
        L2-normalized sparse temporal affinity scores for candidate items.
    """
    if mode not in {"recency", "nostalgia"}:
        raise ValueError("mode must be 'recency' or 'nostalgia'")

    dt_cutoff = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    t_ref_epoch = dt_cutoff.timestamp()
    cutoff_lit = f"TIMESTAMPTZ {literal(dt_cutoff.isoformat())}"

    n_users = len(user_ids)
    n_items = len(item_ids)

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    with connect(root) as db:
        # 1. Fetch user-track temporal interaction summaries strictly before cutoff
        # Age in days: (t_ref_epoch - epoch(played_at)) / 86400.0
        query = f"""
            WITH historical_events AS (
                SELECT 
                    user_id,
                    item_id,
                    ( {t_ref_epoch} - epoch(played_at) ) / 86400.0 AS age_days
                FROM recsys.events_1k
                WHERE (year < year({cutoff_lit}) OR (year=year({cutoff_lit}) AND month<=month({cutoff_lit})))
                  AND played_at < {cutoff_lit}
            )
            SELECT 
                user_id,
                item_id,
                count(*)::BIGINT AS play_count,
                avg(age_days)::DOUBLE AS avg_age
            FROM historical_events
            GROUP BY user_id, item_id
        """
        track_rows = db.execute(query).fetchall()

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for uid, iid, play_count, avg_age in track_rows:
        if uid not in user_map or iid not in item_map:
            continue
        u_idx = user_map[uid]
        i_idx = item_map[iid]

        c_ui = 1.0 + kappa * math.log(1.0 + float(play_count))

        if gamma == 0.0:
            w_time = 1.0
        elif mode == "recency":
            w_time = math.exp(-gamma * max(0.0, float(avg_age)))
        else:  # nostalgia
            w_time = 1.0 - math.exp(-gamma * max(0.0, float(avg_age)))

        score = c_ui * w_time
        if score > 0.0:
            rows.append(u_idx)
            cols.append(i_idx)
            vals.append(score)

    if not rows:
        return sp.csr_matrix((n_users, n_items), dtype=np.float32)

    S_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    # L2 normalize each user's temporal score vector
    norm_sq = S_mat.multiply(S_mat).sum(axis=1)
    norm = np.sqrt(np.asarray(norm_sq).ravel())
    norm[norm == 0.0] = 1.0

    inv_norm = sp.diags(1.0 / norm, format="csr")
    S_norm = inv_norm @ S_mat
    return S_norm.astype(np.float32)


class TemporalMSVD:
    """Temporal and Nostalgia-Enhanced Implicit MSVD Recommender (Stream B)."""

    def __init__(
        self,
        base_msvd: ImplicitMSVD,
        temporal_matrix: sp.csr_matrix,
        alpha: float = 0.1,
        tag_matrix: Optional[sp.csr_matrix] = None,
        beta: float = 0.0,
    ) -> None:
        """Initialize Temporal MSVD.

        Parameters
        ----------
        base_msvd : ImplicitMSVD
            Fitted base collaborative MSVD model.
        temporal_matrix : sp.csr_matrix (shape: n_users x n_items)
            L2-normalized sparse temporal affinity matrix.
        alpha : float, default=0.1
            Temporal fusion weight.
        tag_matrix : sp.csr_matrix or None
            Optional L2-normalized item tag matrix for joint Stream A + Stream B fusion.
        beta : float, default=0.0
            Tag fusion weight.
        """
        self.base_msvd = base_msvd
        self.temporal_matrix = temporal_matrix.tocsr().astype(np.float32)
        self.alpha = float(alpha)
        self.tag_matrix = tag_matrix.tocsr().astype(np.float32) if tag_matrix is not None else None
        self.beta = float(beta)

        self.user_tag_profiles: Optional[np.ndarray] = None
        self.n_users = base_msvd.n_users
        self.n_items = base_msvd.n_items

    def fit_user_tag_profiles(
        self,
        pref_csr: sp.csr_matrix,
        conf_delta_csr: sp.csr_matrix,
    ) -> "TemporalMSVD":
        """Compute user semantic tag profiles if tag matrix is present."""
        if self.tag_matrix is None:
            return self

        n_users = pref_csr.shape[0]
        n_tags = self.tag_matrix.shape[1]

        c_sparse = pref_csr.copy().astype(np.float32)
        c_sparse.data = (1.0 + conf_delta_csr.data.astype(np.float32)) * pref_csr.data.astype(np.float32)

        raw_profiles = c_sparse @ self.tag_matrix
        profiles = raw_profiles.toarray() if sp.issparse(raw_profiles) else np.asarray(raw_profiles)

        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_tag_profiles = (profiles / norms).astype(np.float32)
        return self

    def recommend(
        self,
        user_index: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[Union[Sequence[int], Set[int], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-K recommendations fusing MSVD, tags (Stream A), and temporal affinity (Stream B)."""
        if self.base_msvd.user_factors is None or self.base_msvd.item_factors is None:
            raise RuntimeError("Base MSVD model is not fitted.")
        if k <= 0:
            return []

        # 1. Base MSVD scores: x_u @ Y^T (shape: n_items)
        u_vec = self.base_msvd.user_factors[user_index]
        scores = u_vec @ self.base_msvd.item_factors.T

        # 2. Tag similarity: z_u @ T^T
        if self.beta > 0.0 and self.tag_matrix is not None and self.user_tag_profiles is not None:
            z_u = self.user_tag_profiles[user_index]
            tag_sim = self.tag_matrix @ z_u  # Shape: (n_items,)
            scores = scores + (self.beta * tag_sim)

        # 3. Temporal affinity: s_u (sparse row)
        if self.alpha > 0.0 and user_index < self.temporal_matrix.shape[0]:
            start = self.temporal_matrix.indptr[user_index]
            end = self.temporal_matrix.indptr[user_index + 1]
            if end > start:
                cols = self.temporal_matrix.indices[start:end]
                data = self.temporal_matrix.data[start:end]
                scores = scores.copy()
                scores[cols] += self.alpha * data

        # 4. Seen items exclusion
        if exclude_seen and seen_items is not None:
            if not isinstance(scores, np.ndarray) or scores is u_vec @ self.base_msvd.item_factors.T:
                scores = scores.copy()
            for idx in seen_items:
                if 0 <= idx < self.n_items:
                    scores[idx] = -np.inf

        num_candidates = min(k, self.n_items)
        if num_candidates == 0:
            return []

        if num_candidates < self.n_items:
            candidate_indices = np.argpartition(scores, -num_candidates)[-num_candidates:]
            sorted_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        else:
            sorted_indices = np.argsort(-scores)

        results: List[Tuple[int, float]] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if np.isneginf(score):
                continue
            results.append((int(idx), score))
            if len(results) >= k:
                break

        return results
